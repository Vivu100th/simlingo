"""Synthetic tests for the MTID dreamer-label generator."""

from __future__ import annotations

import gzip
import json
import random
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from mtid.core import validate_dreamer_payload
from mtid.generators.mixed_traffic_dreamer_generator import (
    MTIDConfig,
    discover_boxes,
    frame_paths,
    generate_labels,
    generate_payload_for_frame,
    master_label_path,
    mirrored_label_path,
)


TEMPLATES_PATH = Path("mtid/templates/mixed_traffic_dreamer.json")


def write_json_gz(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        json.dump(payload, handle)


def read_json_gz(path: Path) -> Any:
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def make_measurement(frame_idx: int) -> dict[str, Any]:
    return {
        "pos_global": [float(frame_idx), 0.0],
        "theta": 0.0,
        "speed": 4.0,
        "route": [[float(i), 0.0] for i in range(1, 8)],
        "route_original": [[float(i), 0.0] for i in range(1, 8)],
        "target_point": [10.0, 0.0],
        "target_point_next": [15.0, 0.0],
    }


def build_minimal_dataset(root: Path, future_steps: int = 3) -> Path:
    route = root / "data" / "simlingo" / "routes_training" / "seed_0" / "scenario_0" / "Town01"
    boxes_path = route / "boxes" / "0000.json.gz"
    rgb_path = route / "rgb" / "0000.jpg"
    rgb_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_path.write_bytes(b"synthetic rgb placeholder")

    write_json_gz(
        boxes_path,
        [
            {
                "id": 101,
                "class": "walker",
                "position": [8.0, 2.5, 0.0],
                "speed": 1.0,
                "yaw": 0.0,
                "distance": 8.4,
                "extent": [0.3, 0.3, 0.8],
            }
        ],
    )
    for frame_idx in range(future_steps + 1):
        write_json_gz(route / "measurements" / f"{frame_idx:04d}.json.gz", make_measurement(frame_idx))
    return boxes_path


def make_config(root: Path, output_root: Path, mirror_to_dataset: bool = False) -> MTIDConfig:
    return MTIDConfig(
        dataset_root=root,
        output_root=output_root,
        export_folder_name="mtid_dreamer",
        boxes_glob="data/simlingo/*/*/*/Town*/boxes/*.json.gz",
        templates_path=TEMPLATES_PATH,
        random_seed=7,
        random_subset_count=-1,
        future_steps=3,
        future_horizon_s=0.75,
        overwrite=True,
        mirror_to_dataset=mirror_to_dataset,
    )


class GeneratorPathTests(unittest.TestCase):
    def test_frame_and_export_paths_follow_dataloader_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            boxes_path = build_minimal_dataset(root)
            measurement_path, rgb_path = frame_paths(boxes_path)

            self.assertEqual(measurement_path.parent.name, "measurements")
            self.assertEqual(rgb_path.parent.name, "rgb")
            self.assertEqual(rgb_path.suffix, ".jpg")

            mirror = mirrored_label_path(boxes_path, root, "mtid_dreamer")
            self.assertEqual(mirror.relative_to(root).parts[0], "mtid_dreamer")
            self.assertEqual(mirror.parent.name, "mtid_dreamer")

            master = master_label_path(boxes_path, root, root / "labels", "mtid_dreamer")
            self.assertEqual(master.relative_to(root / "labels").parts[0], "mtid_dreamer")
            self.assertEqual(master.parent.name, "mtid_dreamer")

    def test_discover_boxes_returns_sorted_dataset_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            boxes_path = build_minimal_dataset(root)
            cfg = make_config(root, root / "labels")

            self.assertEqual(discover_boxes(cfg), [boxes_path])


class GeneratorPayloadTests(unittest.TestCase):
    def test_generate_payload_for_frame_creates_valid_jaywalker_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            boxes_path = build_minimal_dataset(root)
            cfg = make_config(root, root / "labels")
            templates = json.loads(TEMPLATES_PATH.read_text())

            payload, skip_reason = generate_payload_for_frame(boxes_path, cfg, templates, random.Random(7))

            self.assertIsNone(skip_reason)
            self.assertIsNotNone(payload)
            assert payload is not None
            self.assertEqual(validate_dreamer_payload(payload), [])
            self.assertIn("jaywalker_crossing", payload)

            option = payload["jaywalker_crossing"][0]
            self.assertTrue(option["allowed"])
            self.assertTrue(option["safe_to_execute"])
            self.assertEqual(option["info"]["actor_ids"], [101])
            self.assertEqual(option["info"]["actor_type"], "pedestrian")
            self.assertEqual(np.asarray(option["waypoints"]).shape, (3, 2))
            self.assertGreater(len(option["dreamer_instruction"][0]), 0)

    def test_generate_payload_reports_missing_future_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            boxes_path = build_minimal_dataset(root, future_steps=2)
            cfg = make_config(root, root / "labels")
            templates = json.loads(TEMPLATES_PATH.read_text())

            payload, skip_reason = generate_payload_for_frame(boxes_path, cfg, templates, random.Random(7))

            self.assertIsNone(payload)
            self.assertEqual(skip_reason, "missing_future_measurement")

    def test_generate_labels_writes_master_and_mirrored_label_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            boxes_path = build_minimal_dataset(root)
            cfg = make_config(root, root / "labels", mirror_to_dataset=True)

            audit = generate_labels(cfg)

            self.assertEqual(audit["frames"], 1)
            self.assertEqual(audit["generated_frames"], 1)
            self.assertGreaterEqual(audit["generated_options"], 1)
            self.assertEqual(audit["mode_counts"]["jaywalker_crossing"], 1)

            master_path = master_label_path(boxes_path, root, cfg.output_root, cfg.export_folder_name)
            mirror_path = mirrored_label_path(boxes_path, root, cfg.export_folder_name)
            self.assertTrue(master_path.exists())
            self.assertTrue(mirror_path.exists())

            payload = read_json_gz(master_path)
            self.assertEqual(validate_dreamer_payload(payload), [])
            self.assertIn("jaywalker_crossing", payload)


if __name__ == "__main__":
    unittest.main()
