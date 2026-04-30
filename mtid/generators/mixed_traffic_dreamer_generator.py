"""
Mixed-Traffic Interaction Dreaming (MTID) label generator.

The generator writes dreamer-style JSON files that can be consumed by the
existing SimLingo `Data_Dreamer` dataloader. It keeps a master copy under
`mtid/outputs/labels` and can mirror labels into the dataset folder under
`<dataset-root>/mtid_dreamer` for training.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mtid.core import (
    actor_constant_velocity,
    actor_crossing_rollout,
    actor_cut_in_rollout,
    actor_dense_flow_rollout,
    actor_summary,
    actor_wrong_way_rollout,
    choose_best_candidate,
    equal_spacing_route,
    finalize_audit,
    heading_conflict,
    make_waypoints_from_measurements,
    new_audit,
    score_candidate,
    transform_waypoints,
    update_audit_counts,
    validate_dreamer_payload,
)


MODE_ACTIONS = {
    "jaywalker_crossing": ["keep", "slow", "brake", "yield"],
    "motorcycle_cut_in": ["keep", "slow", "yield", "nudge_left", "nudge_right"],
    "two_wheeler_filtering": ["keep", "slow", "nudge_left", "nudge_right"],
    "wrong_way_two_wheeler": ["keep", "slow", "brake", "yield", "nudge_left", "nudge_right"],
    "dense_gap_yield": ["keep", "slow", "brake", "yield"],
    "lane_less_corridor": ["keep", "slow", "nudge_left", "nudge_right"],
}


@dataclass(frozen=True)
class MTIDConfig:
    dataset_root: Path
    output_root: Path
    export_folder_name: str
    boxes_glob: str
    templates_path: Path
    random_seed: int = 42
    random_subset_count: int = 50
    sample_uniform_interval: int = 1
    future_steps: int = 10
    future_horizon_s: float = 2.5
    max_actor_distance_m: float = 30.0
    overwrite: bool = False
    mirror_to_dataset: bool = True
    audit_only: bool = False

    @property
    def dt(self) -> float:
        return self.future_horizon_s / max(self.future_steps, 1)


def load_json_gz(path: Path) -> Any:
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def write_json_gz(path: Path, payload: Any, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        json.dump(make_json_safe(payload), handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(make_json_safe(payload), handle, indent=2)


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [make_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        value_f = float(value)
        return None if not np.isfinite(value_f) else value_f
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def discover_boxes(cfg: MTIDConfig) -> list[Path]:
    paths = [Path(path) for path in cfg.dataset_root.glob(cfg.boxes_glob)]
    paths.sort()
    if cfg.sample_uniform_interval > 1:
        paths = paths[:: cfg.sample_uniform_interval]
    if cfg.random_subset_count > 0:
        rng = random.Random(cfg.random_seed)
        rng.shuffle(paths)
        paths = paths[: cfg.random_subset_count]
        paths.sort()
    return paths


def frame_paths(path_boxes: Path) -> tuple[Path, Path]:
    path_measurement = Path(str(path_boxes).replace("/boxes/", "/measurements/"))
    path_rgb = Path(str(path_boxes).replace("/boxes/", "/rgb/").replace(".json.gz", ".jpg"))
    return path_measurement, path_rgb


def future_measurement_paths(path_measurement: Path, steps: int) -> list[Path]:
    stem = path_measurement.name.replace(".json.gz", "")
    try:
        frame_idx = int(stem)
    except ValueError:
        return []
    return [
        path_measurement.with_name(f"{frame_idx + offset:04d}.json.gz")
        for offset in range(1, steps + 1)
    ]


def mirrored_label_path(path_boxes: Path, dataset_root: Path, folder_name: str) -> Path:
    rel_parts = list(path_boxes.relative_to(dataset_root).parts)
    rel_parts = [folder_name if part in {"data", "boxes"} else part for part in rel_parts]
    return dataset_root.joinpath(*rel_parts)


def master_label_path(path_boxes: Path, dataset_root: Path, output_root: Path, folder_name: str) -> Path:
    rel_parts = list(path_boxes.relative_to(dataset_root).parts)
    rel_parts = [folder_name if part in {"data", "boxes"} else part for part in rel_parts]
    return output_root.joinpath(*rel_parts)


def load_templates(path: Path) -> dict[str, Any]:
    with path.open("r") as handle:
        return json.load(handle)


def actor_side(actor: dict[str, Any] | None) -> str:
    if actor is None:
        return "front"
    y = float(actor["position"][1])
    if y > 0.5:
        return "left"
    if y < -0.5:
        return "right"
    return "front"


def actor_label(actor: dict[str, Any] | None) -> str:
    if actor is None:
        return "road user"
    actor_type = actor.get("actor_type")
    type_id = str(actor.get("type_id") or "").lower()
    if actor_type == "pedestrian":
        return "pedestrian"
    if actor_type == "two_wheeler":
        if "diamondback" in type_id or "bicycle" in type_id:
            return "bicycle"
        if "motorcycle" in type_id or "yamaha" in type_id or "harley" in type_id or "vespa" in type_id:
            return "motorcycle"
        return "two-wheeler"
    if actor_type == "vehicle":
        return "vehicle"
    return "road user"


def template_placeholders(actor: dict[str, Any] | None) -> dict[str, str]:
    return {
        "<SIDE>": actor_side(actor),
        "<ACTOR>": actor_label(actor),
    }


def choose_instruction(
    templates: dict[str, Any],
    mode: str,
    actor: dict[str, Any] | None,
    unsafe: bool,
    rng: random.Random,
) -> tuple[str, str, dict[str, str]]:
    mode_templates = templates.get(mode, {})
    key = "unsafe_instructions" if unsafe else "instructions"
    candidates = mode_templates.get(key) or mode_templates.get("instructions") or [mode.replace("_", " ")]
    template = rng.choice(candidates)
    placeholders = template_placeholders(actor)
    instruction = template
    for placeholder, value in placeholders.items():
        instruction = instruction.replace(placeholder, value)
    return instruction, template, placeholders


def choose_unsafe_answer(
    templates: dict[str, Any],
    mode: str,
    actor: dict[str, Any] | None,
    rng: random.Random,
) -> str:
    answers = templates.get(mode, {}).get("unsafe_answers") or [
        "Ignore instruction as it violates the mixed-traffic safety threshold. Waypoints:"
    ]
    answer = rng.choice(answers)
    for placeholder, value in template_placeholders(actor).items():
        answer = answer.replace(placeholder, value)
    return answer


def route_for_action(base_route: np.ndarray, action: str) -> np.ndarray:
    if action in {"nudge_left", "nudge_right"}:
        return transform_waypoints(base_route, action)
    return base_route


def candidate_pool(
    mode: str,
    base_waypoints: np.ndarray,
    base_route: np.ndarray,
    actor_rollouts: list[tuple[dict[str, Any], np.ndarray]],
    dt: float,
) -> list[tuple[str, np.ndarray, np.ndarray, dict[str, Any]]]:
    candidates = []
    for action in MODE_ACTIONS[mode]:
        waypoints = transform_waypoints(base_waypoints, action)
        route = route_for_action(base_route, action)
        score = score_candidate(waypoints, actor_rollouts, dt)
        score["candidate_action"] = action
        candidates.append((action, waypoints, route, score))
    return candidates


def actor_rollout_summaries(
    actor_rollouts: list[tuple[dict[str, Any], np.ndarray]],
) -> list[dict[str, Any]]:
    summaries = []
    for actor, rollout in actor_rollouts:
        summaries.append(
            {
                "actor_id": actor.get("id"),
                "actor_type": actor.get("actor_type"),
                "actor_class": actor.get("class"),
                "actor_type_id": actor.get("type_id"),
                "current_position": actor.get("position"),
                "trajectory": rollout,
            }
        )
    return summaries


def make_option(
    mode: str,
    action: str,
    waypoints: np.ndarray,
    route: np.ndarray,
    score: dict[str, Any],
    actor: dict[str, Any] | None,
    path_rgb: Path,
    templates: dict[str, Any],
    rng: random.Random,
    unsafe: bool,
    scenario_source: str,
    actor_rollouts: list[tuple[dict[str, Any], np.ndarray]] | None = None,
) -> dict[str, Any]:
    instruction, template, placeholders = choose_instruction(templates, mode, actor, unsafe, rng)
    safe_to_execute = not unsafe
    if unsafe:
        answer = choose_unsafe_answer(templates, mode, actor, rng)
        allowed = False
    else:
        answer = "Following the given instruction. Waypoints:"
        allowed = True

    actor_ids = []
    if actor is not None and actor.get("id") is not None:
        actor_ids.append(actor.get("id"))
    actor_info = {}
    if actor is not None:
        actor_info = {
            "actor_position": actor.get("position"),
            "actor_speed": actor.get("speed"),
            "actor_yaw": actor.get("yaw"),
            "actor_distance": actor.get("distance"),
            "actor_class": actor.get("class"),
            "actor_type_id": actor.get("type_id"),
            "risk_actor_id": score.get("actor_id"),
            "risk_actor_type": score.get("actor_type"),
        }

    return {
        "waypoints": waypoints,
        "route": route,
        "rgb_path": str(path_rgb),
        "allowed": allowed,
        "mode": mode,
        "info": {
            "allowed": allowed,
            "mode": mode,
            "candidate_action": action,
            "actor_ids": actor_ids,
            "actor_type": actor.get("actor_type") if actor is not None else None,
            "ttc": score.get("ttc"),
            "min_distance": score.get("min_distance"),
            "risk_score": score.get("risk_score"),
            "scenario_source": scenario_source,
            "safe_by_risk": score.get("safe"),
            "actor_rollouts": actor_rollout_summaries(actor_rollouts or []),
            **actor_info,
        },
        "route_reasoning": route_reasoning(mode, action, score),
        "dreamer_instruction": [instruction],
        "instructions_templates": [template],
        "templates_placeholders": [placeholders],
        "dreamer_answer_safety": answer,
        "safe_to_execute": safe_to_execute,
    }


def route_reasoning(mode: str, action: str, score: dict[str, Any]) -> str:
    ttc = score.get("ttc")
    if ttc is None or not np.isfinite(ttc):
        ttc_text = "no finite TTC"
    else:
        ttc_text = f"TTC {ttc:.2f}s"
    min_dist = score.get("min_distance")
    min_dist_text = "unknown distance" if min_dist is None else f"min distance {min_dist:.2f}m"
    return (
        f"MTID {mode} uses candidate action '{action}' with {min_dist_text}, "
        f"{ttc_text}, and risk score {score.get('risk_score', 0.0):.3f}."
    )


def actors_by_type(actors: list[dict[str, Any]], max_distance: float) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {"pedestrian": [], "two_wheeler": [], "vehicle": []}
    for actor in actors:
        if float(actor["distance"]) > max_distance:
            continue
        grouped.setdefault(actor["actor_type"], []).append(actor)
    for values in grouped.values():
        values.sort(key=lambda actor: abs(float(actor["position"][0])) + abs(float(actor["position"][1])))
    return grouped


def relevant_front(actor: dict[str, Any], max_x: float = 28.0) -> bool:
    x, y = actor["position"]
    return -3.0 <= float(x) <= max_x and abs(float(y)) <= 9.0


def build_mode_specs(
    actors: list[dict[str, Any]],
    cfg: MTIDConfig,
) -> list[tuple[str, dict[str, Any] | None, list[tuple[dict[str, Any], np.ndarray]], str]]:
    grouped = actors_by_type(actors, cfg.max_actor_distance_m)
    specs: list[tuple[str, dict[str, Any] | None, list[tuple[dict[str, Any], np.ndarray]], str]] = []

    pedestrians = [actor for actor in grouped["pedestrian"] if relevant_front(actor, 25.0)]
    if pedestrians:
        actor = pedestrians[0]
        specs.append(
            (
                "jaywalker_crossing",
                actor,
                [(actor, actor_crossing_rollout(actor, cfg.future_steps, cfg.dt))],
                "real_pedestrian_counterfactual_crossing",
            )
        )

    two_wheelers = [actor for actor in grouped["two_wheeler"] if relevant_front(actor, 30.0)]
    if two_wheelers:
        actor = two_wheelers[0]
        specs.append(
            (
                "motorcycle_cut_in",
                actor,
                [(actor, actor_cut_in_rollout(actor, cfg.future_steps, cfg.dt))],
                "real_two_wheeler_counterfactual_cut_in",
            )
        )
        specs.append(
            (
                "two_wheeler_filtering",
                actor,
                [(actor, actor_cut_in_rollout(actor, cfg.future_steps, cfg.dt))],
                "real_two_wheeler_filtering_corridor",
            )
        )
        wrong_way_candidates = [
            item
            for item in two_wheelers
            if item.get("same_direction_as_ego") is False or heading_conflict(item) > 0.5
        ]
        if wrong_way_candidates:
            actor_wrong = wrong_way_candidates[0]
            specs.append(
                (
                    "wrong_way_two_wheeler",
                    actor_wrong,
                    [(actor_wrong, actor_wrong_way_rollout(actor_wrong, cfg.future_steps, cfg.dt))],
                    "real_two_wheeler_wrong_way",
                )
            )

    vehicles = [actor for actor in grouped["vehicle"] if relevant_front(actor, 30.0)]
    if len(vehicles) >= 2:
        rollouts = [(actor, actor_dense_flow_rollout(actor, cfg.future_steps, cfg.dt)) for actor in vehicles[:4]]
        specs.append(("dense_gap_yield", vehicles[0], rollouts, "real_vehicle_dense_gap"))

    nearby_actors = [actor for actor in actors if relevant_front(actor, 25.0)]
    if len(nearby_actors) >= 2:
        rollouts = [
            (actor, actor_constant_velocity(actor, cfg.future_steps, cfg.dt))
            for actor in nearby_actors[:5]
        ]
        specs.append(("lane_less_corridor", nearby_actors[0], rollouts, "road_user_density_corridor_proxy"))

    return specs


def generate_payload_for_frame(
    path_boxes: Path,
    cfg: MTIDConfig,
    templates: dict[str, Any],
    rng: random.Random,
) -> tuple[dict[str, Any] | None, str | None]:
    path_measurement, path_rgb = frame_paths(path_boxes)
    if not path_measurement.exists() or not path_rgb.exists():
        return None, "missing_measurement_or_rgb"

    future_paths = future_measurement_paths(path_measurement, cfg.future_steps)
    if any(not path.exists() for path in future_paths):
        return None, "missing_future_measurement"

    boxes = load_json_gz(path_boxes)
    measurement = load_json_gz(path_measurement)
    future_measurements = [load_json_gz(path) for path in future_paths]
    actors = [summary for box in boxes if (summary := actor_summary(box)) is not None]
    if not actors:
        return None, "no_supported_actors"

    base_waypoints = make_waypoints_from_measurements(measurement, future_measurements, cfg.future_steps)
    base_route = equal_spacing_route(measurement.get("route") or measurement.get("route_original"))
    specs = build_mode_specs(actors, cfg)
    if not specs:
        return None, "no_matching_mtid_mode"

    payload: dict[str, list[dict[str, Any]]] = {}
    for mode, actor, actor_rollouts, source in specs:
        candidates = candidate_pool(mode, base_waypoints, base_route, actor_rollouts, cfg.dt)
        best_action, best_waypoints, best_score = choose_best_candidate(
            [(action, waypoints, score) for action, waypoints, _route, score in candidates]
        )
        best_route = next(route for action, _waypoints, route, _score in candidates if action == best_action)
        options = [
            make_option(
                mode,
                best_action,
                best_waypoints,
                best_route,
                best_score,
                actor,
                path_rgb,
                templates,
                rng,
                unsafe=False,
                scenario_source=source,
                actor_rollouts=actor_rollouts,
            )
        ]

        unsafe_candidates = [item for item in candidates if not item[3]["safe"]]
        if unsafe_candidates:
            unsafe_action, unsafe_waypoints, unsafe_route, unsafe_score = max(
                unsafe_candidates,
                key=lambda item: item[3]["risk_score"],
            )
            options.append(
                make_option(
                    mode,
                    unsafe_action,
                    unsafe_waypoints,
                    unsafe_route,
                    unsafe_score,
                    actor,
                    path_rgb,
                    templates,
                    rng,
                    unsafe=True,
                    scenario_source=source,
                    actor_rollouts=actor_rollouts,
                )
            )

        payload[mode] = options

    errors = validate_dreamer_payload(payload)
    if errors:
        return None, "schema_error:" + "; ".join(errors[:3])
    return payload, None


def audit_dataset(box_paths: list[Path], cfg: MTIDConfig) -> dict[str, Any]:
    audit = new_audit()
    for path_boxes in box_paths:
        try:
            boxes = load_json_gz(path_boxes)
        except Exception:
            audit["skip_reasons"]["box_load_error"] += 1
            continue
        update_audit_counts(audit, path_boxes, boxes, cfg.dataset_root)
    return audit


def generate_labels(cfg: MTIDConfig) -> dict[str, Any]:
    rng = random.Random(cfg.random_seed)
    templates = load_templates(cfg.templates_path)
    box_paths = discover_boxes(cfg)
    audit = audit_dataset(box_paths, cfg)

    if cfg.audit_only:
        return finalize_audit(audit)

    for path_boxes in box_paths:
        payload, skip_reason = generate_payload_for_frame(path_boxes, cfg, templates, rng)
        if skip_reason is not None:
            audit["skip_reasons"][skip_reason] += 1
            continue
        assert payload is not None
        audit["generated_frames"] += 1
        for mode, options in payload.items():
            audit["mode_counts"][mode] += len(options)
            audit["generated_options"] += len(options)

        master_path = master_label_path(path_boxes, cfg.dataset_root, cfg.output_root, cfg.export_folder_name)
        write_json_gz(master_path, payload, cfg.overwrite)
        if cfg.mirror_to_dataset:
            export_path = mirrored_label_path(path_boxes, cfg.dataset_root, cfg.export_folder_name)
            write_json_gz(export_path, payload, cfg.overwrite)

    return finalize_audit(audit)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MTID dreamer labels.")
    parser.add_argument("--dataset-root", default="database/simlingo_v2_all")
    parser.add_argument("--output-root", default="mtid/outputs/labels")
    parser.add_argument("--debug-root", default="mtid/outputs/debug")
    parser.add_argument("--export-folder-name", default="mtid_dreamer")
    parser.add_argument(
        "--boxes-glob",
        default="data/simlingo/*/*/*/Town*/boxes/*.json.gz",
        help="Glob below dataset root for box files.",
    )
    parser.add_argument("--templates-path", default="mtid/templates/mixed_traffic_dreamer.json")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--random-subset-count", type=int, default=50)
    parser.add_argument("--sample-uniform-interval", type=int, default=1)
    parser.add_argument("--future-steps", type=int, default=10)
    parser.add_argument("--future-horizon-s", type=float, default=2.5)
    parser.add_argument("--max-actor-distance-m", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-mirror-to-dataset", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = MTIDConfig(
        dataset_root=Path(args.dataset_root),
        output_root=Path(args.output_root),
        export_folder_name=args.export_folder_name,
        boxes_glob=args.boxes_glob,
        templates_path=Path(args.templates_path),
        random_seed=args.random_seed,
        random_subset_count=args.random_subset_count,
        sample_uniform_interval=args.sample_uniform_interval,
        future_steps=args.future_steps,
        future_horizon_s=args.future_horizon_s,
        max_actor_distance_m=args.max_actor_distance_m,
        overwrite=args.overwrite,
        mirror_to_dataset=not args.no_mirror_to_dataset,
        audit_only=args.audit_only,
    )
    audit = generate_labels(cfg)
    debug_root = Path(args.debug_root)
    audit_path = debug_root / "mtid_audit_report.json"
    write_json(audit_path, audit)
    print(f"Scanned {audit['frames']} frames.")
    print(f"Generated {audit.get('generated_frames', 0)} label files.")
    print(f"Generated {audit.get('generated_options', 0)} options.")
    print(f"Wrote audit report to {audit_path}")


if __name__ == "__main__":
    main()
