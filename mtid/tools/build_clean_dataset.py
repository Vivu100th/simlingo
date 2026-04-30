#!/usr/bin/env python3
"""Build a clean symlink view of SimLingo routes from a data-quality audit."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _dreamer_route_from_data_route(route: str) -> Path | None:
    path = Path(route)
    parts = path.parts
    if not parts or parts[0] != "data":
        return None
    return Path("mtid_dreamer", *parts[1:])


def _route_reasons(route: dict[str, Any], args: argparse.Namespace) -> list[str]:
    reasons: list[str] = []
    frames = _as_int(route.get("frames"))
    slow_fraction = _as_float(route.get("slow_speed_fraction_lt_0_5"))
    stuck_fraction = _as_float(route.get("stuck_displacement_fraction_lt_0_05"))
    low_disp_fraction = _as_float(route.get("low_displacement_fraction_lt_0_20"))
    non_driving_fraction = _as_float(route.get("non_driving_lane_fraction"))
    read_errors = _as_int(route.get("read_errors"))

    if frames < args.min_frames:
        reasons.append("too_few_frames")
    if slow_fraction > args.max_slow_fraction:
        reasons.append("too_many_slow_speed_frames")
    if stuck_fraction > args.max_stuck_fraction:
        reasons.append("too_many_stuck_displacements")
    if low_disp_fraction > args.max_low_displacement_fraction:
        reasons.append("too_many_low_displacements")
    if non_driving_fraction > args.max_non_driving_lane_fraction:
        reasons.append("too_many_non_driving_lane_frames")
    if read_errors > args.max_read_errors:
        reasons.append("too_many_read_errors")
    return reasons


def _safe_remove_tree(path: Path, source_root: Path) -> None:
    output_root = path.resolve()
    if output_root == source_root.resolve():
        raise ValueError("Refusing to overwrite the source dataset root.")
    if output_root == Path("/"):
        raise ValueError("Refusing to overwrite filesystem root.")
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _link_dir(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(source.resolve(), destination, target_is_directory=True)


def build_clean_dataset(args: argparse.Namespace) -> dict[str, Any]:
    report_path = Path(args.report)
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    manifest_path = Path(args.manifest)
    report = _read_json(report_path)
    routes = list(report.get("routes_detail", []))

    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    excluded_reasons: Counter[str] = Counter()
    missing_source_routes = 0
    bad_route_paths = 0

    for route in routes:
        route_name = str(route.get("route", ""))
        reasons = _route_reasons(route, args)
        source_route = source_root / route_name
        if not route_name.startswith("data/"):
            reasons.append("bad_route_path")
            bad_route_paths += 1
        if not source_route.exists():
            reasons.append("missing_source_route")
            missing_source_routes += 1

        if reasons:
            excluded_reasons.update(reasons)
            excluded.append(
                {
                    "route": route_name,
                    "frames": _as_int(route.get("frames")),
                    "reasons": reasons,
                    "slow_speed_fraction_lt_0_5": _as_float(route.get("slow_speed_fraction_lt_0_5")),
                    "stuck_displacement_fraction_lt_0_05": _as_float(
                        route.get("stuck_displacement_fraction_lt_0_05")
                    ),
                    "non_driving_lane_fraction": _as_float(route.get("non_driving_lane_fraction")),
                }
            )
        else:
            kept.append(route)

    dreamer_linked = 0
    dreamer_missing = 0
    data_linked = 0

    if not args.dry_run:
        if output_root.exists() or output_root.is_symlink():
            if not args.overwrite:
                raise FileExistsError(f"{output_root} already exists. Pass --overwrite to replace it.")
            _safe_remove_tree(output_root, source_root)
        output_root.mkdir(parents=True, exist_ok=True)

        for route in kept:
            route_name = str(route["route"])
            _link_dir(source_root / route_name, output_root / route_name)
            data_linked += 1

            if args.include_dreamer:
                dreamer_route = _dreamer_route_from_data_route(route_name)
                if dreamer_route is None:
                    dreamer_missing += 1
                    continue
                source_dreamer_route = source_root / dreamer_route
                if not source_dreamer_route.exists():
                    dreamer_missing += 1
                    continue
                _link_dir(source_dreamer_route, output_root / dreamer_route)
                dreamer_linked += 1

    manifest = {
        "report": str(report_path),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "dry_run": args.dry_run,
        "thresholds": {
            "min_frames": args.min_frames,
            "max_slow_fraction": args.max_slow_fraction,
            "max_stuck_fraction": args.max_stuck_fraction,
            "max_low_displacement_fraction": args.max_low_displacement_fraction,
            "max_non_driving_lane_fraction": args.max_non_driving_lane_fraction,
            "max_read_errors": args.max_read_errors,
        },
        "input_routes": len(routes),
        "kept_routes": len(kept),
        "excluded_routes": len(excluded),
        "kept_frames": sum(_as_int(route.get("frames")) for route in kept),
        "excluded_frames": sum(_as_int(route.get("frames")) for route in excluded),
        "data_linked_routes": data_linked,
        "dreamer_linked_routes": dreamer_linked,
        "dreamer_missing_routes": dreamer_missing,
        "missing_source_routes": missing_source_routes,
        "bad_route_paths": bad_route_paths,
        "excluded_reason_counts": dict(sorted(excluded_reasons.items())),
        "kept": [
            {
                "route": route["route"],
                "frames": _as_int(route.get("frames")),
                "slow_speed_fraction_lt_0_5": _as_float(route.get("slow_speed_fraction_lt_0_5")),
                "stuck_displacement_fraction_lt_0_05": _as_float(
                    route.get("stuck_displacement_fraction_lt_0_05")
                ),
                "non_driving_lane_fraction": _as_float(route.get("non_driving_lane_fraction")),
            }
            for route in kept
        ],
        "excluded": excluded,
    }

    if not args.dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        (output_root / "clean_dataset_manifest.json").write_text(json.dumps(manifest, indent=2))

    return manifest


def _print_summary(manifest: dict[str, Any]) -> None:
    print(f"Report: {manifest['report']}")
    print(f"Source root: {manifest['source_root']}")
    print(f"Output root: {manifest['output_root']}")
    print(f"Input routes: {manifest['input_routes']}")
    print(f"Kept routes/frames: {manifest['kept_routes']} / {manifest['kept_frames']}")
    print(f"Excluded routes/frames: {manifest['excluded_routes']} / {manifest['excluded_frames']}")
    print(f"Excluded reasons: {manifest['excluded_reason_counts']}")
    if manifest["dry_run"]:
        print("Dry run only; no symlinks were written.")
    else:
        print(f"Linked data routes: {manifest['data_linked_routes']}")
        print(f"Linked dreamer routes: {manifest['dreamer_linked_routes']}")
        print(f"Missing dreamer routes: {manifest['dreamer_missing_routes']}")
        print(f"Wrote manifest: {manifest['output_root']}/clean_dataset_manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="mtid/outputs/debug/driving_data_quality_report.json")
    parser.add_argument("--source-root", default="database/simlingo_v2_all")
    parser.add_argument("--output-root", default="database/simlingo_v2_all_clean")
    parser.add_argument("--manifest", default="mtid/outputs/debug/clean_dataset_manifest.json")
    parser.add_argument("--min-frames", type=int, default=20)
    parser.add_argument("--max-slow-fraction", type=float, default=0.5)
    parser.add_argument("--max-stuck-fraction", type=float, default=0.5)
    parser.add_argument("--max-low-displacement-fraction", type=float, default=1.0)
    parser.add_argument("--max-non-driving-lane-fraction", type=float, default=0.05)
    parser.add_argument("--max-read-errors", type=int, default=0)
    parser.add_argument("--include-dreamer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = build_clean_dataset(args)
    _print_summary(manifest)
    if not args.dry_run:
        print(f"Wrote manifest: {args.manifest}")


if __name__ == "__main__":
    main()
