#!/usr/bin/env python3
"""Audit base driving data quality for slow, stuck, and off-road signals."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


def _read_json(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def _norm2(value: Any) -> float | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return math.hypot(float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return None


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
    return values[idx]


def _summary(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": mean(values) if values else None,
        "p10": _quantile(values, 0.10),
        "p50": _quantile(values, 0.50),
        "p90": _quantile(values, 0.90),
        "max": max(values) if values else None,
    }


def _safe_div(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def _route_dirs(data_root: Path) -> list[Path]:
    route_dirs: set[Path] = set()
    for dirpath, dirnames, _ in os.walk(data_root, followlinks=True):
        if "measurements" in dirnames:
            route_dirs.add(Path(dirpath))
    return sorted(route_dirs)


def _frame_id(path: Path) -> str:
    name = path.name
    if name.endswith(".json.gz"):
        return name[: -len(".json.gz")]
    if name.endswith(".json"):
        return name[: -len(".json")]
    return path.stem


def _measurement_files(route_dir: Path) -> list[Path]:
    files = list((route_dir / "measurements").glob("*.json"))
    files.extend((route_dir / "measurements").glob("*.json.gz"))
    return sorted(files, key=lambda path: _frame_id(path))


def _read_lane_type(route_dir: Path, frame_id: str) -> str | None:
    boxes_dir = route_dir / "boxes"
    candidates = [boxes_dir / f"{frame_id}.json.gz", boxes_dir / f"{frame_id}.json"]
    for path in candidates:
        if not path.exists():
            continue
        try:
            boxes = _read_json(path)
        except Exception:
            return "read_error"
        if not isinstance(boxes, list):
            return None
        for item in boxes:
            if isinstance(item, dict) and item.get("class") == "ego_info":
                return item.get("lane_type_str") or item.get("lane_type") or "unknown"
    return None


def _audit_route(route_dir: Path, root: Path, include_boxes: bool) -> dict[str, Any]:
    files = _measurement_files(route_dir)
    speeds: list[float] = []
    positions: list[tuple[float, float]] = []
    target_norms: list[float] = []
    route_first_norms: list[float] = []
    commands: Counter[str] = Counter()
    lane_types: Counter[str] = Counter()

    read_errors = 0
    for path in files:
        try:
            data = _read_json(path)
        except Exception:
            read_errors += 1
            continue
        if not isinstance(data, dict):
            read_errors += 1
            continue
        try:
            speeds.append(float(data.get("speed", 0.0)))
        except (TypeError, ValueError):
            pass
        pos = data.get("pos_global")
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            try:
                positions.append((float(pos[0]), float(pos[1])))
            except (TypeError, ValueError):
                pass
        target_norm = _norm2(data.get("target_point"))
        if target_norm is not None:
            target_norms.append(target_norm)
        route = data.get("route")
        if isinstance(route, list) and route:
            route_norm = _norm2(route[0])
            if route_norm is not None:
                route_first_norms.append(route_norm)
        commands[str(data.get("command", "unknown"))] += 1
        if include_boxes:
            lane_type = _read_lane_type(route_dir, _frame_id(path))
            if lane_type:
                lane_types[str(lane_type)] += 1

    displacements = [
        math.hypot(positions[idx][0] - positions[idx - 1][0], positions[idx][1] - positions[idx - 1][1])
        for idx in range(1, len(positions))
    ]
    frame_count = len(speeds)
    slow_01 = sum(1 for value in speeds if value < 0.1)
    slow_05 = sum(1 for value in speeds if value < 0.5)
    slow_10 = sum(1 for value in speeds if value < 1.0)
    stuck_disp = sum(1 for value in displacements if value < 0.05)
    low_disp = sum(1 for value in displacements if value < 0.20)
    non_driving = sum(count for lane, count in lane_types.items() if lane not in {"Driving", "None"})

    return {
        "route": str(route_dir.relative_to(root)),
        "frames": frame_count,
        "read_errors": read_errors,
        "speed": _summary(speeds),
        "displacement_per_sample": _summary(displacements),
        "target_point_norm": _summary(target_norms),
        "route_first_point_norm": _summary(route_first_norms),
        "slow_speed_fraction_lt_0_1": _safe_div(slow_01, frame_count),
        "slow_speed_fraction_lt_0_5": _safe_div(slow_05, frame_count),
        "slow_speed_fraction_lt_1_0": _safe_div(slow_10, frame_count),
        "stuck_displacement_fraction_lt_0_05": _safe_div(stuck_disp, len(displacements)),
        "low_displacement_fraction_lt_0_20": _safe_div(low_disp, len(displacements)),
        "non_driving_lane_fraction": _safe_div(non_driving, sum(lane_types.values())),
        "lane_types": dict(lane_types),
        "commands": dict(commands),
    }


def audit_dataset(
    dataset_root: Path,
    max_routes: int | None,
    include_boxes: bool,
) -> dict[str, Any]:
    data_root = dataset_root / "data"
    routes = _route_dirs(data_root)
    if max_routes is not None and max_routes > 0:
        routes = routes[:max_routes]

    route_reports = [_audit_route(route, dataset_root, include_boxes) for route in routes]
    speeds: list[float] = []
    displacements: list[float] = []
    target_norms: list[float] = []
    route_first_norms: list[float] = []
    lane_types: Counter[str] = Counter()
    commands: Counter[str] = Counter()
    frames = 0
    slow_01 = slow_05 = slow_10 = stuck_disp = low_disp = displacement_count = 0
    non_driving = lane_count = 0

    for report in route_reports:
        frame_count = int(report["frames"])
        frames += frame_count
        speed_mean = report["speed"]["mean"]
        if speed_mean is not None:
            speeds.append(float(speed_mean))
        disp_mean = report["displacement_per_sample"]["mean"]
        if disp_mean is not None:
            displacements.append(float(disp_mean))
        target_mean = report["target_point_norm"]["mean"]
        if target_mean is not None:
            target_norms.append(float(target_mean))
        route_first_mean = report["route_first_point_norm"]["mean"]
        if route_first_mean is not None:
            route_first_norms.append(float(route_first_mean))
        slow_01 += round(float(report["slow_speed_fraction_lt_0_1"]) * frame_count)
        slow_05 += round(float(report["slow_speed_fraction_lt_0_5"]) * frame_count)
        slow_10 += round(float(report["slow_speed_fraction_lt_1_0"]) * frame_count)
        lane_types.update(report["lane_types"])
        commands.update(report["commands"])
        lane_count += sum(report["lane_types"].values())
        non_driving += round(float(report["non_driving_lane_fraction"]) * sum(report["lane_types"].values()))

        disp_den = max(0, frame_count - 1)
        displacement_count += disp_den
        stuck_disp += round(float(report["stuck_displacement_fraction_lt_0_05"]) * disp_den)
        low_disp += round(float(report["low_displacement_fraction_lt_0_20"]) * disp_den)

    worst_slow = sorted(
        route_reports,
        key=lambda item: (float(item["slow_speed_fraction_lt_0_5"]), int(item["frames"])),
        reverse=True,
    )[:10]
    worst_stuck = sorted(
        route_reports,
        key=lambda item: (float(item["stuck_displacement_fraction_lt_0_05"]), int(item["frames"])),
        reverse=True,
    )[:10]
    worst_non_driving = sorted(
        route_reports,
        key=lambda item: (float(item["non_driving_lane_fraction"]), int(item["frames"])),
        reverse=True,
    )[:10]

    return {
        "dataset_root": str(dataset_root),
        "routes": len(route_reports),
        "frames": frames,
        "global": {
            "route_mean_speed": _summary(speeds),
            "route_mean_displacement_per_sample": _summary(displacements),
            "route_mean_target_point_norm": _summary(target_norms),
            "route_mean_route_first_point_norm": _summary(route_first_norms),
            "slow_speed_fraction_lt_0_1": _safe_div(slow_01, frames),
            "slow_speed_fraction_lt_0_5": _safe_div(slow_05, frames),
            "slow_speed_fraction_lt_1_0": _safe_div(slow_10, frames),
            "stuck_displacement_fraction_lt_0_05": _safe_div(stuck_disp, displacement_count),
            "low_displacement_fraction_lt_0_20": _safe_div(low_disp, displacement_count),
            "non_driving_lane_fraction": _safe_div(non_driving, lane_count),
            "lane_types": dict(lane_types),
            "commands": dict(commands),
        },
        "worst_routes_by_slow_speed_lt_0_5": worst_slow,
        "worst_routes_by_stuck_displacement_lt_0_05": worst_stuck,
        "worst_routes_by_non_driving_lane": worst_non_driving,
        "routes_detail": route_reports,
    }


def _print_summary(report: dict[str, Any]) -> None:
    global_report = report["global"]
    print(f"Dataset: {report['dataset_root']}")
    print(f"Routes: {report['routes']}")
    print(f"Frames: {report['frames']}")
    print(f"Slow speed <0.1 m/s: {global_report['slow_speed_fraction_lt_0_1']:.3f}")
    print(f"Slow speed <0.5 m/s: {global_report['slow_speed_fraction_lt_0_5']:.3f}")
    print(f"Slow speed <1.0 m/s: {global_report['slow_speed_fraction_lt_1_0']:.3f}")
    print(f"Stuck displacement <0.05 m/sample: {global_report['stuck_displacement_fraction_lt_0_05']:.3f}")
    print(f"Low displacement <0.20 m/sample: {global_report['low_displacement_fraction_lt_0_20']:.3f}")
    print(f"Non-driving lane fraction: {global_report['non_driving_lane_fraction']:.3f}")
    print(f"Route mean speed: {global_report['route_mean_speed']}")
    print(f"Route mean displacement/sample: {global_report['route_mean_displacement_per_sample']}")
    print("Worst slow routes:")
    for item in report["worst_routes_by_slow_speed_lt_0_5"][:5]:
        print(f"  {item['slow_speed_fraction_lt_0_5']:.3f} {item['frames']:4d} {item['route']}")
    print("Worst stuck-displacement routes:")
    for item in report["worst_routes_by_stuck_displacement_lt_0_05"][:5]:
        print(f"  {item['stuck_displacement_fraction_lt_0_05']:.3f} {item['frames']:4d} {item['route']}")
    print("Worst non-driving-lane routes:")
    for item in report["worst_routes_by_non_driving_lane"][:5]:
        print(f"  {item['non_driving_lane_fraction']:.3f} {item['frames']:4d} {item['route']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="database/simlingo_v2_all")
    parser.add_argument("--max-routes", type=int, default=-1)
    parser.add_argument("--skip-boxes", action="store_true")
    parser.add_argument("--output", default="mtid/outputs/debug/driving_data_quality_report.json")
    args = parser.parse_args()

    report = audit_dataset(
        dataset_root=Path(args.dataset_root),
        max_routes=args.max_routes if args.max_routes > 0 else None,
        include_boxes=not args.skip_boxes,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    _print_summary(report)
    print(f"Wrote report: {output}")


if __name__ == "__main__":
    main()
