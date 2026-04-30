#!/usr/bin/env python3
"""Summarize CARLA control debug lines from benchmark_output.log."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from statistics import mean


LINE_RE = re.compile(
    r"CARLA control "
    r"step=(?P<step>-?\d+) "
    r"speed=(?P<speed>[-+0-9.]+) "
    r"steer=(?P<steer>[-+0-9.]+) "
    r"throttle=(?P<throttle>[-+0-9.]+) "
    r"brake=(?P<brake>[-+0-9.]+) "
    r"(?P<extra>.*?)"
    r"target=(?P<target>.*?) "
    r"pred_route0=(?P<route>.*?) "
    r"pred_wps0=(?P<wps>.*)$"
)
COLLISION_RE = re.compile(
    r"CARLA collision "
    r"step=(?P<step>-?\d+) "
    r"frame=(?P<frame>-?\d+) "
    r"other_id=(?P<other_id>-?\d+) "
    r"other_type=(?P<other_type>\S+) "
    r"intensity=(?P<intensity>[-+0-9.]+)"
)
NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")
EXTRA_NUMBER_RE = re.compile(r"(?P<key>[A-Za-z_]+)=(?P<value>[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?|nan|inf|-inf)")
HAZARD_METRIC_KEYS = (
    "hazard_present",
    "hazard_distance",
    "hazard_longitudinal",
    "hazard_lateral",
    "hazard_speed",
    "hazard_static",
    "model_route_hazard_min",
    "model_route_hazard_dy_at_x",
    "speed_wps_hazard_min",
    "speed_wps_hazard_dy_at_x",
    "planner_route_hazard_min",
    "planner_route_hazard_dy_at_x",
)


def _numbers(text: str) -> list[float]:
    if text in {"None", ""}:
        return []
    return [float(match.group(0)) for match in NUMBER_RE.finditer(text)]


def _extra_numbers(text: str) -> dict[str, float]:
    values = {}
    for match in EXTRA_NUMBER_RE.finditer(text):
        values[match.group("key")] = float(match.group("value"))
    return values


def _norm2(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return math.hypot(values[0], values[1])


def _parse_log(path: Path) -> list[dict[str, float | int | None]]:
    rows: list[dict[str, float | int | None]] = []
    for line in path.read_text(errors="replace").splitlines():
        match = LINE_RE.search(line)
        if not match:
            continue
        target = _numbers(match.group("target"))
        route = _numbers(match.group("route"))
        wps = _numbers(match.group("wps"))
        extra = _extra_numbers(match.group("extra"))
        row = {
            "step": int(match.group("step")),
            "speed": float(match.group("speed")),
            "steer": float(match.group("steer")),
            "throttle": float(match.group("throttle")),
            "brake": float(match.group("brake")),
            "desired_speed": extra.get("desired_speed"),
            "desired_speed_raw": extra.get("desired_speed_raw"),
            "speed_scale": extra.get("speed_scale"),
            "min_desired_speed": extra.get("min_desired_speed"),
            "delta": extra.get("delta"),
            "stuck": extra.get("stuck"),
            "force_move": extra.get("force_move"),
            "compass": extra.get("compass"),
            "world_speed": extra.get("world_speed"),
            "lane_dist": extra.get("lane_dist"),
            "collision_count": extra.get("collision_count"),
            "target_norm": _norm2(target),
            "route0_norm": _norm2(route),
            "wps0_norm": _norm2(wps),
        }
        for key in HAZARD_METRIC_KEYS:
            row[key] = extra.get(key)
        rows.append(row)
    return rows


def _parse_collisions(path: Path) -> list[dict[str, float | int | str]]:
    collisions: list[dict[str, float | int | str]] = []
    for line in path.read_text(errors="replace").splitlines():
        match = COLLISION_RE.search(line)
        if not match:
            continue
        collisions.append(
            {
                "step": int(match.group("step")),
                "frame": int(match.group("frame")),
                "other_id": int(match.group("other_id")),
                "other_type": match.group("other_type"),
                "intensity": float(match.group("intensity")),
            }
        )
    return collisions


def _first_stuck_step(
    rows: list[dict[str, float | int | None]],
    min_samples: int,
    speed_eps: float,
    throttle_min: float,
    brake_max: float,
) -> int | None:
    streak = 0
    first_step = None
    for row in rows:
        is_stuck = (
            abs(float(row["speed"])) <= speed_eps
            and float(row["throttle"]) >= throttle_min
            and float(row["brake"]) <= brake_max
        )
        if is_stuck:
            if streak == 0:
                first_step = int(row["step"])
            streak += 1
            if streak >= min_samples:
                return first_step
        else:
            streak = 0
            first_step = None
    return None


def _fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def _finite_values(rows: list[dict[str, float | int | None]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value):
            values.append(value)
    return values


def _summary_pair(values: list[float]) -> str:
    if not values:
        return "n/a / n/a"
    return f"{_fmt(mean(values))} / {_fmt(min(values))}"


def summarize(path: Path, stuck_samples: int = 6) -> str:
    rows = _parse_log(path)
    if not rows:
        return f"No CARLA control lines found in {path}"

    speeds = [float(row["speed"]) for row in rows]
    throttles = [float(row["throttle"]) for row in rows]
    brakes = [float(row["brake"]) for row in rows]
    wps_norms = [float(row["wps0_norm"]) for row in rows if row["wps0_norm"] is not None]
    target_norms = [float(row["target_norm"]) for row in rows if row["target_norm"] is not None]
    route_norms = [float(row["route0_norm"]) for row in rows if row["route0_norm"] is not None]
    desired_speeds = _finite_values(rows, "desired_speed")
    desired_speeds_raw = _finite_values(rows, "desired_speed_raw")
    speed_scales = _finite_values(rows, "speed_scale")
    min_desired_speeds = _finite_values(rows, "min_desired_speed")
    deltas = _finite_values(rows, "delta")
    stuck_counts = _finite_values(rows, "stuck")
    force_moves = _finite_values(rows, "force_move")
    world_speeds = _finite_values(rows, "world_speed")
    lane_dists = _finite_values(rows, "lane_dist")
    collision_counts = _finite_values(rows, "collision_count")
    hazard_present = [row for row in rows if row.get("hazard_present") == 1.0]
    static_hazards = [row for row in hazard_present if row.get("hazard_static") == 1.0]
    hazard_distances = _finite_values(rows, "hazard_distance")
    hazard_speeds = _finite_values(rows, "hazard_speed")
    model_route_hazard = _finite_values(rows, "model_route_hazard_min")
    speed_wps_hazard = _finite_values(rows, "speed_wps_hazard_min")
    planner_route_hazard = _finite_values(rows, "planner_route_hazard_min")
    model_route_dy_at_x = _finite_values(rows, "model_route_hazard_dy_at_x")
    speed_wps_dy_at_x = _finite_values(rows, "speed_wps_hazard_dy_at_x")
    planner_route_dy_at_x = _finite_values(rows, "planner_route_hazard_dy_at_x")
    static_model_route_hazard = _finite_values(static_hazards, "model_route_hazard_min")
    static_speed_wps_hazard = _finite_values(static_hazards, "speed_wps_hazard_min")
    static_planner_route_hazard = _finite_values(static_hazards, "planner_route_hazard_min")
    collisions = _parse_collisions(path)
    first_stuck = _first_stuck_step(
        rows,
        min_samples=stuck_samples,
        speed_eps=0.05,
        throttle_min=0.5,
        brake_max=0.1,
    )
    max_speed_row = max(rows, key=lambda row: abs(float(row["speed"])))
    last_rows = rows[-min(10, len(rows)) :]
    last_wps = [float(row["wps0_norm"]) for row in last_rows if row["wps0_norm"] is not None]
    full_throttle = sum(1 for value in throttles if value >= 0.99)
    full_brake = sum(1 for value in brakes if value >= 0.99)

    lines = [
        f"Log: {path}",
        f"Control samples: {len(rows)}",
        f"Step range: {rows[0]['step']} -> {rows[-1]['step']}",
        f"Speed mean/max_abs: {_fmt(mean(speeds))} / {_fmt(float(max_speed_row['speed']))} at step {max_speed_row['step']}",
        f"Full throttle samples: {full_throttle}/{len(rows)}",
        f"Full brake samples: {full_brake}/{len(rows)}",
        f"First stuck step: {_fmt(first_stuck)}",
        f"Target distance mean/max: {_fmt(mean(target_norms) if target_norms else None)} / {_fmt(max(target_norms) if target_norms else None)}",
        f"Pred route0 norm mean: {_fmt(mean(route_norms) if route_norms else None)}",
        f"Pred wps0 norm mean: {_fmt(mean(wps_norms) if wps_norms else None)}",
        f"Pred wps0 norm last10 mean: {_fmt(mean(last_wps) if last_wps else None)}",
    ]
    if desired_speeds:
        lines.append(f"PID desired speed mean/max: {_fmt(mean(desired_speeds))} / {_fmt(max(desired_speeds))}")
    if desired_speeds_raw:
        lines.append(f"PID raw desired speed mean/max: {_fmt(mean(desired_speeds_raw))} / {_fmt(max(desired_speeds_raw))}")
    if speed_scales:
        lines.append(f"PID speed scale min/max: {_fmt(min(speed_scales))} / {_fmt(max(speed_scales))}")
    if min_desired_speeds:
        lines.append(f"PID min desired speed max: {_fmt(max(min_desired_speeds))}")
    if deltas:
        lines.append(f"PID delta mean/max: {_fmt(mean(deltas))} / {_fmt(max(deltas))}")
    if stuck_counts:
        lines.append(f"Debug stuck counter max: {_fmt(max(stuck_counts))}")
    if force_moves:
        lines.append(f"Debug force_move max: {_fmt(max(force_moves))}")
    if world_speeds:
        lines.append(f"World speed mean/max: {_fmt(mean(world_speeds))} / {_fmt(max(world_speeds))}")
    if lane_dists:
        lines.append(f"Lane distance mean/max: {_fmt(mean(lane_dists))} / {_fmt(max(lane_dists))}")
    if collision_counts:
        lines.append(f"Debug collision_count max: {_fmt(max(collision_counts))}")
    if hazard_present:
        lines.append(f"Hazard debug samples: {len(hazard_present)}/{len(rows)} static: {len(static_hazards)}")
    if hazard_distances:
        lines.append(f"Hazard distance mean/min: {_fmt(mean(hazard_distances))} / {_fmt(min(hazard_distances))}")
    if hazard_speeds:
        lines.append(f"Hazard speed mean/min: {_fmt(mean(hazard_speeds))} / {_fmt(min(hazard_speeds))}")
    if model_route_hazard:
        lines.append(
            "Model route hazard clearance mean/min: "
            f"{_fmt(mean(model_route_hazard))} / {_fmt(min(model_route_hazard))}"
        )
    if speed_wps_hazard:
        lines.append(
            "Speed wps hazard clearance mean/min: "
            f"{_fmt(mean(speed_wps_hazard))} / {_fmt(min(speed_wps_hazard))}"
        )
    if planner_route_hazard:
        lines.append(
            "Planner route hazard clearance mean/min: "
            f"{_fmt(mean(planner_route_hazard))} / {_fmt(min(planner_route_hazard))}"
        )
    if model_route_dy_at_x:
        lines.append(f"Model route lateral offset at hazard-x mean: {_fmt(mean(model_route_dy_at_x))}")
    if speed_wps_dy_at_x:
        lines.append(f"Speed wps lateral offset at hazard-x mean: {_fmt(mean(speed_wps_dy_at_x))}")
    if planner_route_dy_at_x:
        lines.append(f"Planner route lateral offset at hazard-x mean: {_fmt(mean(planner_route_dy_at_x))}")
    if static_hazards:
        lines.append(
            "Static hazard clearance mean/min "
            f"model_route: {_summary_pair(static_model_route_hazard)} "
            f"speed_wps: {_summary_pair(static_speed_wps_hazard)} "
            f"planner_route: {_summary_pair(static_planner_route_hazard)}"
        )
    if collisions:
        max_collision = max(collisions, key=lambda item: float(item["intensity"]))
        last_collision = collisions[-1]
        lines.append(f"Collision events: {len(collisions)}")
        lines.append(
            "Max collision: "
            f"step {max_collision['step']} other {max_collision['other_type']} "
            f"intensity {_fmt(float(max_collision['intensity']))}"
        )
        lines.append(
            "Last collision: "
            f"step {last_collision['step']} other {last_collision['other_type']} "
            f"intensity {_fmt(float(last_collision['intensity']))}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", nargs="?", default="benchmark_output.log")
    parser.add_argument(
        "--stuck-samples",
        type=int,
        default=6,
        help="Consecutive debug samples required before reporting a stuck step.",
    )
    args = parser.parse_args()
    print(summarize(Path(args.log_path), stuck_samples=args.stuck_samples))


if __name__ == "__main__":
    main()
