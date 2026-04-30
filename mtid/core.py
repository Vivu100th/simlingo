"""Core geometry, rollout, and schema helpers for MTID.

The helpers here deliberately avoid CARLA imports so they can be unit-tested
quickly and used from small dataset generation scripts.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_DREAMER_KEYS = {
    "mode",
    "waypoints",
    "route",
    "rgb_path",
    "allowed",
    "info",
    "route_reasoning",
    "dreamer_instruction",
    "instructions_templates",
    "templates_placeholders",
    "dreamer_answer_safety",
    "safe_to_execute",
}


def as_xy(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.size < 2:
        raise ValueError(f"Expected at least two coordinates, got {value!r}")
    return arr[:2]


def inverse_conversion_2d(point: Any, translation: Any, yaw: float) -> np.ndarray:
    """Convert a global 2D point into an ego-local coordinate frame."""
    point_xy = as_xy(point)
    translation_xy = as_xy(translation)
    rotation = np.array(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
        dtype=float,
    )
    return rotation.T @ (point_xy - translation_xy)


def equal_spacing_route(points: Any, num_points: int = 20) -> np.ndarray:
    """Return route points spaced at one-meter intervals from the ego origin."""
    route = np.asarray(points, dtype=float)
    if route.ndim != 2 or route.shape[1] < 2 or len(route) == 0:
        return np.zeros((num_points, 2), dtype=float)
    route = route[:, :2]
    route = np.concatenate((np.zeros_like(route[:1]), route), axis=0)
    shifted = np.roll(route, 1, axis=0)
    shifted[0] = shifted[1]
    dists = np.linalg.norm(route - shifted, axis=1)
    dists = np.cumsum(dists)
    dists += np.arange(len(dists)) * 1e-4
    samples = np.arange(0, num_points, 1)
    return np.stack(
        [
            np.interp(samples, dists, route[:, 0]),
            np.interp(samples, dists, route[:, 1]),
        ],
        axis=1,
    )


def classify_actor(box: dict[str, Any]) -> str | None:
    cls = box.get("class")
    if cls == "ego_car":
        return None
    if cls == "walker":
        return "pedestrian"
    if cls == "car":
        type_id = str(box.get("type_id") or "").lower()
        wheels = box.get("number_of_wheels")
        try:
            wheel_count = int(float(wheels))
        except (TypeError, ValueError):
            wheel_count = None
        if wheel_count == 2 or "motorcycle" in type_id or "bicycle" in type_id:
            return "two_wheeler"
        return "vehicle"
    return None


def actor_summary(box: dict[str, Any]) -> dict[str, Any] | None:
    actor_type = classify_actor(box)
    if actor_type is None:
        return None
    position = box.get("position")
    if position is None:
        return None
    return {
        "id": box.get("id"),
        "actor_type": actor_type,
        "class": box.get("class"),
        "type_id": box.get("type_id"),
        "number_of_wheels": box.get("number_of_wheels"),
        "position": [float(position[0]), float(position[1])],
        "speed": float(box.get("speed") or 0.0),
        "yaw": float(box.get("yaw") or 0.0),
        "distance": float(box.get("distance") or np.linalg.norm(position[:2])),
        "extent": list(box.get("extent") or [1.0, 0.5, 0.5]),
        "same_direction_as_ego": box.get("same_direction_as_ego"),
        "same_road_as_ego": box.get("same_road_as_ego"),
        "lane_relative_to_ego": box.get("lane_relative_to_ego"),
        "vehicle_cuts_in": box.get("vehicle_cuts_in"),
    }


def make_waypoints_from_measurements(
    current_measurement: dict[str, Any],
    future_measurements: list[dict[str, Any]],
    count: int = 10,
) -> np.ndarray:
    ego_pos = as_xy(current_measurement["pos_global"])
    ego_yaw = float(current_measurement["theta"])
    points: list[np.ndarray] = []
    for measurement in future_measurements[:count]:
        points.append(inverse_conversion_2d(measurement["pos_global"], ego_pos, ego_yaw))
    if not points:
        route = np.asarray(current_measurement.get("route") or [[0.0, 0.0]], dtype=float)
        points = [route[min(i, len(route) - 1), :2] for i in range(count)]
    while len(points) < count:
        points.append(points[-1].copy())
    return np.asarray(points[:count], dtype=float)


def transform_waypoints(waypoints: np.ndarray, action: str) -> np.ndarray:
    """Create a simple counterfactual ego trajectory from expert waypoints."""
    wps = np.asarray(waypoints, dtype=float).copy()
    if action == "keep":
        return wps
    if action == "slow":
        return wps * np.array([0.68, 0.92])
    if action == "brake":
        return wps * np.array([0.28, 0.7])
    if action == "yield":
        return wps * np.array([0.45, 0.85])
    if action == "nudge_left":
        ramp = np.linspace(0.1, 0.9, len(wps))
        wps[:, 1] += ramp
        return wps
    if action == "nudge_right":
        ramp = np.linspace(0.1, 0.9, len(wps))
        wps[:, 1] -= ramp
        return wps
    raise ValueError(f"Unknown candidate action: {action}")


def actor_constant_velocity(actor: dict[str, Any], steps: int, dt: float) -> np.ndarray:
    pos = as_xy(actor["position"])
    yaw = float(actor.get("yaw") or 0.0)
    speed = float(actor.get("speed") or 0.0)
    velocity = np.array([math.cos(yaw), math.sin(yaw)], dtype=float) * speed
    return np.asarray([pos + velocity * dt * (idx + 1) for idx in range(steps)], dtype=float)


def actor_crossing_rollout(actor: dict[str, Any], steps: int, dt: float) -> np.ndarray:
    pos = as_xy(actor["position"])
    side = -1.0 if pos[1] > 0 else 1.0
    forward_speed = max(0.3, min(1.2, float(actor.get("speed") or 0.8) * 0.4))
    lateral_speed = side * 1.4
    velocity = np.array([forward_speed, lateral_speed], dtype=float)
    return np.asarray([pos + velocity * dt * (idx + 1) for idx in range(steps)], dtype=float)


def actor_cut_in_rollout(actor: dict[str, Any], steps: int, dt: float) -> np.ndarray:
    pos = as_xy(actor["position"])
    side = -1.0 if pos[1] > 0 else 1.0
    forward_speed = max(1.0, float(actor.get("speed") or 4.0))
    lateral_speed = side * min(2.0, max(0.8, abs(pos[1]) / max(steps * dt, 0.1)))
    velocity = np.array([forward_speed, lateral_speed], dtype=float)
    return np.asarray([pos + velocity * dt * (idx + 1) for idx in range(steps)], dtype=float)


def actor_wrong_way_rollout(actor: dict[str, Any], steps: int, dt: float) -> np.ndarray:
    pos = as_xy(actor["position"])
    speed = max(2.0, float(actor.get("speed") or 4.0))
    velocity = np.array([-speed, 0.15 * (-1.0 if pos[1] > 0 else 1.0)], dtype=float)
    return np.asarray([pos + velocity * dt * (idx + 1) for idx in range(steps)], dtype=float)


def actor_dense_flow_rollout(actor: dict[str, Any], steps: int, dt: float) -> np.ndarray:
    pos = as_xy(actor["position"])
    speed = max(0.0, float(actor.get("speed") or 0.0))
    velocity = np.array([max(1.0, speed), 0.0], dtype=float)
    return np.asarray([pos + velocity * dt * (idx + 1) for idx in range(steps)], dtype=float)


def min_distance(ego_traj: np.ndarray, actor_traj: np.ndarray) -> float:
    length = min(len(ego_traj), len(actor_traj))
    if length == 0:
        return float("inf")
    return float(np.linalg.norm(ego_traj[:length] - actor_traj[:length], axis=1).min())


def approximate_ttc(
    ego_traj: np.ndarray,
    actor_traj: np.ndarray,
    dt: float,
    conflict_distance: float = 4.0,
) -> float:
    """Estimate when two trajectories first enter the same conflict region."""
    length = min(len(ego_traj), len(actor_traj))
    if length < 2:
        return float("inf")

    dt_safe = max(dt, 1e-6)
    rel = np.asarray(actor_traj[:length], dtype=float) - np.asarray(ego_traj[:length], dtype=float)
    distances = np.linalg.norm(rel, axis=1)
    already_in_conflict = np.flatnonzero(distances <= conflict_distance)
    if already_in_conflict.size:
        return float(already_in_conflict[0] * dt_safe)

    rel_prev = rel[:-1]
    rel_next = rel[1:]
    rel_velocity = (rel_next - rel_prev) / dt_safe
    speed_sq = np.sum(rel_velocity * rel_velocity, axis=1)
    valid = speed_sq > 1e-6
    time_to_cpa = np.full(len(rel_prev), np.inf, dtype=float)
    time_to_cpa[valid] = -np.sum(rel_prev[valid] * rel_velocity[valid], axis=1) / speed_sq[valid]
    valid &= (time_to_cpa >= 0.0) & (time_to_cpa <= dt_safe)
    if not np.any(valid):
        return float("inf")

    closest_points = rel_prev + rel_velocity * time_to_cpa[:, None]
    closest_distances = np.linalg.norm(closest_points, axis=1)
    valid &= closest_distances <= conflict_distance
    if not np.any(valid):
        return float("inf")

    segment_times = np.arange(len(rel_prev), dtype=float) * dt_safe
    ttc_values = segment_times[valid] + time_to_cpa[valid]
    return float(np.clip(ttc_values.min(), 0.0, 99.0))


def heading_conflict(actor: dict[str, Any]) -> float:
    yaw = float(actor.get("yaw") or 0.0)
    direction = np.array([math.cos(yaw), math.sin(yaw)], dtype=float)
    ego_forward = np.array([1.0, 0.0], dtype=float)
    return float(np.clip(-np.dot(direction, ego_forward), 0.0, 1.0))


def risk_score(min_dist: float, ttc: float, actor_type: str) -> float:
    class_weight = {
        "pedestrian": 1.4,
        "two_wheeler": 1.2,
        "vehicle": 1.0,
    }.get(actor_type, 1.0)
    ttc_term = 0.0 if math.isinf(ttc) else math.exp(-ttc / 2.0)
    return float(class_weight * (math.exp(-min_dist / 4.0) + ttc_term))


def safe_distance_for_actor(actor_type: str) -> float:
    return {
        "pedestrian": 3.0,
        "two_wheeler": 2.0,
        "vehicle": 3.5,
    }.get(actor_type, 3.0)


def score_candidate(
    ego_waypoints: np.ndarray,
    actor_rollouts: list[tuple[dict[str, Any], np.ndarray]],
    dt: float,
) -> dict[str, Any]:
    if not actor_rollouts:
        progress = float(np.linalg.norm(ego_waypoints[-1] - ego_waypoints[0]))
        return {
            "risk_score": 0.0,
            "min_distance": float("inf"),
            "ttc": float("inf"),
            "safe": True,
            "progress": progress,
            "comfort": trajectory_comfort(ego_waypoints),
            "actor_id": None,
            "actor_type": None,
        }

    worst: dict[str, Any] | None = None
    for actor, actor_traj in actor_rollouts:
        dist = min_distance(ego_waypoints, actor_traj)
        ttc = approximate_ttc(ego_waypoints, actor_traj, dt)
        actor_type = actor["actor_type"]
        risk = risk_score(dist, ttc, actor_type)
        safe = dist >= safe_distance_for_actor(actor_type) and (math.isinf(ttc) or ttc >= 1.0)
        result = {
            "risk_score": risk,
            "min_distance": dist,
            "ttc": ttc,
            "safe": safe,
            "actor_id": actor.get("id"),
            "actor_type": actor_type,
        }
        if worst is None or result["risk_score"] > worst["risk_score"]:
            worst = result

    assert worst is not None
    worst["progress"] = float(np.linalg.norm(ego_waypoints[-1] - ego_waypoints[0]))
    worst["comfort"] = trajectory_comfort(ego_waypoints)
    return worst


def trajectory_comfort(waypoints: np.ndarray) -> float:
    if len(waypoints) < 3:
        return 0.0
    diffs = np.diff(waypoints, axis=0)
    second = np.diff(diffs, axis=0)
    return float(np.linalg.norm(second, axis=1).mean())


def choose_best_candidate(
    candidates: list[tuple[str, np.ndarray, dict[str, Any]]],
) -> tuple[str, np.ndarray, dict[str, Any]]:
    safe_candidates = [item for item in candidates if item[2]["safe"]]
    pool = safe_candidates or candidates
    return min(
        pool,
        key=lambda item: (
            item[2]["risk_score"],
            item[2]["comfort"] * 0.1,
            -item[2]["progress"] * 0.01,
        ),
    )


def collision_rectangles_overlap(
    center_a: Any,
    extent_a: Any,
    center_b: Any,
    extent_b: Any,
) -> bool:
    """Axis-aligned overlap helper for fast conservative synthetic tests."""
    a = as_xy(center_a)
    b = as_xy(center_b)
    ea = np.asarray(extent_a, dtype=float)[:2]
    eb = np.asarray(extent_b, dtype=float)[:2]
    return bool(np.all(np.abs(a - b) <= (ea + eb)))


def validate_dreamer_option(option: dict[str, Any]) -> list[str]:
    errors = []
    missing = REQUIRED_DREAMER_KEYS - option.keys()
    if missing:
        errors.append(f"missing keys: {sorted(missing)}")
    if "waypoints" in option and not (
        isinstance(option["waypoints"], str) and option["waypoints"] == "org"
    ):
        arr = np.asarray(option["waypoints"], dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            errors.append("waypoints must be an Nx2 array or 'org'")
    if "route" in option and not (
        isinstance(option["route"], str) and option["route"] == "org"
    ):
        arr = np.asarray(option["route"], dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            errors.append("route must be an Nx2 array or 'org'")
    if "dreamer_instruction" in option and not option["dreamer_instruction"]:
        errors.append("dreamer_instruction must be non-empty")
    if "info" in option and not isinstance(option["info"], dict):
        errors.append("info must be a dict")
    return errors


def validate_dreamer_payload(payload: dict[str, Any]) -> list[str]:
    errors = []
    if not isinstance(payload, dict) or not payload:
        return ["payload must be a non-empty dict"]
    for mode, options in payload.items():
        if not isinstance(options, list) or not options:
            errors.append(f"{mode}: must contain a non-empty option list")
            continue
        for idx, option in enumerate(options):
            for error in validate_dreamer_option(option):
                errors.append(f"{mode}[{idx}]: {error}")
    return errors


def update_audit_counts(
    audit: dict[str, Any],
    path_boxes: Path,
    boxes: list[dict[str, Any]],
    dataset_root: Path,
) -> None:
    rel = path_boxes.relative_to(dataset_root)
    parts = rel.parts
    split = parts[2] if len(parts) > 2 else "unknown"
    town = next((part for part in parts if part.startswith("Town")), "unknown")

    audit["frames"] += 1
    audit["by_split"][split] += 1
    audit["by_town"][town] += 1
    for box in boxes:
        actor_type = classify_actor(box)
        cls = str(box.get("class"))
        audit["by_class"][cls] += 1
        if box.get("type_id") is not None:
            audit["by_type_id"][str(box.get("type_id"))] += 1
        if box.get("number_of_wheels") is not None:
            audit["by_wheels"][str(box.get("number_of_wheels"))] += 1
        if actor_type:
            audit["by_actor_type"][actor_type] += 1


def new_audit() -> dict[str, Any]:
    return {
        "frames": 0,
        "generated_frames": 0,
        "generated_options": 0,
        "by_split": defaultdict(int),
        "by_town": defaultdict(int),
        "by_class": defaultdict(int),
        "by_type_id": defaultdict(int),
        "by_wheels": defaultdict(int),
        "by_actor_type": defaultdict(int),
        "mode_counts": defaultdict(int),
        "skip_reasons": defaultdict(int),
    }


def finalize_audit(audit: dict[str, Any]) -> dict[str, Any]:
    finalized = {}
    for key, value in audit.items():
        if isinstance(value, defaultdict):
            finalized[key] = dict(sorted(value.items()))
        elif isinstance(value, Counter):
            finalized[key] = dict(value)
        else:
            finalized[key] = value
    return finalized
