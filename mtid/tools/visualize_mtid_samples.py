"""Render quick RGB + BEV previews for generated MTID dreamer labels."""

from __future__ import annotations

import argparse
import gzip
import json
import random
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CANVAS_W = 900
RGB_H = 360
BEV_H = 420
PAD = 18
BEV_SCALE = 11.0


def load_json_gz(path: Path) -> Any:
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def collect_options(labels_root: Path, mode: str | None) -> list[tuple[Path, str, int, dict[str, Any]]]:
    samples: list[tuple[Path, str, int, dict[str, Any]]] = []
    for path in sorted(labels_root.glob("**/*.json.gz")):
        payload = load_json_gz(path)
        for option_mode, options in payload.items():
            if mode is not None and option_mode != mode:
                continue
            for idx, option in enumerate(options):
                samples.append((path, option_mode, idx, option))
    return samples


def select_samples(
    samples: list[tuple[Path, str, int, dict[str, Any]]],
    count: int,
    seed: int,
    balanced: bool,
) -> list[tuple[Path, str, int, dict[str, Any]]]:
    rng = random.Random(seed)
    if not balanced:
        shuffled = samples[:]
        rng.shuffle(shuffled)
        return shuffled[:count]

    by_mode: dict[str, list[tuple[Path, str, int, dict[str, Any]]]] = {}
    for sample in samples:
        by_mode.setdefault(sample[1], []).append(sample)
    for mode_samples in by_mode.values():
        rng.shuffle(mode_samples)

    selected: list[tuple[Path, str, int, dict[str, Any]]] = []
    mode_names = sorted(by_mode)
    while len(selected) < count and any(by_mode.values()):
        for mode_name in mode_names:
            if by_mode[mode_name]:
                selected.append(by_mode[mode_name].pop())
                if len(selected) >= count:
                    break
    return selected


def open_rgb(path_text: str, width: int = CANVAS_W) -> Image.Image:
    path = Path(path_text)
    if not path.exists():
        image = Image.new("RGB", (width, RGB_H), (35, 35, 35))
        draw = ImageDraw.Draw(image)
        draw.text((PAD, PAD), f"Missing RGB: {path}", fill=(240, 240, 240))
        return image
    image = Image.open(path).convert("RGB")
    image.thumbnail((width, RGB_H), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, RGB_H), (12, 15, 18))
    offset = ((width - image.width) // 2, (RGB_H - image.height) // 2)
    canvas.paste(image, offset)
    return canvas


def to_points(value: Any) -> np.ndarray:
    if isinstance(value, str):
        return np.zeros((0, 2), dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.zeros((0, 2), dtype=float)
    return arr[:, :2]


def project_bev(point: np.ndarray, origin: tuple[int, int]) -> tuple[int, int]:
    x, y = float(point[0]), float(point[1])
    return int(origin[0] - y * BEV_SCALE), int(origin[1] - x * BEV_SCALE)


def draw_polyline(
    draw: ImageDraw.ImageDraw,
    points: np.ndarray,
    origin: tuple[int, int],
    fill: tuple[int, int, int],
    width: int,
) -> None:
    if len(points) < 2:
        return
    projected = [project_bev(point, origin) for point in points]
    draw.line(projected, fill=fill, width=width, joint="curve")
    for px, py in projected:
        draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=fill)


def actor_color(actor_type: str | None) -> tuple[int, int, int]:
    if actor_type == "pedestrian":
        return (240, 80, 80)
    if actor_type == "two_wheeler":
        return (164, 107, 255)
    if actor_type == "vehicle":
        return (255, 173, 51)
    return (90, 100, 110)


def draw_actor_marker(
    draw: ImageDraw.ImageDraw,
    position: Any,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    label: str,
    radius: int = 8,
) -> None:
    point = np.asarray(position, dtype=float)
    px, py = project_bev(point, origin)
    draw.ellipse(
        (px - radius, py - radius, px + radius, py + radius),
        fill=color,
        outline=(20, 20, 20),
        width=2,
    )
    if label:
        draw.text((px + radius + 2, py - radius), label, fill=color)


def draw_actor_rollouts(draw: ImageDraw.ImageDraw, info: dict[str, Any], origin: tuple[int, int]) -> None:
    rollouts = info.get("actor_rollouts") or []
    if not rollouts:
        position = info.get("actor_position")
        if position is not None:
            draw_actor_marker(
                draw,
                position,
                origin,
                actor_color(info.get("actor_type")),
                str(info.get("actor_ids") or ""),
            )
        return

    risk_actor_id = info.get("risk_actor_id")
    primary_ids = set(info.get("actor_ids") or [])
    for rollout_info in rollouts:
        actor_type = rollout_info.get("actor_type")
        color = actor_color(actor_type)
        actor_id = rollout_info.get("actor_id")
        current_position = rollout_info.get("current_position")
        trajectory = to_points(rollout_info.get("trajectory"))
        points = trajectory
        if current_position is not None:
            current = np.asarray(current_position, dtype=float).reshape(1, -1)[:, :2]
            points = np.concatenate([current, trajectory], axis=0) if len(trajectory) else current

        width = 5 if actor_id == risk_actor_id else 3
        if len(points) >= 2:
            draw_polyline(draw, points, origin, color, width)
        if current_position is not None:
            label = str(actor_id) if actor_id in primary_ids or actor_id == risk_actor_id else ""
            draw_actor_marker(draw, current_position, origin, color, label, radius=8 if label else 5)


def draw_bev(option: dict[str, Any], mode: str) -> Image.Image:
    image = Image.new("RGB", (CANVAS_W, BEV_H), (246, 248, 250))
    draw = ImageDraw.Draw(image)
    origin = (CANVAS_W // 2, BEV_H - 52)

    for meters in range(0, 36, 5):
        y = origin[1] - int(meters * BEV_SCALE)
        draw.line((40, y, CANVAS_W - 40, y), fill=(224, 229, 234), width=1)
        draw.text((44, y - 12), f"{meters}m", fill=(120, 128, 138))
    draw.line((origin[0], 22, origin[0], BEV_H - 24), fill=(210, 216, 222), width=1)

    route = to_points(option.get("route"))
    waypoints = to_points(option.get("waypoints"))
    draw_polyline(draw, route, origin, (66, 133, 244), 3)
    draw_actor_rollouts(draw, option.get("info", {}), origin)
    color = (32, 164, 96) if option.get("safe_to_execute") else (220, 64, 64)
    draw_polyline(draw, waypoints, origin, color, 5)

    ego = (origin[0] - 10, origin[1] - 20, origin[0] + 10, origin[1] + 20)
    draw.rounded_rectangle(ego, radius=3, fill=(38, 50, 56), outline=(0, 0, 0))
    draw.polygon(
        [(origin[0], origin[1] - 28), (origin[0] - 8, origin[1] - 14), (origin[0] + 8, origin[1] - 14)],
        fill=(38, 50, 56),
    )
    draw.text((PAD, PAD), f"BEV: {mode}", fill=(28, 35, 42))
    draw.text((CANVAS_W - 360, PAD), "blue route | green/red ego | warm/purple/red actors", fill=(95, 105, 115))
    return image


def format_info(path: Path, mode: str, option_idx: int, option: dict[str, Any]) -> list[str]:
    info = option.get("info", {})
    ttc = info.get("ttc")
    ttc_text = "inf" if ttc is None else f"{ttc:.2f}s"
    raw_lines = [
        f"{mode} option={option_idx} safe={option.get('safe_to_execute')} allowed={option.get('allowed')}",
        f"action={info.get('candidate_action')} actor={info.get('actor_type')} ids={info.get('actor_ids')}",
        f"actor_rollouts={len(info.get('actor_rollouts') or [])} risk_actor={info.get('risk_actor_id')}",
        f"risk={info.get('risk_score', 0.0):.3f} min_dist={info.get('min_distance', 0.0):.2f}m ttc={ttc_text}",
        f"source={info.get('scenario_source')}",
        f"label={path}",
        f"instruction={option.get('dreamer_instruction', [''])[0]}",
    ]
    lines: list[str] = []
    for raw_line in raw_lines:
        lines.extend(textwrap.wrap(raw_line, width=115, subsequent_indent="  ") or [""])
    return lines


def render_sample(path: Path, mode: str, option_idx: int, option: dict[str, Any], output_path: Path) -> None:
    rgb = open_rgb(option.get("rgb_path", ""))
    bev = draw_bev(option, mode)
    info_h = 168
    canvas = Image.new("RGB", (CANVAS_W, RGB_H + BEV_H + info_h), (255, 255, 255))
    canvas.paste(rgb, (0, 0))
    canvas.paste(bev, (0, RGB_H))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    y = RGB_H + BEV_H + 10
    for line in format_info(path, mode, option_idx, option):
        draw.text((PAD, y), line, fill=(28, 35, 42), font=font)
        y += 18

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize generated MTID dreamer labels.")
    parser.add_argument("--labels-root", default="mtid/outputs/labels")
    parser.add_argument("--output-root", default="mtid/outputs/visualizations")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", default=None)
    parser.add_argument("--clean-output", action="store_true", help="Remove existing JPG previews in the output folder first.")
    parser.add_argument("--no-balanced", action="store_true", help="Use pure random sampling instead of mode-balanced sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels_root = Path(args.labels_root)
    output_root = Path(args.output_root)
    if args.clean_output and output_root.exists():
        for old_preview in output_root.glob("*.jpg"):
            old_preview.unlink()
    samples = collect_options(labels_root, args.mode)
    if not samples:
        raise SystemExit(f"No MTID options found under {labels_root}")

    selected = select_samples(samples, max(args.count, 0), args.seed, balanced=not args.no_balanced)
    for sample_idx, (path, mode, option_idx, option) in enumerate(selected):
        stem = f"{sample_idx:03d}_{mode}_{'safe' if option.get('safe_to_execute') else 'unsafe'}"
        render_sample(path, mode, option_idx, option, output_root / f"{stem}.jpg")
    print(f"Wrote {len(selected)} visualizations to {output_root}")


if __name__ == "__main__":
    main()
