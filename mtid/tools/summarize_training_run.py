"""Summarize an MTID/SimLingo Lightning training output directory."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable


METRIC_NAMES = (
    "train/loss_step",
    "train/loss_epoch",
    "val/loss",
    "train_losses/language_loss",
    "train_losses/route_loss",
    "train_losses/speed_wps_loss",
    "val_losses/language_loss",
    "val_losses/route_loss",
    "val_losses/speed_wps_loss",
    "lr-AdamW",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Lightning output directory, or a direct path to metrics.csv.",
    )
    return parser.parse_args()


def find_metrics_path(path: Path) -> Path:
    if path.is_file():
        return path

    matches = sorted(path.glob("log/**/metrics.csv"))
    if not matches:
        raise FileNotFoundError(f"No metrics.csv found under {path}")
    if len(matches) > 1:
        print("multiple metrics.csv files found; using the last sorted match")
    return matches[-1]


def find_checkpoint_paths(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("checkpoints/*.ckpt"))


def infer_run_dir(input_path: Path, metrics_path: Path) -> Path:
    if input_path.is_dir():
        return input_path

    for parent in metrics_path.parents:
        if (parent / ".hydra").exists() or (parent / "checkpoints").exists():
            return parent
    return metrics_path.parent


def to_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def to_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def latest_values(rows: Iterable[dict[str, str]]) -> tuple[int | None, dict[str, float]]:
    last_step: int | None = None
    latest: dict[str, float] = {}

    for row in rows:
        step = to_int(row.get("step"))
        if step is not None:
            last_step = step if last_step is None else max(last_step, step)

        for name in METRIC_NAMES:
            value = to_float(row.get(name))
            if value is not None:
                latest[name] = value

    return last_step, latest


def human_size(path: Path) -> str:
    size = path.stat().st_size
    for unit in ("B", "K", "M", "G", "T"):
        if size < 1024 or unit == "T":
            return f"{size:.1f}{unit}" if unit != "B" else f"{size}B"
        size /= 1024
    return f"{size:.1f}T"


def main() -> None:
    args = parse_args()
    metrics_path = find_metrics_path(args.run_dir)
    run_dir = infer_run_dir(args.run_dir, metrics_path)

    with metrics_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    last_step, latest = latest_values(rows)
    checkpoints = find_checkpoint_paths(run_dir)

    print(f"run_dir: {run_dir}")
    print(f"metrics: {metrics_path}")
    print(f"rows: {len(rows)}")
    print(f"last_step: {last_step}")
    print("")
    print("latest_metrics:")
    for name in METRIC_NAMES:
        if name in latest:
            print(f"  {name}: {latest[name]:.6g}")

    print("")
    print("checkpoints:")
    if not checkpoints:
        print("  none")
    for checkpoint in checkpoints:
        print(f"  {checkpoint.name}: {human_size(checkpoint)}")


if __name__ == "__main__":
    main()
