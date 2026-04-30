"""Summarize SimLingo/MTID prediction JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_PREDICTIONS = Path(
    "simlingo_training/outputs/2026_04_27_10_33_14_simlingo_mtid_1k/predictions"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_PREDICTIONS,
        help="Prediction directory or a run directory containing predictions/.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5,
        help="Maximum per-sample rows to print.",
    )
    return parser.parse_args()


def resolve_predictions_dir(path: Path) -> Path:
    if path.name == "predictions":
        return path
    if (path / "predictions").is_dir():
        return path / "predictions"
    return path


def newest_file(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.glob(pattern), key=lambda p: (p.stat().st_mtime, p.name))
    return matches[-1] if matches else None


def load_json(path: Path | None) -> Any:
    if path is None:
        return None
    with path.open("r") as handle:
        return json.load(handle)


def print_metric_block(results: dict[str, Any]) -> None:
    print("metrics:")
    for key in sorted(results):
        value = results[key]
        if isinstance(value, dict):
            items = ", ".join(f"{item_key}={item_value}" for item_key, item_value in value.items())
            print(f"  {key}: {items}")
        else:
            print(f"  {key}: {value}")


def shorten(text: str, max_chars: int = 180) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def print_samples(samples: list[dict[str, Any]], max_samples: int) -> None:
    print("")
    print("samples:")
    if not samples:
        print("  none")
        return

    for row in samples[:max_samples]:
        path = Path(row.get("path", "")).name
        print(
            "  "
            f"idx={row.get('sample_index')} "
            f"mode={row.get('mode')} "
            f"allowed={row.get('allowed')} "
            f"wp_ade_instr={row.get('waypoint_ade_to_instruction')} "
            f"wp_ade_org={row.get('waypoint_ade_to_original')} "
            f"path={path}"
        )
        print(f"    prompt: {shorten(row.get('prompt', ''))}")
        print(f"    pred:   {shorten(row.get('pred_language', ''))}")
        print(f"    gt:     {shorten(row.get('gt_language', ''))}")


def main() -> None:
    args = parse_args()
    predictions_dir = resolve_predictions_dir(args.path)
    if not predictions_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {predictions_dir}")

    results_path = newest_file(predictions_dir, "dreamer_results_rank_*.json")
    samples_path = newest_file(predictions_dir, "mtid_samples_all_rank_*.json")
    language_path = newest_file(predictions_dir, "language_preds_all_rank_*.json")

    results = load_json(results_path) or {}
    samples = load_json(samples_path) or []

    print(f"predictions_dir: {predictions_dir}")
    print(f"results_file: {results_path.name if results_path else 'none'}")
    print(f"samples_file: {samples_path.name if samples_path else 'none'}")
    print(f"language_file: {language_path.name if language_path else 'none'}")
    print("")
    print_metric_block(results)
    print_samples(samples, args.max_samples)


if __name__ == "__main__":
    main()
