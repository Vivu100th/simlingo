#!/usr/bin/env python3
"""
Collect one Bench2Drive scenario, train/resume SimLingo Base on that dataset,
convert the latest training checkpoint for the agent, then benchmark the same
route.

The script is intentionally conservative:
- it only prunes checkpoints inside outputs/*_<train-name>/checkpoints
- it resumes from the newest checkpoint in that same train-name scope by default
- it uses checkpoint=... only when no resume checkpoint is found
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PYTHON = "/home/vivu/miniconda3/envs/simlingo_personal/bin/python"
DEFAULT_BASELINE_BIN = (
    "outputs/2026_05_07_18_28_45_simlingo_base_seed_42/"
    "checkpoints/epoch=029_fp32/pytorch_model.bin"
)


def abs_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def run_logged(command: list[str], log_path: Path, env: dict[str, str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 80)
    print("RUN:", " ".join(command))
    print("LOG:", log_path)
    print("=" * 80)

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env or os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return process.wait()


def checkpoint_mtime(path: Path) -> float:
    if path.is_file():
        return path.stat().st_mtime
    latest = path.stat().st_mtime
    for child in path.rglob("*"):
        try:
            latest = max(latest, child.stat().st_mtime)
        except OSError:
            pass
    return latest


def hydra_quote(value: str | Path) -> str:
    text = str(value)
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


def checkpoint_dirs_for_train_name(train_name: str) -> list[Path]:
    return sorted(REPO_ROOT.glob(f"outputs/*_{train_name}/checkpoints"))


def checkpoint_candidates(train_name: str, scope: str) -> list[Path]:
    if scope == "pipeline":
        search_dirs = checkpoint_dirs_for_train_name(train_name)
    elif scope == "global":
        search_dirs = sorted(REPO_ROOT.glob("outputs/*/checkpoints"))
    else:
        raise ValueError(f"Unsupported resume scope: {scope}")

    candidates: list[Path] = []
    for checkpoint_dir in search_dirs:
        if not checkpoint_dir.is_dir():
            continue
        for child in checkpoint_dir.iterdir():
            if child.name == "last.ckpt" or child.name.startswith("epoch=") and child.name.endswith(".ckpt"):
                candidates.append(child)
    return candidates


def latest_checkpoint(train_name: str, scope: str) -> Path | None:
    candidates = checkpoint_candidates(train_name, scope)
    if not candidates:
        return None
    return max(candidates, key=checkpoint_mtime)


def prune_checkpoints(train_name: str, keep_last_n: int, preserve: Iterable[Path] | None = None) -> None:
    if keep_last_n < 1:
        raise ValueError("--keep-last-n must be >= 1")

    preserve_set = {path.resolve() for path in preserve or []}

    for checkpoint_dir in checkpoint_dirs_for_train_name(train_name):
        entries = [
            child
            for child in checkpoint_dir.iterdir()
            if child.name == "last.ckpt"
            or child.name.startswith("epoch=")
            or child.name.endswith("_fp32")
        ]
        keep = set(sorted(entries, key=checkpoint_mtime, reverse=True)[:keep_last_n])
        keep.update(entry for entry in entries if entry.resolve() in preserve_set)
        if not keep:
            continue
        print(f"Pruning checkpoints in {checkpoint_dir}")
        for entry in entries:
            if entry in keep:
                print(f"  keep   {entry}")
                continue
            print(f"  remove {entry}")
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()


def count_collected_routes(dataset_root: Path) -> int:
    data_dir = dataset_root / "data"
    if not data_dir.exists():
        return 0
    simlingo_routes = list((data_dir / "simlingo").glob("*/*/*/Town*"))
    if simlingo_routes:
        return sum(1 for path in simlingo_routes if path.is_dir() and (path / "rgb").is_dir())
    return sum(1 for path in data_dir.rglob("Town*") if path.is_dir() and (path / "rgb").is_dir())


def convert_checkpoint_for_benchmark(checkpoint: Path, python_bin: str) -> Path:
    if checkpoint.is_dir():
        zero_script = checkpoint / "zero_to_fp32.py"
        if not zero_script.exists():
            raise RuntimeError(f"Directory checkpoint has no zero_to_fp32.py: {checkpoint}")
        output_dir = checkpoint.with_name(checkpoint.name.replace(".ckpt", "_fp32"))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_bin = output_dir / "pytorch_model.bin"
        command = [python_bin, str(zero_script), str(checkpoint), str(output_bin)]
        subprocess.check_call(command, cwd=REPO_ROOT)
        return output_bin

    if checkpoint.suffix != ".ckpt":
        return checkpoint

    output_dir = checkpoint.with_name(checkpoint.stem + "_fp32")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_bin = output_dir / "pytorch_model.bin"
    script = (
        "import sys, torch\n"
        "src, dst = sys.argv[1], sys.argv[2]\n"
        "obj = torch.load(src, map_location='cpu', weights_only=False)\n"
        "state = obj.get('state_dict', obj)\n"
        "if 'model_state_dict' in state:\n"
        "    state = state['model_state_dict']\n"
        "torch.save(state, dst)\n"
        "print(dst)\n"
    )
    subprocess.check_call([python_bin, "-c", script, str(checkpoint), str(output_bin)], cwd=REPO_ROOT)
    return output_bin


def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=name.replace("-", "_"), action="store_true", help=help_text)
    group.add_argument(f"--no-{name}", dest=name.replace("-", "_"), action="store_false")
    parser.set_defaults(**{name.replace("-", "_"): default})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-file", default="leaderboard/data/bench2drive_split/bench2drive_07.xml")
    parser.add_argument("--scenario-group", default="parking_exit_one_scenario")
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--train-name", default="")
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", DEFAULT_PYTHON))
    parser.add_argument("--carla-root", default=os.environ.get("CARLA_ROOT", "/home/vivu/software/carla0915"))
    parser.add_argument("--initial-checkpoint", default=os.environ.get("SIMLINGO_INITIAL_CHECKPOINT", DEFAULT_BASELINE_BIN))
    parser.add_argument("--resume-scope", choices=["pipeline", "global"], default="pipeline")
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--repetition-start", type=int, default=0)
    parser.add_argument("--route-timeout-seconds", type=int, default=1800)
    parser.add_argument("--keep-last-n", type=int, default=1)
    parser.add_argument("--benchmark-result", default="")
    parser.add_argument("--benchmark-debug", type=int, default=0)
    parser.add_argument("--visible-carla", action="store_true")
    parser.add_argument("--carla-resx", default="1280")
    parser.add_argument("--carla-resy", default="720")
    parser.add_argument("--carla-quality", default="Low")
    add_bool_arg(parser, "collect", True, "Run data collection")
    add_bool_arg(parser, "train", True, "Run training")
    add_bool_arg(parser, "benchmark", True, "Run Bench2Drive")
    add_bool_arg(parser, "auto-start-carla", True, "Auto-start CARLA for data collection")
    add_bool_arg(parser, "resume-latest", True, "Resume from latest checkpoint in scope")
    add_bool_arg(parser, "prune-checkpoints", True, "Delete old checkpoints in this train-name scope")
    add_bool_arg(parser, "convert-checkpoint", True, "Convert latest ckpt to pytorch_model.bin for benchmark")
    return parser.parse_args()


def train_command(args: argparse.Namespace, dataset_root: Path, resume_checkpoint: Path | None) -> list[str]:
    dataset_rel = dataset_root.relative_to(REPO_ROOT)
    command = [
        args.python_bin,
        "-m",
        "simlingo_base_training.train",
        "experiment=simlingo_base_1",
        "data_module=carla_no_buckets",
        f"data_module.data_path={hydra_quote(dataset_rel)}",
        f"data_module.batch_size={args.batch_size}",
        f"data_module.num_workers={args.num_workers}",
        f"max_epochs={args.max_epochs}",
        f"gpus={args.gpus}",
        "strategy=auto",
        "debug=true",
        "enable_wandb=false",
        f"name={args.train_name}",
    ]
    if resume_checkpoint is not None:
        command.extend(["resume=true", f"resume_path={hydra_quote(resume_checkpoint)}"])
    else:
        initial = abs_path(args.initial_checkpoint)
        if initial.exists():
            command.append(f"checkpoint={hydra_quote(initial)}")
        else:
            print(f"Initial checkpoint not found, training from scratch: {initial}")
    return command


def main() -> int:
    args = parse_args()
    if args.dataset_name == "":
        args.dataset_name = f"simlingo_v2_{args.scenario_group}"
    if args.train_name == "":
        args.train_name = f"one_scenario_{args.scenario_group}"

    route_file = abs_path(args.route_file)
    dataset_root = REPO_ROOT / "database" / args.dataset_name
    logs_dir = REPO_ROOT / "outputs" / "one_scenario_pipeline_logs"
    benchmark_result = (
        abs_path(args.benchmark_result)
        if args.benchmark_result
        else REPO_ROOT / "outputs" / f"benchmark_{args.train_name}.json"
    )

    print("Pipeline configuration")
    print(f"  route_file:     {route_file}")
    print(f"  scenario_group: {args.scenario_group}")
    print(f"  dataset_root:   {dataset_root}")
    print(f"  train_name:     {args.train_name}")
    print(f"  python_bin:     {args.python_bin}")

    if args.collect:
        env = os.environ.copy()
        env.update(
            {
                "SIMLINGO_CODE_ROOT": str(REPO_ROOT),
                "CARLA_ROOT": args.carla_root,
                "PYTHON_BIN": args.python_bin,
                "ROUTE_FILE": str(route_file),
                "SCENARIO_GROUP": args.scenario_group,
                "DATASET_NAME": args.dataset_name,
                "REPETITIONS": str(args.repetitions),
                "REPETITION_START": str(args.repetition_start),
                "MAX_RETRIES": "1",
                "AUTO_START_CARLA": "1" if args.auto_start_carla else "0",
                "ROUTE_TIMEOUT_SECONDS": str(args.route_timeout_seconds),
            }
        )
        code = run_logged(
            [args.python_bin, "collect_dataset_workstation.py"],
            logs_dir / f"collect_{args.scenario_group}.log",
            env=env,
        )
        if code != 0:
            return code

    route_count = count_collected_routes(dataset_root)
    print(f"Collected route dirs ready for dataloader: {route_count}")
    if args.train and route_count < 2:
        print(
            "ERROR: one-route training needs at least 2 collected route dirs "
            "because the dataloader splits train/val by route. Re-run with "
            "--repetitions 2 or more."
        )
        return 2

    resume_checkpoint = None
    if args.train and args.resume_latest:
        resume_checkpoint = latest_checkpoint(args.train_name, args.resume_scope)
        if resume_checkpoint is not None:
            print(f"Resume checkpoint selected: {resume_checkpoint}")

    if args.train and args.prune_checkpoints:
        preserve = [resume_checkpoint] if resume_checkpoint is not None else []
        prune_checkpoints(args.train_name, args.keep_last_n, preserve=preserve)

    if args.train:
        code = run_logged(
            train_command(args, dataset_root, resume_checkpoint),
            logs_dir / f"train_{args.train_name}.log",
        )
        if code != 0:
            return code

    if args.benchmark:
        benchmark_checkpoint = latest_checkpoint(args.train_name, "pipeline")
        if benchmark_checkpoint is None:
            benchmark_checkpoint = abs_path(args.initial_checkpoint)
        if args.convert_checkpoint:
            benchmark_checkpoint = convert_checkpoint_for_benchmark(benchmark_checkpoint, args.python_bin)
        env = os.environ.copy()
        env.update(
            {
                "SIMLINGO_BENCHMARK_ROUTES": str(route_file),
                "SIMLINGO_BENCHMARK_CHECKPOINT": str(benchmark_checkpoint),
                "SIMLINGO_BENCHMARK_RESULT": str(benchmark_result),
                "SIMLINGO_BENCHMARK_RESUME": "0",
                "SIMLINGO_BENCHMARK_DEBUG": str(args.benchmark_debug),
            }
        )
        if args.visible_carla:
            env.update(
                {
                    "SIMLINGO_VISIBLE_CARLA": "1",
                    "SIMLINGO_CARLA_RESX": args.carla_resx,
                    "SIMLINGO_CARLA_RESY": args.carla_resy,
                    "SIMLINGO_CARLA_QUALITY": args.carla_quality,
                }
            )
        code = run_logged(
            [args.python_bin, "run_benchmark_local.py"],
            logs_dir / f"benchmark_{args.train_name}.log",
            env=env,
        )
        if code != 0:
            return code

    print("\nPipeline finished")
    print(f"  dataset:    {dataset_root}")
    if args.benchmark:
        print(f"  checkpoint: {benchmark_checkpoint}")
    print(f"  result:     {benchmark_result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
