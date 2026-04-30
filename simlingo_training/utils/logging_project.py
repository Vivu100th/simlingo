import json
import os
import glob
import subprocess
import argparse
import logging

from git import Repo
from omegaconf import OmegaConf
from pathlib import Path
from datetime import datetime

from hydra.utils import get_original_cwd, to_absolute_path


def _resolve_git_root(start_dir):
    path = Path(start_dir).resolve()
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate

    try:
        root = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
        ).decode("ascii").strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return path

    return Path(root)


def _git_output(working_dir, *args):
    return (
        subprocess.check_output(["git", "-C", str(working_dir), *args])
        .decode("ascii")
        .strip()
    )


def setup_logging(cfg, save_folder=None):
    
    if save_folder is None:
        working_dir = _resolve_git_root(get_original_cwd())
        save_folder = 'log'
    else:
        # get working dir
        working_dir = _resolve_git_root(os.getcwd())
        save_folder = save_folder + '/log'
    # Log args
    # Path(save_folder).mkdir(parents=True, exist_ok=True)
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    arg_dict = OmegaConf.to_container(cfg, resolve=True)
    args = argparse.Namespace(**arg_dict)
    with open(os.path.join(save_folder, "args.txt"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    # Log git
    try:
        sha = _git_output(working_dir, "rev-parse", "HEAD")
        commit = _git_output(working_dir, "log", "-1")
        branch = _git_output(working_dir, "branch")
        repo = Repo(working_dir)
        diff = repo.git.diff("HEAD")
    except Exception as exc:
        sha = "unavailable"
        commit = "unavailable"
        branch = "unavailable"
        diff = f"Git logging unavailable: {exc}"

    with open(os.path.join(save_folder, "git_info.txt"), "w") as f:
        # write current date and time
        f.write(
            f"Run started at: {str(datetime.now().strftime('%d/%m/%Y %H:%M:%S'))}\n"
        )
        f.write(f"Git state: {sha}\n")
        f.write(f"Git commit: {commit}\n")
        f.write(f"Git branch: {branch}\n\n")
        f.write(diff)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def sync_wandb(cfg):
    # TODO: sync wandb - still not working correctly
    wandb_files = glob.glob(f"./wandb/offline*/*.wandb")
    os.environ["TMPDIR"] = "/home/geiger/krenz73/tmp"
    for wandb_file in wandb_files:
        if os.path.getsize(wandb_file) > 5000000:
            os.system(f"wandb sync {wandb_file}")
