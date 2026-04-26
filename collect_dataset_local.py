"""
Generates a dataset for training on a LOCAL machine (no SLURM required).
Each route is run sequentially on a single machine.

CARLA server is started AUTOMATICALLY in offscreen mode.
No need to start it manually.

Usage:
  python collect_dataset_local.py

Best run inside a tmux terminal so it can continue if you disconnect.

How to know when data is ready for training:
  - At the end of collection (or on Ctrl+C), a summary is printed.
  - Look for "✅ DATASET IS READY FOR TRAINING" message.
  - Key requirement: at least 1 route with score_route >= 94% AND
    frames captured in rgb/ and measurements/ folders.
"""

from datetime import datetime
import os
import subprocess
import time
import glob
import json
import sys
import gzip
import signal
import atexit
from pathlib import Path
import random
import re


# ============================================================================
# CONFIGURATION - Edit these before running!
# ============================================================================

# Path to the simlingo code directory
CODE_ROOT = r"/home/vivu/Desktop/simlingo"

# Path to CARLA installation root
CARLA_ROOT = "/home/vivu/software/carla0915"

# Dataset naming
# Default to a stable merged dataset so multiple collection days are easy to train together.
# Set USE_DAILY_DATASET=True if you want to go back to one folder per day.
USE_DAILY_DATASET = False
DATE = datetime.today().strftime("%Y_%m_%d")
DATASET_NAME = "simlingo_v2_" + DATE if USE_DAILY_DATASET else "simlingo_v2_all"
ROOT_FOLDER = r"database/"  # With ending slash
DATA_SAVE_DIRECTORY = ROOT_FOLDER + DATASET_NAME

# Route repetitions
REPETITIONS = 1
REPETITION_START = 0

# CARLA server port
CARLA_WORLD_PORT = 2000
CARLA_TM_PORT = 8000

# How many times to retry a failed route
MAX_RETRIES = 3

# Auto-start CARLA settings
AUTO_START_CARLA = False   # Set to False if you want to start CARLA manually
CARLA_STARTUP_WAIT = 40   # Seconds to wait for CARLA to fully load

# ============================================================================
# END CONFIGURATION
# ============================================================================

# Global reference to CARLA process so we can clean up on exit
_carla_process = None


def setup_environment():
    """Set up the required environment variables for CARLA and leaderboard."""
    os.environ["SCENARIO_RUNNER_ROOT"] = f"{CODE_ROOT}/scenario_runner_autopilot"
    os.environ["LEADERBOARD_ROOT"] = f"{CODE_ROOT}/leaderboard_autopilot"
    os.environ["CARLA_ROOT"] = CARLA_ROOT
    os.environ["CARLA_SERVER"] = f"{CARLA_ROOT}/CarlaUE4.sh"

    # Build PYTHONPATH
    pythonpath_additions = [
        f"{CARLA_ROOT}/PythonAPI/carla",
        f"{CARLA_ROOT}/PythonAPI",
        f"{CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.8-linux-x86_64.egg",
        f"{CODE_ROOT}/leaderboard_autopilot",
        f"{CODE_ROOT}/scenario_runner_autopilot",
        CODE_ROOT,
    ]
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = ":".join(pythonpath_additions) + ":" + existing_pythonpath

    # Needed by data_agent
    os.environ["DATAGEN"] = "1"
    os.environ["REPETITIONS"] = "1"
    os.environ["DEBUG_CHALLENGE"] = "0"
    os.environ["CHALLENGE_TRACK_CODENAME"] = "MAP"
    os.environ["RESUME"] = "1"


# ============================================================================
# CARLA SERVER MANAGEMENT
# ============================================================================

def start_carla_server():
    """Start CARLA server in offscreen mode. Returns the subprocess."""
    global _carla_process
    carla_bin = os.path.join(CARLA_ROOT, "CarlaUE4.sh")

    if not os.path.isfile(carla_bin):
        print(f"ERROR: CARLA executable not found at {carla_bin}")
        sys.exit(1)

    print(f"Starting CARLA server (offscreen) from {carla_bin}...")
    print(f"  Port: {CARLA_WORLD_PORT}")
    print(f"  Waiting {CARLA_STARTUP_WAIT}s for server to initialize...")

    _carla_process = subprocess.Popen(
        [
            carla_bin,
            "-RenderOffScreen",
            "-nosound",
            f"-carla-rpc-port={CARLA_WORLD_PORT}",
            "-quality-level=Low",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,  # Create new process group for clean kill
    )

    # Wait for CARLA to be ready by trying to connect
    connected = False
    for elapsed in range(CARLA_STARTUP_WAIT):
        remaining = CARLA_STARTUP_WAIT - elapsed
        print(f"\r  Waiting for CARLA... {remaining}s remaining  ", end="", flush=True)
        time.sleep(1)

        # Check if CARLA process died
        if _carla_process.poll() is not None:
            print(f"\nERROR: CARLA process exited with code {_carla_process.returncode}")
            sys.exit(1)

        # Try to connect after 10 seconds
        if elapsed >= 10:
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex(('localhost', CARLA_WORLD_PORT))
                sock.close()
                if result == 0:
                    connected = True
                    break
            except Exception:
                pass

    print()  # newline after \r
    if connected:
        print(f"  ✓ CARLA server is running (PID: {_carla_process.pid})")
    else:
        print(f"  ⚠ Could not verify CARLA connection, but process is running. Proceeding...")

    return _carla_process


def stop_carla_server():
    """Stop the CARLA server if we started it."""
    global _carla_process
    if _carla_process is not None and _carla_process.poll() is None:
        print("\nStopping CARLA server...")
        try:
            os.killpg(os.getpgid(_carla_process.pid), signal.SIGTERM)
            _carla_process.wait(timeout=15)
            print("  ✓ CARLA server stopped.")
        except Exception:
            # Force kill if graceful shutdown fails
            try:
                os.killpg(os.getpgid(_carla_process.pid), signal.SIGKILL)
                print("  ✓ CARLA server force-killed.")
            except Exception:
                pass
        _carla_process = None


# Register cleanup so CARLA is stopped even if the script crashes
atexit.register(stop_carla_server)


# ============================================================================
# DATASET VALIDATION
# ============================================================================

def validate_dataset(data_root):
    """
    Check collected data and report if it's ready for training.
    Returns (is_ready, stats_dict).

    The training code (simlingo_base_training) requires:
      - rgb/ folder with .jpg images
      - measurements/ folder with .json.gz files
      - results.json.gz with score_route >= 94%
    """
    data_path = os.path.join(data_root, "data")
    results_path = os.path.join(data_root, "results")

    stats = {
        "total_routes_found": 0,
        "perfect_routes": 0,      # score == 100%
        "usable_routes": 0,       # score >= 94%
        "crashed_routes": 0,      # score < 94% or crashed
        "total_frames": 0,
        "total_rgb_images": 0,
        "total_measurements": 0,
        "estimated_size_gb": 0.0,
        "usable_route_list": [],
    }

    if not os.path.exists(data_path):
        return False, stats

    # Find all route directories (contain rgb/ and measurements/)
    route_dirs = glob.glob(f"{data_path}/**/Town*", recursive=True)

    for route_dir in route_dirs:
        rgb_dir = os.path.join(route_dir, "rgb")
        meas_dir = os.path.join(route_dir, "measurements")
        results_file = os.path.join(route_dir, "results.json.gz")

        if not os.path.isdir(rgb_dir) or not os.path.isdir(meas_dir):
            continue

        stats["total_routes_found"] += 1

        # Count frames
        rgb_files = glob.glob(os.path.join(rgb_dir, "*.jpg"))
        meas_files = glob.glob(os.path.join(meas_dir, "*.json.gz"))
        num_frames = len(rgb_files)
        stats["total_rgb_images"] += num_frames
        stats["total_measurements"] += len(meas_files)

        # Check result quality
        is_usable = False
        score_route = 0.0
        if os.path.exists(results_file):
            try:
                with gzip.open(results_file, 'rt') as f:
                    result_data = json.load(f)
                score_route = result_data.get('scores', {}).get('score_route', 0.0)
                score_composed = result_data.get('scores', {}).get('score_composed', 0.0)
                num_infractions = result_data.get('num_infractions', 999)

                if score_composed >= 100.0:
                    stats["perfect_routes"] += 1
                    is_usable = True
                elif score_route >= 94.0:
                    # Usable if only infractions are min_speed or outside_route_lanes
                    infractions = result_data.get('infractions', {})
                    minor = len(infractions.get('min_speed_infractions', []))
                    minor += len(infractions.get('outside_route_lanes', []))
                    if num_infractions == minor:
                        is_usable = True

            except Exception:
                pass

        if is_usable and num_frames > 10:  # At least 10 frames
            stats["usable_routes"] += 1
            stats["total_frames"] += num_frames
            stats["usable_route_list"].append({
                "path": route_dir,
                "frames": num_frames,
                "score": score_route,
            })
        elif num_frames > 0:
            stats["crashed_routes"] += 1

    # Estimate total size
    try:
        result = subprocess.run(
            ["du", "-s", "-b", "-L", data_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            size_bytes = int(result.stdout.split()[0])
            stats["estimated_size_gb"] = size_bytes / (1024**3)
    except Exception:
        pass

    is_ready = stats["usable_routes"] >= 1 and stats["total_frames"] >= 100
    return is_ready, stats


def print_dataset_summary(data_root):
    """Print a comprehensive summary of the collected dataset."""
    is_ready, stats = validate_dataset(data_root)

    print(f"\n{'='*60}")
    print(f"📊 DATASET VALIDATION REPORT")
    print(f"{'='*60}")
    print(f"  Data path:            {data_root}")
    print(f"  Total routes found:   {stats['total_routes_found']}")
    print(f"  ✓ Perfect routes:     {stats['perfect_routes']}  (score = 100%)")
    print(f"  ✓ Usable routes:      {stats['usable_routes']}  (score >= 94%)")
    print(f"  ✗ Crashed/unusable:   {stats['crashed_routes']}")
    print(f"  Total usable frames:  {stats['total_frames']}")
    print(f"  RGB images:           {stats['total_rgb_images']}")
    print(f"  Measurements:         {stats['total_measurements']}")
    print(f"  Estimated size:       {stats['estimated_size_gb']:.2f} GB")
    print(f"{'─'*60}")

    if is_ready:
        print(f"  ✅ DATASET IS READY FOR TRAINING")
        print(f"")
        print(f"  Để train, chỉnh sửa config sau:")
        print(f"    simlingo_base_training/config/experiment/simlingo_base_1.yaml")
        print(f"  Đổi data_path thành:")
        print(f"    data_path: {os.path.relpath(data_root, CODE_ROOT)}")
        print(f"")
        print(f"  Sau đó chạy:")
        print(f"    cd {CODE_ROOT}")
        print(f"    python -m simlingo_base_training.train experiment=simlingo_base_1")
    else:
        print(f"  ⚠️  DATASET CHƯA ĐỦ ĐỂ TRAIN")
        if stats['usable_routes'] == 0:
            print(f"  Lý do: Chưa có route nào hoàn thành đạt yêu cầu (score >= 94%)")
            print(f"  → Tiếp tục chạy collect_dataset_local.py")
        elif stats['total_frames'] < 100:
            print(f"  Lý do: Quá ít frames ({stats['total_frames']}). Cần ít nhất ~100 frames.")
            print(f"  → Tiếp tục chạy thêm routes")
        print(f"")
        print(f"  💡 Gợi ý: Cần ít nhất 5-10 routes usable (~1000+ frames) để train")
        print(f"     có ý nghĩa. Lý tưởng là 50+ routes (~10,000+ frames).")

    print(f"{'='*60}")

    # Show top 5 usable routes
    if stats['usable_route_list']:
        print(f"\n  📁 Top usable routes (by frame count):")
        sorted_routes = sorted(stats['usable_route_list'], key=lambda x: x['frames'], reverse=True)
        for i, r in enumerate(sorted_routes[:5]):
            short_path = os.path.relpath(r['path'], data_root)
            print(f"    {i+1}. {short_path}  ({r['frames']} frames, score={r['score']:.1f}%)")
        if len(sorted_routes) > 5:
            print(f"    ... and {len(sorted_routes)-5} more routes")
        print()


def is_route_completed(result_file):
    """Check if a route has already been completed successfully."""
    if not os.path.exists(result_file):
        return False

    try:
        with open(result_file, "r", encoding="utf-8") as f:
            evaluation_data = json.load(f)

        progress = evaluation_data["_checkpoint"]["progress"]
        if len(progress) < 2 or progress[0] < progress[1]:
            return False

        for record in evaluation_data["_checkpoint"]["records"]:
            if record["scores"]["score_route"] <= 0.00000000001:
                return False
            if record["status"] in [
                "Failed - Agent couldn't be set up",
                "Failed",
                "Failed - Simulation crashed",
                "Failed - Agent crashed",
            ]:
                return False

        return True
    except (json.JSONDecodeError, KeyError, Exception):
        return False


def run_single_route(route_file, agent, checkpoint_endpoint, save_path, seed, town, repetition):
    """Run a single route using the leaderboard evaluator. Returns True if successful."""
    env = os.environ.copy()
    env["ROUTES"] = route_file
    env["TEAM_AGENT"] = agent
    env["TEAM_CONFIG"] = route_file
    env["CHECKPOINT_ENDPOINT"] = checkpoint_endpoint
    env["SAVE_PATH"] = save_path
    env["TOWN"] = town
    env["REPETITION"] = str(repetition)
    env["TM_SEED"] = str(seed)

    run_command = [
        "python",
        "leaderboard/leaderboard/leaderboard_evaluator_local.py",
        f"--port={CARLA_WORLD_PORT}",
        f"--traffic-manager-port={CARLA_TM_PORT}",
        f"--traffic-manager-seed={seed}",
        f"--routes={route_file}",
        "--repetitions=1",
        "--track=MAP",
        f"--checkpoint={checkpoint_endpoint}",
        f"--agent={agent}",
        f"--agent-config={route_file}",
        "--debug=0",
        "--resume=1",
        "--timeout=600",
    ]

    print(f"  Running leaderboard evaluator...")
    print(f"  Command: {' '.join(run_command)}")

    try:
        result = subprocess.run(
            run_command,
            cwd=CODE_ROOT,
            env=env,
            timeout=3600,  # 1 hour timeout per route
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("  TIMEOUT: Route took longer than 1 hour, skipping.")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    setup_environment()

    # ── Auto-start CARLA ──
    if AUTO_START_CARLA:
        start_carla_server()
    else:
        print("Assuming CARLA is already running on port", CARLA_WORLD_PORT)
        print("If not, set AUTO_START_CARLA = True or start it manually.\n")

    route_folder = f"{CODE_ROOT}/data/simlingo"

    # Find all route files
    routes = glob.glob(f"{route_folder}/**/*balanced*/*.xml", recursive=True)
    routes_lb1 = glob.glob(f"{route_folder}/**/*lb1*/**/*.xml", recursive=True)
    routes = routes + routes_lb1

    if not routes:
        print(f"ERROR: No route files found in {route_folder}")
        print("Make sure CODE_ROOT is set correctly and route XML files exist.")
        stop_carla_server()
        sys.exit(1)

    # Shuffle routes (same seed as SLURM version for consistency)
    random.seed(42)
    random.shuffle(routes)

    num_routes = len(routes)
    seed_counter = 1000000 * REPETITION_START - 1

    print(f"\n{'='*60}")
    print(f"LOCAL DATASET COLLECTION")
    print(f"{'='*60}")
    print(f"Code root:     {CODE_ROOT}")
    print(f"CARLA root:    {CARLA_ROOT}")
    print(f"CARLA port:    {CARLA_WORLD_PORT}")
    print(f"Dataset:       {DATASET_NAME}")
    print(f"Save to:       {CODE_ROOT}/{DATA_SAVE_DIRECTORY}")
    print(f"Total routes:  {num_routes}")
    print(f"Repetitions:   {REPETITIONS}")
    print(f"{'='*60}\n")

    completed = 0
    skipped = 0
    failed = 0
    total_jobs = num_routes * (REPETITIONS - REPETITION_START)
    job_number = 0

    try:
        for repetition in range(REPETITION_START, REPETITIONS):
            for route in routes:
                seed_counter += 1
                job_number += 1

                # Extract town name
                try:
                    town = re.search(r'Town(\d+)', route).group(0)
                except AttributeError:
                    if 'validation' in route:
                        town = 'Town13'
                    elif 'training' in route:
                        town = 'Town12'
                    else:
                        print(f"  Town not found in route {route}, skipping.")
                        skipped += 1
                        continue

                # Build paths
                scenario_type = route.split("/")[-5:-1]
                scenario_type = "/".join(scenario_type)
                routefile_number = route.split("/")[-1].split(".")[0]
                ckpt_endpoint = f"{CODE_ROOT}/{DATA_SAVE_DIRECTORY}/results/{scenario_type}/{routefile_number}_result.json"
                save_path = f"{CODE_ROOT}/{DATA_SAVE_DIRECTORY}/data/{scenario_type}"
                Path(save_path).mkdir(parents=True, exist_ok=True)
                Path(ckpt_endpoint).parent.mkdir(parents=True, exist_ok=True)
                agent = f"{CODE_ROOT}/team_code/data_agent.py"

                print(f"\n{'─'*60}")
                print(f"[{job_number}/{total_jobs}] Route: {routefile_number} | Town: {town} | Rep: {repetition}")
                print(f"  Scenario: {scenario_type}")
                print(f"  Seed: {seed_counter}")

                # Check if already completed
                if is_route_completed(ckpt_endpoint):
                    print(f"  SKIPPED: Route already completed successfully.")
                    skipped += 1
                    continue

                # Try running the route (with retries)
                success = False
                for attempt in range(1, MAX_RETRIES + 1):
                    print(f"  Attempt {attempt}/{MAX_RETRIES}...")

                    success = run_single_route(
                        route_file=route,
                        agent=agent,
                        checkpoint_endpoint=ckpt_endpoint,
                        save_path=save_path,
                        seed=seed_counter,
                        town=town,
                        repetition=repetition,
                    )

                    if success and is_route_completed(ckpt_endpoint):
                        print(f"  ✓ Route completed successfully.")
                        completed += 1
                        break
                    else:
                        print(f"  ✗ Route failed or incomplete.")
                        if attempt < MAX_RETRIES:
                            print(f"  Waiting 10s before retry...")
                            time.sleep(10)
                            if AUTO_START_CARLA:
                                print(f"  [Auto-Recovery] Restarting CARLA to clear potential crash/hang...")
                                stop_carla_server()
                                subprocess.run("killall -9 CarlaUE4-Linux-Shipping", shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                                time.sleep(2)
                                start_carla_server()

                if not success or not is_route_completed(ckpt_endpoint):
                    failed += 1
                    print(f"  ✗ Route failed after {MAX_RETRIES} attempts.")

                # Print progress summary
                print(f"\n  Progress: ✓{completed} | ⏭{skipped} | ✗{failed} / {total_jobs} total")

                if AUTO_START_CARLA:
                    print(f"  [Routine Maintenance] Restarting CARLA to free memory and prevent crashes...")
                    stop_carla_server()
                    subprocess.run("killall -9 CarlaUE4-Linux-Shipping", shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                    time.sleep(5)
                    start_carla_server()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C).")

    # ── Collection Summary ──
    print(f"\n{'='*60}")
    print(f"DATASET COLLECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Completed: {completed}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")
    print(f"Total:     {total_jobs}")
    print(f"{'='*60}")

    # ── Dataset Validation ──
    full_data_root = os.path.join(CODE_ROOT, DATA_SAVE_DIRECTORY)
    print_dataset_summary(full_data_root)

    # ── Stop CARLA ──
    if AUTO_START_CARLA:
        stop_carla_server()


if __name__ == "__main__":
    main()
