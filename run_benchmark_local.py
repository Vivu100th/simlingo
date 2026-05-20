"""
Runs a local closed-loop Bench2Drive evaluation for a SimLingo Base checkpoint.

Usage:
  python run_benchmark_local.py

Optional environment overrides:
  CARLA_ROOT=/path/to/carla0915
  SIMLINGO_CODE_ROOT=/path/to/simlingo
  SIMLINGO_BENCHMARK_ROUTES=leaderboard/data/bench2drive_split/bench2drive_07.xml
  SIMLINGO_BENCHMARK_CHECKPOINT=outputs/.../checkpoints/epoch=029_fp32/pytorch_model.bin
  SIMLINGO_BENCHMARK_RESULT=outputs/benchmark_base_results.json
  SIMLINGO_BENCHMARK_RESUME=0

This script will:
1. Configure the CARLA/Bench2Drive environment.
2. Run the leaderboard evaluator using the selected SimLingo Base checkpoint.
3. Save the results to `outputs/benchmark_base_results.json`.
"""

import os
import subprocess
import time
import sys
import signal
import atexit

# ============================================================================
# CONFIGURATION
# ============================================================================

CODE_ROOT = (
    os.environ.get("SIMLINGO_CODE_ROOT")
    or os.environ.get("CODE_ROOT")
    or os.path.dirname(os.path.abspath(__file__))
)
CARLA_ROOT = os.environ.get("CARLA_ROOT", "/home/vivu/software/carla0915")

# The route file to evaluate on. The default is a single Bench2Drive ParkingExit
# route so a local CARLA integration check can finish on one GPU.
# Override examples:
#   SIMLINGO_BENCHMARK_ROUTES=leaderboard/data/routes_devtest.xml python run_benchmark_local.py
#   SIMLINGO_BENCHMARK_ROUTES=leaderboard/data/bench2drive220.xml python run_benchmark_local.py
ROUTES_FILE = os.environ.get(
    "SIMLINGO_BENCHMARK_ROUTES",
    f"{CODE_ROOT}/leaderboard/data/bench2drive_split/bench2drive_07.xml",
)
if not os.path.isabs(ROUTES_FILE):
    ROUTES_FILE = os.path.join(CODE_ROOT, ROUTES_FILE)

# Default SimLingo Base checkpoint. Override with SIMLINGO_BENCHMARK_CHECKPOINT
# when evaluating a newly trained checkpoint.
CHECKPOINT_MODEL = os.environ.get(
    "SIMLINGO_BENCHMARK_CHECKPOINT",
    f"{CODE_ROOT}/outputs/2026_05_07_18_28_45_simlingo_base_seed_42/checkpoints/epoch=029_fp32/pytorch_model.bin",
)
if not os.path.isabs(CHECKPOINT_MODEL):
    CHECKPOINT_MODEL = os.path.join(CODE_ROOT, CHECKPOINT_MODEL)

# Where to save the benchmark results. Override this when running multiple
# checkpoint comparisons so stale partial JSON files do not collide.
RESULTS_JSON = os.environ.get(
    "SIMLINGO_BENCHMARK_RESULT",
    f"{CODE_ROOT}/outputs/benchmark_base_results.json",
)
if not os.path.isabs(RESULTS_JSON):
    RESULTS_JSON = os.path.join(CODE_ROOT, RESULTS_JSON)

CARLA_WORLD_PORT = 2000
CARLA_TM_PORT = 8000
CARLA_STARTUP_WAIT = 40
RESUME_BENCHMARK = bool(int(os.environ.get("SIMLINGO_BENCHMARK_RESUME", "0")))
BENCHMARK_DEBUG = int(os.environ.get("SIMLINGO_BENCHMARK_DEBUG", "0"))

_carla_process = None

# ============================================================================

def setup_environment():
    os.environ["SCENARIO_RUNNER_ROOT"] = f"{CODE_ROOT}/Bench2Drive/scenario_runner"
    os.environ["LEADERBOARD_ROOT"] = f"{CODE_ROOT}/Bench2Drive/leaderboard"
    os.environ["CARLA_ROOT"] = CARLA_ROOT
    os.environ["CARLA_SERVER"] = f"{CARLA_ROOT}/CarlaUE4.sh"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if "CUDA_HOME" not in os.environ:
        python_prefix = os.path.dirname(os.path.dirname(sys.executable))
        if os.path.exists(os.path.join(python_prefix, "bin", "nvcc")):
            os.environ["CUDA_HOME"] = python_prefix

    pythonpath_additions = [
        f"{CARLA_ROOT}/PythonAPI/carla",
        f"{CARLA_ROOT}/PythonAPI",
        f"{CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.8-linux-x86_64.egg",
        f"{CODE_ROOT}/Bench2Drive/leaderboard",
        f"{CODE_ROOT}/Bench2Drive/scenario_runner",
        CODE_ROOT,
    ]
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = ":".join(pythonpath_additions) + ":" + existing_pythonpath

    os.environ["DEBUG_CHALLENGE"] = "0"
    os.environ["CHALLENGE_TRACK_CODENAME"] = "MAP"
    os.environ["RESUME"] = "1" if RESUME_BENCHMARK else "0"


def start_carla_server():
    global _carla_process
    carla_bin = os.path.join(CARLA_ROOT, "CarlaUE4.sh")
    print(f"Starting CARLA server (offscreen) from {carla_bin}...")
    
    _carla_process = subprocess.Popen(
        [carla_bin, "-RenderOffScreen", "-nosound", f"-carla-rpc-port={CARLA_WORLD_PORT}", "-quality-level=Low"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid,
    )

    for elapsed in range(CARLA_STARTUP_WAIT):
        print(f"\r  Waiting for CARLA... {CARLA_STARTUP_WAIT - elapsed}s remaining  ", end="", flush=True)
        time.sleep(1)
        if elapsed >= 10:
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex(('localhost', CARLA_WORLD_PORT))
                sock.close()
                if result == 0:
                    break
            except Exception:
                pass
    print(f"\n  ✓ CARLA server is running (PID: {_carla_process.pid})")


def stop_carla_server():
    global _carla_process
    if _carla_process is not None and _carla_process.poll() is None:
        print("\nStopping CARLA server...")
        try:
            os.killpg(os.getpgid(_carla_process.pid), signal.SIGTERM)
            _carla_process.wait(timeout=15)
        except Exception:
            try:
                os.killpg(os.getpgid(_carla_process.pid), signal.SIGKILL)
            except Exception:
                pass
        _carla_process = None


atexit.register(stop_carla_server)


def run_benchmark():
    agent = f"{CODE_ROOT}/team_code/agent_simlingo.py"
    if not RESUME_BENCHMARK and os.path.exists(RESULTS_JSON):
        print(f"Removing stale benchmark result before fresh run: {RESULTS_JSON}")
        os.remove(RESULTS_JSON)

    run_command = [
        sys.executable, "Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py",
        f"--port={CARLA_WORLD_PORT}",
        f"--traffic-manager-port={CARLA_TM_PORT}",
        f"--routes={ROUTES_FILE}",
        "--repetitions=1",
        "--track=SENSORS",
        f"--checkpoint={RESULTS_JSON}",
        f"--agent={agent}",
        f"--agent-config={CHECKPOINT_MODEL}",
        f"--debug={BENCHMARK_DEBUG}",
    ]
    if RESUME_BENCHMARK:
        run_command.append("--resume=1")

    print(f"\n{'='*60}")
    print(f"STARTING BENCHMARK EVALUATION")
    print(f"{'='*60}")
    print(f"  Model:  {CHECKPOINT_MODEL}")
    print(f"  Routes: {ROUTES_FILE}")
    print(f"  Output: {RESULTS_JSON}")
    print(f"  Command: {' '.join(run_command)}\n")

    try:
        completed = subprocess.run(run_command, cwd=CODE_ROOT, env=os.environ.copy())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C).")
        return 130

    if completed.returncode != 0:
        print(f"\nBenchmark failed with exit code {completed.returncode}.")
        return completed.returncode
    
    print(f"\n{'='*60}")
    print(f"EVALUATION FINISHED. Results saved to:\n{RESULTS_JSON}")
    print(f"{'='*60}")
    return 0


def main():
    setup_environment()
    # CARLA is started by leaderboard_evaluator.py, so we DO NOT start it here!
    sys.exit(run_benchmark())


if __name__ == "__main__":
    main()
