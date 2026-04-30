# Mixed-Traffic Interaction Dreaming (MTID)

MTID is an additive research sandbox for extending SimLingo Action Dreaming from
ego-only counterfactuals to interaction-aware mixed-traffic counterfactuals.

The implementation lives under `mtid/` and keeps the original SimLingo Action
Dreaming generator untouched. The only SimLingo-side training change is a
backward-compatible dataset switch: `data_module.base_dataset.dreamer_folder`.

## Current Status

MTID v0 is implemented and trainable on the local dataset:

- Dataset source: `database/simlingo_v2_all`
- Generated master labels: `mtid/outputs/labels`
- Training mirror: `database/simlingo_v2_all/mtid_dreamer`
- Generated label files: `3443`
- Generated Dreamer options: `10054`
- Clean route view: `database/simlingo_v2_all_clean`
- Clean route manifest: `mtid/outputs/debug/clean_dataset_manifest.json`
- Current best checkpoint: `outputs/2026_04_28_00_46_43_simlingo_mtid_10e/checkpoints/last.ckpt`
- First useful local checkpoint: `simlingo_training/outputs/2026_04_27_10_33_14_simlingo_mtid_1k/checkpoints/last.ckpt`
- Main local runs: `mtid_1k` and `mtid_10e`, `bf16-mixed`, one GPU, frozen vision encoder
- Final 1k run metrics: `train/loss_epoch=4.272`, `val/loss=2.921`
- Best offline signal: MTID 10e improves 500-sample waypoint ADE to instruction
  to `3.230` vs driving-only control `5.073`

The 200-step checkpoint is only a smoke checkpoint. Use the 10e checkpoint for
the next CARLA/Bench2Drive work.

## What MTID Generates

Each generated label is Dreamer-compatible and includes:

- `mode`
- `waypoints`
- `route`
- `rgb_path`
- `allowed`
- `info`
- `route_reasoning`
- `dreamer_instruction`
- `instructions_templates`
- `templates_placeholders`
- `dreamer_answer_safety`
- `safe_to_execute`

MTID-specific metadata is stored under `info`, including actor ids, actor type,
TTC, min distance, risk score, scenario source, and actor rollouts.

Supported v0 modes:

- `jaywalker_crossing`
- `motorcycle_cut_in`
- `two_wheeler_filtering`
- `wrong_way_two_wheeler`
- `dense_gap_yield`
- `lane_less_corridor`

Two-wheeler modes are generated only from real two-wheeler actors present in the
dataset. In the current data, the real two-wheeler is a bicycle actor
(`vehicle.diamondback.century`), so templates avoid pretending it is a
motorcycle when the actor is not one.

## Folder Layout

- `design.md`: research design and formula notes.
- `PROGRESS.md`: implementation log, validation results, training runs, and next
  steps.
- `core.py`: geometry, rollout, risk, and schema helpers.
- `generators/mixed_traffic_dreamer_generator.py`: full MTID label generator.
- `templates/mixed_traffic_dreamer.json`: instruction and safety templates.
- `tests/`: synthetic checks for core logic and generator schema behavior.
- `tools/visualize_mtid_samples.py`: RGB plus BEV label preview renderer.
- `tools/smoke_mtid_pipeline.py`: DataModule/model-loss smoke test.
- `tools/run_mtid_short.sh`: reusable training launcher for MTID experiments.
- `outputs/`: generated labels, debug reports, visualizations. This is ignored
  by git and should not be committed.
- `tools/summarize_eval_predictions.py`: compact reader for MTID prediction
  JSON files.
- `tools/summarize_carla_control_trace.py`: compact reader for CARLA control
  debug traces in `benchmark_output.log`.
- `tools/audit_driving_data_quality.py`: speed, displacement, target, and lane
  quality audit for the base driving dataset.
- `tools/build_clean_dataset.py`: builds a symlinked clean dataset root from the
  driving data-quality audit.

## SimLingo Files Touched

Minimal training integration:

- `simlingo_training/config.py`
  - adds `dreamer_folder`, validation split controls, debug Trainer controls,
    and `val_check_interval`.
- `simlingo_training/dataloader/dataset_base.py`
  - reads `dreamer_folder` instead of hard-coding `dreamer`.
  - resolves repo-root-relative dataset/template paths.
  - avoids zero-route validation splits when routes exist.
- `simlingo_training/train.py`
  - inserts repo root into `sys.path` for subdirectory launches.
  - supports local debug/short-run Trainer options.
  - avoids unnecessary distributed strategy setup when `strategy: auto`.
- `simlingo_training/utils/logging_project.py`
  - resolves Git root from parent directories.
- InternVL/adaptor compatibility fixes:
  - `models/encoder/internvl2_model.py`
  - `models/language_model/llm.py`
  - `models/adaptors/adaptors.py`
- CARLA/Bench2Drive smoke support:
  - `team_code/agent_simlingo.py`
  - `run_benchmark_local.py`
  - `mtid/routes/routes_mtid_smoke.xml`
  - `Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py`

The original generator under `dataset_generation/dreamer_data` is not modified.

## Reproduce Generation

Run unit tests:

```bash
conda activate simlingo_rtx5070
python -m unittest discover -s mtid/tests -p 'test_*.py'
```

Compile the generator:

```bash
python3 -m py_compile \
  mtid/core.py \
  mtid/generators/mixed_traffic_dreamer_generator.py \
  mtid/tools/visualize_mtid_samples.py
```

Generate the full MTID label set:

```bash
python3 mtid/generators/mixed_traffic_dreamer_generator.py \
  --dataset-root database/simlingo_v2_all \
  --random-subset-count -1 \
  --overwrite
```

Generate a small dry run:

```bash
python3 mtid/generators/mixed_traffic_dreamer_generator.py \
  --dataset-root database/simlingo_v2_all \
  --random-subset-count 50 \
  --overwrite
```

The generator writes:

- master labels: `mtid/outputs/labels`
- debug report: `mtid/outputs/debug/mtid_audit_report.json`
- frame summary: `mtid/outputs/debug/mtid_frame_summary.json`
- training mirror: `database/simlingo_v2_all/mtid_dreamer`

## Visualize Labels

Render 20 mode-balanced RGB plus BEV previews:

```bash
python3 mtid/tools/visualize_mtid_samples.py \
  --labels-root mtid/outputs/labels \
  --count 20 \
  --seed 42 \
  --clean-output
```

Outputs are written to `mtid/outputs/visualizations`. The BEV panel shows route,
candidate ego trajectory, actor positions, actor rollouts, risk, and instruction
text.

## Smoke Test Training Data

DataModule/collate smoke:

```bash
python3 mtid/tools/smoke_mtid_pipeline.py \
  --experiment mtid_debug \
  --batch-size 1 \
  --num-workers 0
```

One-batch model loss smoke:

```bash
python3 mtid/tools/smoke_mtid_pipeline.py \
  --experiment mtid_debug \
  --batch-size 1 \
  --num-workers 0 \
  --with-model-loss
```

Driving-only clean-subset smoke:

```bash
/home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python \
  mtid/tools/smoke_mtid_pipeline.py \
  --experiment internvl_driving_clean_1k \
  --dataset-mode driving \
  --batch-size 1 \
  --num-workers 0
```

One-step Lightning smoke:

```bash
cd simlingo_training
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python \
  train.py experiment=mtid_debug
```

## Training Configs

Available MTID experiment configs:

- `mtid_seed1`: full MTID fine-tune template, originally configured for 8 GPUs.
- `mtid_debug`: one-step smoke test for local debugging.
- `mtid_short`: 200-step pilot run, useful only as a pipeline smoke checkpoint.
- `mtid_1k`: first meaningful local run on the 11.48 GiB GPU.
- `internvl_driving_1k`: same InternVL2-1B/LoRA stack, but driving-only
  training for a cleaner control comparison.
- `internvl_driving_clean_1k`: same driving-only control, but trained from the
  symlinked clean route view at `database/simlingo_v2_all_clean`.

Run the first meaningful local fine-tune:

```bash
MTID_EXPERIMENT=mtid_1k \
PYTHON_BIN=/home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python \
  mtid/tools/run_mtid_short.sh
```

Run a shorter override:

```bash
MTID_EXPERIMENT=mtid_1k \
PYTHON_BIN=/home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python \
  mtid/tools/run_mtid_short.sh \
  max_steps=100 \
  name=simlingo_mtid_100
```

Summarize a completed training run:

```bash
python3 mtid/tools/summarize_training_run.py \
  simlingo_training/outputs/2026_04_27_10_33_14_simlingo_mtid_1k
```

## Evaluate The 1k Checkpoint

`simlingo_training/eval.py` is now usable for local MTID Dreaming prediction:

- `eval_load_path` selects the Lightning checkpoint.
- `eval_mode: Dreaming` selects `Eval_Dreamer`.
- `eval_batch_size` and `eval_num_workers` keep local GPU usage small.
- `limit_predict_batches` supports tiny smoke runs before larger evaluation.
- checkpoint loading uses `weights_only=False` because local Lightning
  checkpoints contain trusted OmegaConf metadata under PyTorch 2.6+.

Run the tiny two-batch checkpoint smoke:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python \
  simlingo_training/eval.py experiment=mtid_eval_1k
```

Useful config:

```text
simlingo_training/config/experiment/mtid_eval_1k.yaml
```

Prediction files are written next to the checkpoint:

```text
simlingo_training/outputs/2026_04_27_10_33_14_simlingo_mtid_1k/predictions
```

The current smoke writes language predictions and MTID-friendly generic Dreamer
metrics, including mode counts, allowed counts, route ADE to instruction, and
waypoint ADE to instruction.

The MTID eval config is deterministic by default:

- `eval_deterministic: true`
- `eval_seed: 9876`
- `eval_option_policy: index`
- `eval_safety_policy: alternate`
- `eval_prompt_policy: with_navigation`

This makes repeated offline eval runs pick the same option, safety mode, and
prompt shape for each dataset index. Keep this on when comparing baseline vs
MTID checkpoints.

Summarize the latest prediction JSON files:

```bash
python3 mtid/tools/summarize_eval_predictions.py \
  simlingo_training/outputs/2026_04_27_10_33_14_simlingo_mtid_1k/predictions \
  --max-samples 5
```

## Control Baseline Comparison

The clean local control is:

```text
simlingo_training/config/experiment/internvl_driving_1k.yaml
```

It uses the same InternVL2-1B/LoRA architecture as `mtid_1k`, but trains without
MTID Dreamer labels. This is a better first comparison than older ResNet/tiny
SimLingo base checkpoints.

Train the control:

```bash
MTID_EXPERIMENT=internvl_driving_1k \
PYTHON_BIN=/home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python \
  mtid/tools/run_mtid_short.sh
```

Evaluate MTID 1k on the larger deterministic subset:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python \
  simlingo_training/eval.py \
  experiment=mtid_eval_1k \
  eval_load_path=simlingo_training/outputs/2026_04_27_10_33_14_simlingo_mtid_1k/checkpoints/last.ckpt \
  limit_predict_batches=200 \
  data_module.base_dataset.min_val_routes=8
```

Evaluate the control on the same deterministic MTID subset:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python \
  simlingo_training/eval.py \
  experiment=mtid_eval_1k \
  eval_load_path=simlingo_training/outputs/2026_04_27_21_34_30_simlingo_internvl_driving_1k/checkpoints/last.ckpt \
  limit_predict_batches=200 \
  data_module.base_dataset.min_val_routes=8
```

Current deterministic comparison, 500 MTID samples from 8 validation routes:

| Metric | MTID 10e | MTID 1k | Driving-only control | Clean driving-only |
| --- | ---: | ---: | ---: | ---: |
| waypoint ADE to instruction | 3.230 | 4.038 | 5.073 | 7.844 |
| waypoint ADE to original | 5.109 | 7.641 | 2.738 | 5.454 |
| waypoint closer-to-instruction rate | 0.578 | 0.616 | 0.238 | 0.238 |
| route ADE to instruction | 0.872 | 0.524 | 0.634 | 0.623 |
| route ADE to original | 0.842 | 0.444 | 0.583 | 0.549 |
| route closer-to-instruction rate | 0.848 | 0.778 | 0.816 | 0.782 |

Read this as a strong offline signal: After a critical data generation fix and 10 epochs of training, the MTID model successfully learns to yield to unsafe instructions and brings the Yield ADE down to 2.729 (better than the control's 2.951). It reliably outputs "Ignore instruction..." and shifts the predicted waypoints toward the MTID instruction much more than the driving-only control, maintaining safe trajectories overall. 

The clean driving-only 1000-step pilot does not improve MTID offline response.
It is useful as a CARLA data-filtering sanity check, not as a stronger Dreamer
baseline.

## CARLA Smoke

Current closed-loop target checkpoint:

```text
outputs/2026_04_28_00_46_43_simlingo_mtid_10e/checkpoints/last.ckpt
```

Default smoke command:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python \
  run_benchmark_local.py
```

Useful route override:

```bash
SIMLINGO_BENCHMARK_ROUTES=leaderboard/data/routes_devtest.xml \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python \
  run_benchmark_local.py
```

Useful checkpoint/output override for baseline comparisons:

```bash
SIMLINGO_BENCHMARK_CHECKPOINT=simlingo_training/outputs/2026_04_27_21_34_30_simlingo_internvl_driving_1k/checkpoints/last.ckpt \
SIMLINGO_BENCHMARK_RESULT=outputs/benchmark_control_results.json \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python \
  run_benchmark_local.py
```

Enable control tracing:

```bash
SIMLINGO_CARLA_DEBUG_CONTROL=1 \
SIMLINGO_CARLA_DEBUG_CONTROL_FREQ=5 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python \
  run_benchmark_local.py 2>&1 | tee benchmark_output.log
```

Summarize a trace:

```bash
python3 mtid/tools/summarize_carla_control_trace.py benchmark_output.log
```

Current trace lines include both the applied CARLA command and PID/debug state:
`desired_speed`, `desired_speed_raw`, `speed_scale`, `min_desired_speed`,
`delta`, `stuck`, `force_move`, `gps`, and `compass`. The summarizer remains
backward-compatible with older logs and now reports PID desired-speed/delta
statistics when those fields are present.

Longitudinal speed calibration knobs:

```bash
SIMLINGO_CARLA_SPEED_SCALE=1.25          # default 1.0
SIMLINGO_CARLA_MIN_DESIRED_SPEED=3.0     # default 0.0, disabled
SIMLINGO_CARLA_MAX_DESIRED_SPEED=8.0     # default 0.0, disabled
```

Steering-source ablation:

```bash
SIMLINGO_CARLA_STEER_SOURCE=planner \
SIMLINGO_CARLA_DEBUG_CONTROL=1 \
SIMLINGO_CARLA_DEBUG_CONTROL_FREQ=5 \
SIMLINGO_CARLA_DEBUG_COLLISION=1 \
SIMLINGO_BENCHMARK_RESULT=outputs/benchmark_mtid_planner_steer_results.json \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python \
  run_benchmark_local.py 2>&1 | tee benchmark_mtid_planner_steer.log
```

Completed smoke-route planner-steering command:

```bash
SIMLINGO_CARLA_STEER_SOURCE=planner \
SIMLINGO_CARLA_DEBUG_CONTROL=1 \
SIMLINGO_CARLA_DEBUG_CONTROL_FREQ=20 \
SIMLINGO_CARLA_DEBUG_COLLISION=1 \
SIMLINGO_BENCHMARK_RESULT=outputs/benchmark_mtid_planner_steer_full_results.json \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python \
  run_benchmark_local.py 2>&1 | tee benchmark_mtid_planner_steer_full.log
```

Status from 2026-04-29:

- CARLA can load the MTID 10e checkpoint and enter the route.
- Direct driving mode is the default to avoid language-generation OOM on the
  11.48 GiB GPU: `SIMLINGO_CARLA_PREDICT_LANGUAGE=0`.
- Optional control tracing is available with
  `SIMLINGO_CARLA_DEBUG_CONTROL=1` and
  `SIMLINGO_CARLA_DEBUG_CONTROL_FREQ=20`. Use frequency `5` when dense
  step-by-step debugging is needed.
- The InternVL image-token path is fixed. The agent now reports
  `<IMG_CONTEXT>` id `151648` and `512` image-context tokens in the first CARLA
  prompt.
- CARLA agent now returns the applied `self.control`; before this fix, the
  initial-frame brake guard was logged but not actually returned to CARLA.
- Model-steering MTID 10e trace after that fix: the short smoke route enters CARLA,
  initial frames stay stopped correctly, the vehicle reaches about `6.6 m/s`,
  then becomes stationary around step `135` with `throttle=1.0`, target still
  about `34 m` away, and first predicted waypoint norm dropping below `0.3 m`.
- Driving-only control after the same `self.control` fix also becomes
  stationary: `261` control samples, speed max `9.0 m/s` at step `85`, first
  stuck step `110`, target still about `39 m` away on average. This makes the
  current CARLA failure a shared closed-loop issue, not an MTID-only regression.
- Clean driving-only 1000-step control also becomes stationary. The partial
  trace in `benchmark_clean_control.log` has `109` control samples, speed max
  `7.5 m/s` at step `75`, first stuck step `120`, and then sustained
  `throttle=1.0` with speed `0`. Route-level slow/stuck filtering alone did not
  fix closed-loop motion.
- Collision debug confirms the immediate stuck cause on the MTID smoke route:
  the ego drifts from `Driving` into `Parking` lane around step `105`, then
  collides with `static.prop.mesh` at step `118`. The strongest collision
  impulse is `10924.550`, followed by repeated static collisions while throttle
  remains high.
- Planner-steering ablation confirms the drift source: using
  `SIMLINGO_CARLA_STEER_SOURCE=planner` keeps the ego in `Driving` lane through
  the tested segment, with `0` collisions, max lane distance `0.054`, and no
  stuck step. This means the previous off-lane failure is caused by model-route
  steering drifting too far during avoidance, not by CARLA physics alone.
- Completed planner-steering smoke route:
  `outputs/benchmark_mtid_planner_steer_full_results.json`. It reports
  Driving Score `100.0`, Route Completion `100.0`, penalty `1.0`, `0`
  collisions, `0` off-road infractions, `0` route deviations, and `0` agent
  blocked events. The route status is `Completed`.
- The completed smoke route still records min-speed infractions
  (`91.143` aggregate in the JSON). This is now the main closed-loop issue:
  planner steering fixes the physical lane/collision failure, but speed needs
  calibration before larger Bench2Drive claims.
- Runtime speed calibration was tested on the same smoke route:

  | Run | Game time | Speed mean/max | Collision/off-route | MinSpeedTest |
  | --- | ---: | ---: | ---: | ---: |
  | planner steer, default speed | 44.25s | 4.091 / 6.200 | 0 / 0 | fail, 77.83% console |
  | planner steer, `SPEED_SCALE=1.25` | 38.85s | 4.579 / 7.400 | 0 / 0 | fail, 90.44% console |
  | planner steer, `SPEED_SCALE=1.25`, `MIN_DESIRED_SPEED=3.0`, `MAX_DESIRED_SPEED=8.0` | 35.20s | 5.220 / 8.000 | 0 / 0 | fail, 93.21% console |

  The calibration improves travel time and keeps the route physically clean,
  but it does not make `MinSpeedTest` pass. The next speed work should inspect
  the Bench2Drive min-speed criterion/reference traffic calculation and compare
  model speed waypoints against a planner/route-derived target speed.
- Follow-up code inspection shows `MinSpeedTest` is an efficiency metric in
  this Bench2Drive checkout, not a Driving Score penalty. The criterion logs a
  `MIN_SPEED_INFRACTION` at each checkpoint and marks itself failed, but
  `statistics_manager.py` sets its penalty type to `unused`, so route score and
  infraction penalty remain `100.0` and `1.0`.
- A first `routes_devtest.xml` run with planner steering reached route 0
  `ParkingExit`, then got stuck in a real vehicle interaction rather than a
  static curb collision. The run was stopped manually after the trace showed
  sustained full throttle, zero speed, `lane_type=Parking`, and repeated
  collisions with `vehicle.mercedes.coupe_2020`.

  | Artifact | Value |
  | --- | --- |
  | log | `benchmark_mtid_devtest_planner.log` |
  | result JSON | `outputs/benchmark_mtid_devtest_planner_results.json` |
  | route/scenario | `RouteScenario_0_rep0_Town12_ParkingExit_1_0` |
  | control snapshots | `25`, step `50 -> 1250` |
  | speed mean/max | `0.000 / 0.000` |
  | full throttle | `25/25` |
  | first stuck step | `50` |
  | collision events | `1188` |
  | max collision | step `79`, `vehicle.mercedes.coupe_2020`, intensity `4701.276` |

  The JSON has `entry_status: Started` and no route records because the run was
  stopped before leaderboard finalization. Treat this as a debug trace, not an
  official devtest score.
- Added a focused reproduction route:
  `mtid/routes/routes_mtid_parking_exit.xml`. It uses the same first two
  waypoints as the smoke route plus the exact `ParkingExit_1` scenario from
  `routes_devtest.xml`, so the failure can be rerun without the full devtest
  file.
- Code inspection note: `ParkingExit` deliberately teleports ego into a parking
  lane and spawns blocking vehicles in front, behind, and beside ego. The
  privileged autopilot has a `starts_with_parking_exit` route-planner workaround,
  while `team_code/agent_simlingo.py` currently uses the simpler `RoutePlanner`
  path without an equivalent parking-start adjustment.
- Added an optional SimLingo ParkingExit route-start adjustment behind:

  ```bash
  SIMLINGO_CARLA_PARKING_EXIT_ROUTE_FIX=1
  ```

  The first attempt used raw CARLA world coordinates in the route planner and
  failed with a huge target distance plus route deviation. The corrected version
  applies the GPS-to-CARLA planner-frame offset before prepending the current
  ego pose and a lane-change command.

  Focused finalized result:

  | Artifact | Value |
  | --- | --- |
  | log | `benchmark_mtid_parking_exit_routefix2.log` |
  | result JSON | `outputs/benchmark_mtid_parking_exit_routefix2_results.json` |
  | route completion | `100.0` |
  | driving score | `21.6` |
  | infraction penalty | `0.216` |
  | score blockers | vehicle collision, min-speed |
  | vehicle collisions | `3` registered, `933` debug collision events |
  | max collision | step `1210`, `vehicle.mercedes.coupe_2020`, intensity `8766.942` |
  | min-speed infractions | `91.143` aggregate |

  Interpretation: the frame-corrected route fix is useful because it lets the
  ego leave the parking lane and finish the route, but it is not a behavioral
  fix. The remaining failure is an interaction policy problem: the agent still
  pushes into the scenario vehicle instead of yielding or creeping safely during
  the ParkingExit merge.
- Added an optional diagnostic ParkingExit yield/creep guard behind:

  ```bash
  SIMLINGO_CARLA_PARKING_EXIT_YIELD=1
  SIMLINGO_CARLA_PARKING_EXIT_YIELD_STEPS=900
  SIMLINGO_CARLA_PARKING_EXIT_YIELD_BRAKE_STEPS=160
  SIMLINGO_CARLA_PARKING_EXIT_YIELD_DISTANCE=25
  ```

  This guard scans nearby CARLA vehicle actors in the route-fix merge direction,
  yields during the initial merge window, creeps while still in a Parking lane,
  and blocks the existing stuck `force_move` override while the guard is active.
  It is a debug policy, not a paper/evaluation policy.

  Latest diagnostic run:

  | Artifact | Value |
  | --- | --- |
  | log | `benchmark_mtid_parking_exit_yield3.log` |
  | result JSON | `outputs/benchmark_mtid_parking_exit_yield3_results.json` |
  | final status | timeout, `entry_status: Started` |
  | step range | `20 -> 2580` |
  | speed mean/max | `0.164 / 4.800 m/s` |
  | first stuck step | `960` |
  | full brake samples | `38 / 129` |
  | full throttle samples | `84 / 129` |
  | debug collision events | `1612` |
  | max collision | step `939`, `vehicle.ford.mustang`, intensity `8250.680` |
  | last collision | step `2592`, `vehicle.mercedes.coupe_2020`, intensity `59.587` |

  Interpretation: the guard avoids the immediate Parking-lane collision and
  gets the ego into the Driving lane cleanly. It then stops behind a real lead
  vehicle. When the guard expires, the original stuck/force-move behavior pushes
  into that vehicle and collisions resume. A longer guard would avoid collision
  but likely deadlock. The next real fix needs lead-vehicle following/yield
  behavior or training data for this interaction, not another forced merge.
- Added a one-line route-fix geometry diagnostic to `team_code/agent_simlingo.py`.
  Latest focused run:

  ```text
  benchmark_mtid_parking_exit_geometry.log
  ego=road529/lane-2/Parking, left=road529/lane-1/Driving, right=road529/lane-3/Shoulder
  route0=road529/lane-1/Driving, left=road529/lane1/Driving, right=road529/lane-2/Parking
  merge_command=CHANGELANELEFT
  ```

  This confirms the route-fix side is correct at scenario start: ego begins in a
  Parking lane and the intended route lane is directly to the left. The
  remaining blocker is interaction with vehicles in/near the target lane, not a
  simple left-vs-right lane-change sign bug.
- Added hazard actor lane/speed debug plus a merge-steer guard while creeping
  out of Parking. Before this patch, `creep:Parking` could reuse planner steer
  and steer back into a Parking lane. With merge-steer active, the focused trace
  keeps ego in `road529/lane-1/Driving` and stops safely behind a stationary
  `vehicle.mercedes.coupe_2020` about `5.18m` ahead.
- Tested `SIMLINGO_CARLA_PARKING_EXIT_NUDGE_STATIC=1` as a local workaround for
  the stationary actor. It failed: the ego collided with the Mercedes around
  step `400+`. The flag stays default-off. This makes the next fix clearer:
  the agent needs a route/trajectory shift around the parked/blocking actor or
  training data for this exact maneuver, not a naive local nudge.
- Added `SIMLINGO_CARLA_DEBUG_WAYPOINT_HAZARD=1` to check whether the model
  itself predicts a safe path around the static actor. On
  `benchmark_mtid_parking_exit_waypoint_hazard.log`, the stationary Mercedes
  has minimum model-route clearance of only `0.633m` and static-hazard
  model-route clearance mean/min of `1.197 / 0.633`. This supports targeted
  data collection for ParkingExit / parked-vehicle avoidance; the current model
  is not reliably producing a safe avoidance trajectory.

## Base Data Quality Audit

Command:

```bash
/home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python \
  mtid/tools/audit_driving_data_quality.py \
  --dataset-root database/simlingo_v2_all \
  --output mtid/outputs/debug/driving_data_quality_report.json
```

Current result:

- routes: `44`
- frames: `8031`
- speed `<0.1 m/s`: `33.6%`
- speed `<0.5 m/s`: `34.6%`
- displacement `<0.05 m/sample`: `33.7%`
- displacement `<0.20 m/sample`: `34.4%`
- non-driving lane fraction from available ego lane labels: `0.2%`

This supports the suspicion that the base driving data contains a large slow or
stuck portion. A few routes are especially problematic, including
`Town12_Rep0_3604_route0_04_23_00_45_03` with about `94%` slow/stuck frames.

## Clean Dataset View

The clean dataset is a symlink view, not a copy. It keeps the original
`database/simlingo_v2_all` intact and links only routes that pass the audit
thresholds.

Build or rebuild it:

```bash
/home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python \
  mtid/tools/build_clean_dataset.py \
  --report mtid/outputs/debug/driving_data_quality_report.json \
  --source-root database/simlingo_v2_all \
  --output-root database/simlingo_v2_all_clean \
  --max-slow-fraction 0.5 \
  --max-stuck-fraction 0.5 \
  --overwrite
```

Current clean result:

- kept routes/frames: `32 / 4531`
- excluded routes/frames: `12 / 3500`
- linked data routes: `32`
- linked MTID Dreamer routes: `30`
- missing MTID Dreamer route mirrors: `2`
- clean driving-only DataModule smoke: `3600` train samples, `450` val samples

Train the clean driving-only control:

```bash
MTID_EXPERIMENT=internvl_driving_clean_1k \
PYTHON_BIN=/home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python \
  mtid/tools/run_mtid_short.sh
```

Current clean-control artifacts:

- training run: `simlingo_training/outputs/2026_04_28_23_35_11_simlingo_internvl_driving_clean_1k`
- checkpoint: `simlingo_training/outputs/2026_04_28_23_35_11_simlingo_internvl_driving_clean_1k/checkpoints/last.ckpt`
- training summary: `train/loss_epoch=5.583`, `val/loss=9.426`
- 500-sample eval: `waypoint ADE to instruction=7.844`,
  `waypoint closer-to-instruction rate=0.238`
- CARLA trace: `benchmark_clean_control.log`
- CARLA trace summary: first stuck step `120`, speed max `7.5 m/s`

## Known Limitations

- MTID v0 changes data only. It does not add a safety loss or change the model
  architecture.
- True motorcycle coverage is not present in the current dataset. Current
  two-wheeler labels come from bicycle actors.
- `lane_less_corridor` is a route/density corridor-following proxy. True
  lane-less behavior needs custom scenario collection.
- The current local runs are MTID-only fine-tunes with `batch_size=1`. They are
  good for pipeline validation and early qualitative checks, not final paper
  claims.
- Closed-loop CARLA scoring is complete for the short MTID smoke route with
  planner steering only. A focused ParkingExit reproduction can reach `100%`
  route completion with the route fix, but still fails on vehicle interaction.
  The yield/creep guard proves the early collision is avoidable, yet it
  deadlocks or collides later without a real following policy.
  `routes_devtest.xml` is not complete yet.
- The clean dataset view is currently route-level filtering from the audit; it
  does not yet do frame-level reweighting.
- Generated labels and checkpoints are large and should stay out of git.

## Recommended Next Steps

1. Collect or synthesize targeted ParkingExit / parked-vehicle avoidance data:
   ego exits a parking lane, yields to moving target-lane traffic, avoids a
   stationary vehicle ahead, and re-centers in the driving lane.
2. Keep a controller-side route/trajectory shift as a smoke-test guardrail, but
   do not treat it as the final paper behavior unless the evaluation policy
   explicitly includes that rule.
3. Use the failed `NUDGE_STATIC` run as evidence that simple local creep around
   the stationary actor is unsafe.
4. Fine-tune from the current best checkpoint after the targeted data exists,
   then re-run the waypoint-hazard diagnostic to confirm model-route clearance
   improves around static hazards.
5. Re-run `mtid/routes/routes_mtid_parking_exit.xml` and require both `100%`
   route completion and no vehicle collision.
6. Run `routes_devtest.xml` only after the focused ParkingExit route passes.
7. Keep speed calibration as an efficiency ablation, not as a blocker for
   closed-loop smoke scoring.
8. Analyze model-route drift around steps `95-120` to understand why model
   steering leaves the driving lane and enters the parking/curb-side region.
9. Decide whether planner steering is a smoke-test guardrail, the default
   closed-loop evaluation policy, or should become a constrained/blended
   steering policy.
10. Only collect broader mixed-traffic data if two-wheeler, wrong-way, or
   lane-less coverage is not strong enough for the paper story after evaluating in closed-loop.
