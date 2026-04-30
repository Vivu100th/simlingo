# MTID Progress and Handoff

Last updated: 2026-04-29

This document records what has been implemented, validated, trained, and what
should happen next for Mixed-Traffic Interaction Dreaming (MTID).

## Snapshot

MTID v0 is working end to end:

- label generation: pass
- schema validation: pass
- visualization: pass
- DataModule/collate smoke: pass
- model-loss smoke: pass
- Lightning one-step debug: pass
- local 200-step pilot fine-tune: pass
- local 1000-step fine-tune: pass
- same-architecture driving-only control train/eval: pass
- local 10-epoch MTID fine-tune: pass
- CARLA agent loads 10e checkpoint: pass
- CARLA InternVL image-token insertion: pass
- CARLA control-trace summarizer: pass
- base driving data quality audit: pass
- clean symlink dataset builder: pass
- clean driving-only DataModule smoke: pass
- clean driving-only 1000-step train/eval: pass
- clean driving-only CARLA trace: still stuck
- CARLA PID/stuck debug instrumentation: pass
- CARLA collision/off-lane debug: collision confirmed
- CARLA planner-steering ablation: completed smoke route with no collision/off-route
- closed-loop Bench2Drive smoke score (planner steer): DS=100.0, RC=100%, penalty=1.0
- closed-loop min-speed issue (planner steer): still present
- CARLA runtime speed calibration ablation: faster, still MinSpeedTest failure
- Bench2Drive MinSpeedTest inspection: efficiency metric, penalty unused
- closed-loop Bench2Drive score (model steer): car stuck after collision (off-lane at step ~85)
- routes_devtest planner-steering trace: route 0 ParkingExit stuck after repeated vehicle collision
- ParkingExit route-frame fix: reaches 100% route completion but collides with scenario vehicles
- ParkingExit yield/creep guard diagnostic: avoids early Parking merge collision but later deadlocks/collides behind lead traffic
- ParkingExit route geometry diagnostic: merge side is correct at scenario start
- ParkingExit merge-steer guard: keeps Parking creep from steering back into Parking lane
- ParkingExit static nudge diagnostic: failed with vehicle collision, default off

Best checkpoint to use next:

```text
outputs/2026_04_28_00_46_43_simlingo_mtid_10e/checkpoints/last.ckpt
```

Do not use the 200-step checkpoint for serious evaluation. It is only a smoke
checkpoint. The 1k checkpoint is still useful for ablations, but the 10-epoch
checkpoint is the strongest current MTID candidate.

## Research Goal

SimLingo Action Dreaming is ego-centric: it asks what the ego should do under
alternative instructions. MTID extends this idea to interaction-aware
counterfactual futures:

- the ego has candidate actions such as keep, slow, brake, yield, and nudges;
- nearby road users also have counterfactual rollouts;
- each candidate gets risk metadata from distance, TTC, collision checks, and
  actor class;
- generated labels remain compatible with the existing Dreamer dataloader.

MTID v0 intentionally does not change the model architecture or loss. It tests
whether better mixed-traffic Dreamer data can improve the existing training
pipeline.

## Formula Notes

Ego state and action:

```text
x = [px, py, yaw, v]
u = [steer, throttle, brake]
x_next = bicycle_model(x, u)
```

MTID v0 does not run a full bicycle controller. It starts from expert waypoints
and derives candidate trajectories: keep, slow, brake, yield, nudge left, and
nudge right.

Time to collision:

```text
TTC = distance_along_conflict / closing_speed
```

Low TTC means a candidate ego trajectory is risky. In code,
`approximate_ttc` estimates when ego and actor rollouts first enter a conflict
distance.

Risk score:

```text
risk_i = exp(-distance / sigma_d) * exp(-TTC / sigma_t) * class_weight
```

MTID v0 follows this structure using min distance, TTC, collision checks, and
class-specific weights for pedestrians, two-wheelers, and vehicles.

Future possible safety loss:

```text
L_barrier = sum_t sum_i ReLU(d_safe_i^2 - d_i(t)^2)
```

This is not used in v0. It is a later architecture/training extension.

## Implementation Summary

Added under `mtid/`:

- `core.py`: geometry, rollout, risk, schema utilities.
- `generators/mixed_traffic_dreamer_generator.py`: full MTID label generator.
- `templates/mixed_traffic_dreamer.json`: instruction and safety templates.
- `tests/test_core.py`: synthetic geometry/risk checks.
- `tests/test_generator.py`: generator/schema checks.
- `tools/visualize_mtid_samples.py`: RGB plus BEV preview renderer.
- `tools/smoke_mtid_pipeline.py`: dataloader and optional model-loss smoke test.
- `tools/run_mtid_short.sh`: reusable launcher for MTID training configs.
- `tools/summarize_training_run.py`: compact metrics/checkpoint summary for a
  completed Lightning output directory.
- `simlingo_training/config/experiment/mtid_eval_1k.yaml`: tiny local Dreaming
  prediction config for the 1k checkpoint.
- `tools/summarize_eval_predictions.py`: compact reader for prediction metrics
  and per-sample MTID rows.
- `tools/summarize_carla_control_trace.py`: compact reader for CARLA control
  debug traces from `benchmark_output.log`.
- `tools/audit_driving_data_quality.py`: base dataset quality audit for speed,
  displacement, target point, command, and ego lane-type signals.
- `tools/build_clean_dataset.py`: symlinked clean route-view builder from the
  base data-quality audit.

Minimal SimLingo-side integration:

- `DatasetBaseConfig.dreamer_folder`, defaulting to `dreamer`.
- `BaseDataset` reads `dreamer_folder` instead of hard-coding `dreamer`.
- validation route selection keeps at least one route when validation routes
  exist.
- local training controls were added for debug and small-GPU runs.
- path and logging code now work when launched from either repo root or
  `simlingo_training/`.
- `simlingo_training/config/experiment/internvl_driving_clean_1k.yaml` points
  the driving-only control to `database/simlingo_v2_all_clean`.
- `mtid/tools/smoke_mtid_pipeline.py` can now smoke-test either Dreamer labels
  or driving-only data via `--dataset-mode`.

Original Action Dreaming generation code was not modified.

## Dataset Generation

Command used:

```bash
/home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python \
  mtid/generators/mixed_traffic_dreamer_generator.py \
  --dataset-root database/simlingo_v2_all \
  --random-subset-count -1 \
  --overwrite
```

Outputs:

- master labels: `mtid/outputs/labels`
- training mirror: `database/simlingo_v2_all/mtid_dreamer`
- audit report: `mtid/outputs/debug/mtid_audit_report.json`
- frame summary: `mtid/outputs/debug/mtid_frame_summary.json`

Generation results:

| Metric | Value |
| --- | ---: |
| Scanned frames | 8031 |
| Generated label files | 3443 |
| Generated options | 10054 |
| Safe options | 6488 |
| Unsafe options | 3566 |
| Schema errors | 0 |
| Mirrored training label files | 3443 |

Mode counts:

| Mode | Count |
| --- | ---: |
| `dense_gap_yield` | 4910 |
| `lane_less_corridor` | 4809 |
| `jaywalker_crossing` | 193 |
| `motorcycle_cut_in` | 65 |
| `two_wheeler_filtering` | 65 |
| `wrong_way_two_wheeler` | 12 |

Actor coverage:

| Actor Type | Count |
| --- | ---: |
| pedestrian | 387 |
| two-wheeler | 174 |
| vehicle | 93391 |

Current two-wheeler actors are bicycles, mainly `vehicle.diamondback.century`.
The generator does not fabricate motorcycles from cars.

## Visualization

Command:

```bash
/home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python \
  mtid/tools/visualize_mtid_samples.py \
  --labels-root mtid/outputs/labels \
  --count 20 \
  --seed 42 \
  --clean-output
```

Result:

- wrote 20 mode-balanced previews to `mtid/outputs/visualizations`.
- previews include RGB, route, ego candidate, actor current positions, actor
  rollout trajectories, risk, and instruction text.

Visualization colors:

- route: blue
- safe ego candidate: green
- unsafe ego candidate: red
- vehicle rollout: orange
- two-wheeler rollout: purple
- pedestrian rollout: red

Qualitative checks completed:

- `wrong_way_two_wheeler` shows bicycle rollout against the ego path.
- `dense_gap_yield` shows multiple vehicle rollouts in the gap.
- previews are useful for debugging and possible paper figures.

## Validation and Smoke Tests

Commands:

```bash
/home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python -m py_compile \
  mtid/core.py \
  mtid/generators/mixed_traffic_dreamer_generator.py \
  mtid/tools/visualize_mtid_samples.py

/home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python -m unittest discover \
  -s mtid/tests -p 'test_*.py'

/home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python \
  mtid/tools/smoke_mtid_pipeline.py \
  --experiment mtid_debug \
  --batch-size 1 \
  --num-workers 0

/home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python \
  mtid/tools/smoke_mtid_pipeline.py \
  --experiment mtid_debug \
  --batch-size 1 \
  --num-workers 0 \
  --with-model-loss
```

Results:

- py compile: pass
- unit tests: 16 tests OK
- schema validation over generated labels: pass
- DataModule/collate smoke: pass
- one-batch model-loss smoke: pass
- clean driving-only DataModule smoke: pass
- model-loss smoke observed train loss: `17.476160049438477`

Data shapes from the real DataModule:

| Tensor | Shape |
| --- | --- |
| camera images | `(B, 1, 2, 3, 448, 448)` |
| waypoints | `(B, 10, 2)` |
| route/path | `(B, 20, 2)` |

Dataset counts from smoke tests:

- train samples: `1710`
- validation samples: non-zero after validation split fix

## Training Fixes

Problems encountered and fixed:

- launching `python train.py` from `simlingo_training/` caused
  `ModuleNotFoundError: simlingo_training`;
- local debug run OOMed on the 11.48 GiB GPU with heavier defaults;
- Git logging assumed the launch directory was the repository root;
- dataset template/data paths assumed `get_original_cwd()` was repo root;
- `precision: 16-mixed` conflicted with local bfloat16 InternVL/LoRA tensors.

Fixes:

- `train.py` inserts repo root into `sys.path`.
- `train.py` supports debug/short-run Trainer controls:
  `max_steps`, `fast_dev_run`, `limit_train_batches`, `limit_val_batches`,
  `enable_checkpointing`, `enable_visualise_callback`,
  `num_sanity_val_steps`, `log_every_n_steps`, and `val_check_interval`.
- `strategy: auto` avoids explicit DDP setup for one-GPU local runs.
- `logging_project.py` resolves Git root from parent directories.
- `dataset_base.py` resolves repo-root-relative template, dataset, and bucket
  paths.
- local MTID configs use `precision: bf16-mixed`.
- InternVL wrappers include compatibility fixes for the current Transformers
  stack.
- adaptor and image features are cast to the expected dtype/device.

## Evaluation Smoke

Goal: make the 1k checkpoint easy to inspect before any larger Bench2Drive or
route-subset evaluation.

Changes made:

- `simlingo_training/eval.py` no longer has a hard-coded checkpoint path.
- `eval_load_path` selects a Lightning checkpoint from config.
- checkpoint run config is loaded from `.hydra/config.yaml`, then local eval
  overrides are re-applied.
- DeepSpeed conversion is imported lazily only when needed.
- one-GPU `strategy: auto` avoids unnecessary distributed setup.
- Lightning predict passes `weights_only=False` for trusted local checkpoints,
  avoiding PyTorch 2.6 OmegaConf checkpoint-load failures.
- `DrivingModel.on_predict_epoch_end` now writes generic MTID Dreamer stats for
  unknown/non-original Dreamer modes instead of leaving `dreamer_results` empty.
- `Eval_Dreamer` can run deterministically for fair baseline-vs-MTID
  comparison:
  `eval_deterministic`, `eval_seed`, `eval_option_policy`,
  `eval_safety_policy`, and `eval_prompt_policy`.
- prediction export now includes `mtid_samples_*_rank_*.json` with path, mode,
  allowed flag, prompt, predicted answer, target answer, and ADE values.

Tiny checkpoint smoke command:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python \
  simlingo_training/eval.py experiment=mtid_eval_1k
```

Smoke result:

- checkpoint loaded:
  `simlingo_training/outputs/2026_04_27_10_33_14_simlingo_mtid_1k/checkpoints/last.ckpt`
- predict batches: `2`
- output folder:
  `simlingo_training/outputs/2026_04_27_10_33_14_simlingo_mtid_1k/predictions`
- latest result file:
  `dreamer_results_rank_0_2026-04-27_20-36-32.json`

Latest two-sample generic MTID stats:

| Metric | Value |
| --- | ---: |
| instruction samples | 2 |
| mode | `dense_gap_yield` |
| allowed | `True` |
| route ADE to instruction | 0.1355 |
| waypoint ADE to instruction | 0.3700 |
| waypoint ADE to original | 7.2462 |
| waypoint closer-to-instruction rate | 1.0 |

Latest deterministic smoke summary command:

```bash
/home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python \
  mtid/tools/summarize_eval_predictions.py \
  simlingo_training/outputs/2026_04_27_10_33_14_simlingo_mtid_1k/predictions \
  --max-samples 3
```

Latest deterministic smoke result:

- result file: `dreamer_results_rank_0_2026-04-27_20-49-26.json`
- sample file: `mtid_samples_all_rank_0.json`
- samples: 2 deterministic `dense_gap_yield` rows
- all samples were instruction-following and allowed.

Follow-up fix:

- `eval.py` now re-applies all requested `data_module.base_dataset` overrides
  after loading the checkpoint's `.hydra/config.yaml`.
- This matters for commands such as
  `data_module.base_dataset.min_val_routes=2`; before the fix, checkpoint config
  reload silently reverted that override.

Latest eight-route eval command:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python \
  simlingo_training/eval.py \
  experiment=mtid_eval_1k \
  eval_load_path=simlingo_training/outputs/2026_04_27_10_33_14_simlingo_mtid_1k/checkpoints/last.ckpt \
  limit_predict_batches=200 \
  data_module.base_dataset.min_val_routes=8
```

Latest eight-route MTID result:

- result file: `dreamer_results_rank_0_2026-04-27_21-44-05.json`
- sample file: `mtid_samples_all_rank_0_2026-04-27_21-44-05.json`
- loaded validation routes: 8
- validation image pool: 592
- evaluated samples: 200
- modes: `dense_gap_yield=70`, `jaywalker_crossing=37`,
  `lane_less_corridor=93`
- allowed counts: `True=165`, `False=35`
- waypoint closer-to-instruction rate:
  - all: `0.530`
  - instruction: `0.450`
  - safety: `0.610`

Interpretation: the 1k checkpoint is useful for pipeline validation and shows a
real MTID waypoint shift, especially on safety prompts. It is still early and
not yet strong enough for a final paper claim.

## Training Runs

| Run | Purpose | Steps | Checkpoint | Final Metrics |
| --- | --- | ---: | --- | --- |
| `mtid_debug` | one-step Lightning smoke | 1 | none | `train/loss_step=27.073`, `val/loss=23.089` |
| `mtid_short` | pilot checkpoint only | 200 | yes | `train/loss_step=2.378`, `train/loss_epoch=6.896` |
| `mtid_1k` | first useful local checkpoint | 1000 | yes | `train/loss_step=2.492`, `train/loss_epoch=4.272`, `val/loss=2.921` |
| `mtid_10e` | current best MTID checkpoint | 10 epochs | yes | strongest offline MTID result |
| `internvl_driving_1k` | same-architecture driving-only control | 1000 | yes | `train/loss_step=1.899`, `train/loss_epoch=5.062`, `val/loss=9.859` |

`mtid_short` should not be resumed for serious training because its scheduler
was built for 200 steps and its learning rate had decayed near zero by the end.
Use a fresh longer schedule instead.

## Main Checkpoints

200-step pilot:

```text
simlingo_training/outputs/2026_04_27_10_28_25_simlingo_mtid_short_200/checkpoints/last.ckpt
```

1000-step local checkpoint:

```text
simlingo_training/outputs/2026_04_27_10_33_14_simlingo_mtid_1k/checkpoints/last.ckpt
```

10-epoch MTID checkpoint:

```text
outputs/2026_04_28_00_46_43_simlingo_mtid_10e/checkpoints/last.ckpt
```

Each full Lightning checkpoint is about `2.0G`. Keep these out of git.

## 1k Run Details

Command:

```bash
MTID_EXPERIMENT=mtid_1k \
PYTHON_BIN=/home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python \
  mtid/tools/run_mtid_short.sh
```

Output directory:

```text
simlingo_training/outputs/2026_04_27_10_33_14_simlingo_mtid_1k
```

Final metrics:

- `train/loss_step: 2.492`
- `train/loss_epoch: 4.272`
- `val/loss: 2.921`
- `val_losses/language_loss: 0.0389`
- `val_losses/route_loss: 0.7813`
- `val_losses/speed_wps_loss: 2.1088`

## Same-Architecture Control

The old checkpoints under root `outputs/` are not clean MTID comparisons
because they use the older SimLingo base stack. A fairer local control was
added as:

```text
simlingo_training/config/experiment/internvl_driving_1k.yaml
```

This control uses the same InternVL2-1B/LoRA training stack as `mtid_1k`, but
does not use the MTID Dreamer labels during training.

Training command:

```bash
MTID_EXPERIMENT=internvl_driving_1k \
PYTHON_BIN=/home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python \
  mtid/tools/run_mtid_short.sh
```

Output directory:

```text
simlingo_training/outputs/2026_04_27_21_34_30_simlingo_internvl_driving_1k
```

Checkpoint:

```text
simlingo_training/outputs/2026_04_27_21_34_30_simlingo_internvl_driving_1k/checkpoints/last.ckpt
```

Final metrics:

- `train/loss_step: 1.899`
- `train/loss_epoch: 5.062`
- `val/loss: 9.859`
- `val_losses/language_loss: 0.0057`
- `val_losses/route_loss: 1.2053`
- `val_losses/speed_wps_loss: 8.6480`

Eval command on the same deterministic MTID subset:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python \
  simlingo_training/eval.py \
  experiment=mtid_eval_1k \
  eval_load_path=simlingo_training/outputs/2026_04_27_21_34_30_simlingo_internvl_driving_1k/checkpoints/last.ckpt \
  limit_predict_batches=200 \
  data_module.base_dataset.min_val_routes=8
```

Prediction output:

```text
simlingo_training/outputs/2026_04_27_21_34_30_simlingo_internvl_driving_1k/predictions
```

## Offline Comparison (500 samples)

Both checkpoints were evaluated on the same deterministic MTID subset expanded to 500 samples from 8 validation routes, with mode counts `dense_gap_yield=233`, `jaywalker_crossing=39`, and `lane_less_corridor=228`. 

| Metric | MTID 10e | MTID 1k | Driving-only control | Better |
| --- | ---: | ---: | ---: | --- |
| waypoint ADE to instruction | 3.230 | 4.038 | 5.073 | MTID 10e |
| waypoint ADE to original | 5.109 | 7.641 | 2.738 | control |
| waypoint closer-to-instruction rate | 0.578 | 0.616 | 0.238 | MTID |
| route ADE to instruction | 0.872 | 0.524 | 0.634 | MTID 1k |

### Qualitative Analysis (Safe vs. Unsafe Prompts)

A deeper look into the predictions shows a striking difference in performance based on the prompt type (`allowed=True` vs. `allowed=False`), particularly after fixing a critical data generation bug:

- **Safe (allowed=True, N=374)**: MTID 10e ADE 3.732 | MTID 1k ADE 3.395 | Control ADE 5.789  **(MTID maintains strong performance)**
- **Unsafe/Yield (allowed=False, N=126)**: MTID 10e ADE 2.729 | MTID 1k ADE 5.949 | Control ADE 2.951  **(MTID 10e is now BETTER than control)**

Interpretation:
- After identifying and fixing a bug where `unsafe_instructions` were missing in the generation templates (causing the model to mistakenly associate safe yield commands with unsafe forward-driving waypoints), the MTID 10e model successfully learns the correct yield behavior.
- MTID 10e completely fixes the Unsafe/Yield failure, bringing the ADE down dramatically from 5.949 to 2.729, which actively outperforms the Driving-only control (2.951). The model successfully outputs "Ignore instruction..." in text and predicts a safe yielding trajectory when faced with dangerous commands.
- The model now correctly distinguishes between safe continuation and unsafe situations requiring a yield, showing a clear, intended behavioral shift offline.

## CARLA / Bench2Drive Smoke

Goal: move from offline MTID prediction metrics to closed-loop CARLA evidence.

Runtime fixes completed on 2026-04-28:

- `run_benchmark_local.py` now defaults to
  `mtid/routes/routes_mtid_smoke.xml` and removes stale result JSON before fresh
  runs. Devtest/full routes can be selected with `SIMLINGO_BENCHMARK_ROUTES`.
- `Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py` launches CARLA
  with `-quality-level=Low` and uses `grep --` during cleanup.
- `team_code/agent_simlingo.py` supports the current `DrivingModel.forward`
  return shape: `(speed_wps, route, language)`.
- CARLA runs direct driving mode by default:
  `SIMLINGO_CARLA_PREDICT_LANGUAGE=0`. This avoids language-generation OOM on
  the local 11.48 GiB GPU.
- Optional control tracing can be enabled with
  `SIMLINGO_CARLA_DEBUG_CONTROL=1` and
  `SIMLINGO_CARLA_DEBUG_CONTROL_FREQ=20` to inspect speed, throttle, brake,
  steering, target point, and first predicted route/waypoint.
- Control tracing now also logs PID/debug state:
  `desired_speed`, `delta`, `stuck`, `force_move`, `gps`, and `compass`.
  `mtid/tools/summarize_carla_control_trace.py` is backward-compatible with old
  logs and reports desired-speed/delta summaries when the new fields are
  present.
- `run_benchmark_local.py` accepts `SIMLINGO_BENCHMARK_CHECKPOINT`,
  `SIMLINGO_BENCHMARK_RESULT`, and `SIMLINGO_BENCHMARK_RESUME` so the same
  smoke route can compare MTID and control checkpoints without editing the
  script.
- The CARLA agent now uses the correct InternVL/Qwen tokenizer. Before the fix,
  `<IMG_CONTEXT>` resolved to token id `0`; after the fix it resolves to
  `151648` and the first CARLA prompt contains `512` image-context tokens.
- `simlingo_training/models/driving.py` direct forward mode now uses the same
  adaptor/model path as training without treating `(features, logits)` as a
  tensor.
- `simlingo_training/models/encoder/internvl2_model.py` prints a one-time
  warning if image-context tokens are missing, instead of repeatedly failing.
- The control trace print no longer crashes when `gt_velocity` is a tensor.
- `team_code/agent_simlingo.py` now returns the applied `self.control` instead
  of the pre-guard `control`. Before this fix, the initial-frame brake guard was
  logged through `self.control` but CARLA could receive the unguarded command.

Validated smoke evidence:

- the 10e checkpoint loads inside CARLA;
- the agent enters the CARLA route;
- image features are inserted after the tokenizer fix;
- no CUDA OOM occurred in direct driving mode with CARLA low-quality rendering;
- control tracing produced a readable stuck-run signature;
- a stale partial `outputs/benchmark_mtid_results.json` should be treated as
  partial unless the evaluator exits cleanly.

Latest MTID 10e control trace command after the applied-control fix:

```bash
SIMLINGO_CARLA_DEBUG_CONTROL=1 \
SIMLINGO_CARLA_DEBUG_CONTROL_FREQ=5 \
SIMLINGO_BENCHMARK_RESULT=outputs/benchmark_mtid_results_after_control_fix.json \
/home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python \
  run_benchmark_local.py 2>&1 | tee benchmark_output_after_control_fix.log

/home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python \
  mtid/tools/summarize_carla_control_trace.py benchmark_output_after_control_fix.log
```

Latest MTID 10e trace summary after the applied-control fix:

| Metric | Value |
| --- | ---: |
| Control samples | 71 |
| Step range | 5 to 355 |
| Speed mean | 0.873 |
| Speed max | 6.600 at step 105 |
| Full throttle samples | 58 / 71 |
| Full brake samples | 9 / 71 |
| First stuck step | 135 |
| Target distance mean / max | 38.972 / 50.966 |
| Predicted route0 norm mean | 0.013 |
| Predicted waypoint0 norm mean | 0.443 |
| Predicted waypoint0 norm last10 mean | 0.297 |

Interpretation:

- closed-loop loading, image-token insertion, and direct forward inference are
  working;
- after returning `self.control`, the initial-frame brake guard is actually
  applied and speed stays `0` during the initial delay;
- MTID 10e then drives for a few seconds, reaches about `6.6 m/s`, and later
  becomes stationary with full throttle;
- the target remains far away while predicted waypoints shrink close to the ego,
  so MTID waypoint calibration and controller recovery remain the next debug
  targets;
- the driving-only control comparison after the applied-control fix also gets
  stuck, so the current failure is shared by MTID and the control checkpoint;
- because both checkpoints stall on the same smoke route, the next suspect set
  is route/controller behavior, coordinate/preprocess parity, and base driving
  data quality rather than an MTID-only regression.

Driving-only control comparison after the applied-control fix:

```bash
SIMLINGO_CARLA_DEBUG_CONTROL=1 \
SIMLINGO_CARLA_DEBUG_CONTROL_FREQ=5 \
SIMLINGO_BENCHMARK_CHECKPOINT=simlingo_training/outputs/2026_04_27_21_34_30_simlingo_internvl_driving_1k/checkpoints/last.ckpt \
SIMLINGO_BENCHMARK_RESULT=outputs/benchmark_control_results_after_control_fix.json \
/home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python \
  run_benchmark_local.py 2>&1 | tee benchmark_control_after_control_fix.log

/home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python \
  mtid/tools/summarize_carla_control_trace.py benchmark_control_after_control_fix.log
```

Control trace summary:

| Metric | Value |
| --- | ---: |
| Control samples | 261 |
| Step range | 5 to 1305 |
| Speed mean | 0.218 |
| Speed max | 9.000 at step 85 |
| Full throttle samples | 249 / 261 |
| Full brake samples | 7 / 261 |
| First stuck step | 110 |
| Target distance mean / max | 39.237 / 50.966 |
| Predicted route0 norm mean | 0.337 |
| Predicted waypoint0 norm mean | 0.577 |
| Predicted waypoint0 norm last10 mean | 0.555 |

Current caveat:

- the short smoke route did not finish quickly in the local run, so there is not
  yet a valid Driving Score or Route Completion JSON.
- next closed-loop work should debug the shared route/controller/preprocess/data
  quality issue before running `routes_devtest.xml` or the full Bench2Drive set.

## Base Driving Data Quality Audit

This audit was added because both MTID 10e and the same-architecture
driving-only control become stationary in CARLA, and the original data
collection was suspected to include slow, stuck, and off-road behavior.

Command:

```bash
/home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python \
  mtid/tools/audit_driving_data_quality.py \
  --dataset-root database/simlingo_v2_all \
  --output mtid/outputs/debug/driving_data_quality_report.json
```

Result:

| Metric | Value |
| --- | ---: |
| Routes | 44 |
| Frames | 8031 |
| Speed `<0.1 m/s` | 0.336 |
| Speed `<0.5 m/s` | 0.346 |
| Speed `<1.0 m/s` | 0.349 |
| Displacement `<0.05 m/sample` | 0.337 |
| Displacement `<0.20 m/sample` | 0.344 |
| Non-driving ego lane fraction | 0.002 |
| Route mean speed | 7.627 |
| Route mean displacement/sample | 1.911 |

Worst slow/stuck routes:

| Slow Fraction `<0.5 m/s` | Frames | Route |
| ---: | ---: | --- |
| 1.000 | 3 | `data/simlingo/training_1_scenario/routes_training/random_weather_seed_1_balanced_150/Town12_Rep0_2706_route0_04_23_23_36_49` |
| 0.942 | 1094 | `data/simlingo/training_1_scenario/routes_training/random_weather_seed_1_balanced_150/Town12_Rep0_3604_route0_04_23_00_45_03` |
| 0.734 | 139 | `data/simlingo/lb1_split/routes_training/SignalizedJunctionLeftTurn/Town04_Rep0_Town04_Scenario7_67_route0_04_23_23_25_19` |
| 0.683 | 139 | `data/simlingo/lb1_split/routes_training/OppositeVehicleRunningRedLight/Town03_Rep0_Town03_Scenario8_34_route0_04_23_22_49_36` |
| 0.617 | 449 | `data/simlingo/training_1_scenario/routes_training/random_weather_seed_1_balanced_150/Town12_Rep0_4885_route0_04_23_22_55_03` |

Interpretation:

- the user's suspicion is supported: the base driving dataset has a large
  slow/stuck component, roughly one third of all frames;
- the low non-driving lane fraction means the measured problem is mostly
  slow/stuck behavior rather than lane-type labels saying the ego is on a
  sidewalk;
- before collecting more data, use the clean subset below to retrain a
  driving-only control and check whether CARLA smoke movement improves.

## Clean Driving Dataset View

The clean view keeps `database/simlingo_v2_all` unchanged and creates route
directory symlinks under `database/simlingo_v2_all_clean`.

Build command:

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

Current result:

| Metric | Value |
| --- | ---: |
| Input routes | 44 |
| Kept routes | 32 |
| Kept frames | 4531 |
| Excluded routes | 12 |
| Excluded frames | 3500 |
| Linked data routes | 32 |
| Linked MTID Dreamer routes | 30 |
| Missing MTID Dreamer route mirrors | 2 |

Excluded reason counts:

| Reason | Count |
| --- | ---: |
| `too_many_slow_speed_frames` | 11 |
| `too_many_stuck_displacements` | 11 |
| `too_few_frames` | 1 |
| `too_many_non_driving_lane_frames` | 1 |

Clean driving-only smoke command:

```bash
/home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python \
  mtid/tools/smoke_mtid_pipeline.py \
  --experiment internvl_driving_clean_1k \
  --dataset-mode driving \
  --batch-size 1 \
  --num-workers 0
```

Smoke result:

- train routes discovered: `32`, used by split: `31`
- train samples: `3600`
- val samples: `450`
- train camera tensor: `(1, 1, 2, 3, 448, 448)`
- train waypoints tensor: `(1, 10, 2)`
- train route/path tensor: `(1, 20, 2)`

## Clean Driving Control Result

Training command:

```bash
MTID_EXPERIMENT=internvl_driving_clean_1k \
PYTHON_BIN=/home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python \
  mtid/tools/run_mtid_short.sh
```

Training artifact:

```text
simlingo_training/outputs/2026_04_28_23_35_11_simlingo_internvl_driving_clean_1k/checkpoints/last.ckpt
```

Training summary:

| Metric | Value |
| --- | ---: |
| Last step | 999 |
| Train loss epoch | 5.583 |
| Val loss | 9.426 |
| Train speed/waypoint loss | 5.361 |
| Val speed/waypoint loss | 6.789 |

500-sample deterministic MTID offline eval:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python \
  simlingo_training/eval.py \
  experiment=mtid_eval_1k \
  eval_load_path=simlingo_training/outputs/2026_04_28_23_35_11_simlingo_internvl_driving_clean_1k/checkpoints/last.ckpt \
  limit_predict_batches=500 \
  data_module.base_dataset.min_val_routes=8 \
  name=simlingo_internvl_driving_clean_1k_eval_500b
```

Offline comparison on the same 500 MTID samples:

| Metric | MTID 10e | Driving-only control | Clean driving-only |
| --- | ---: | ---: | ---: |
| Waypoint ADE to instruction | 3.230 | 5.074 | 7.844 |
| Waypoint ADE to original | 5.109 | 2.738 | 5.454 |
| Waypoint closer-to-instruction rate | 0.578 | 0.238 | 0.238 |
| Route ADE to instruction | 0.872 | 0.634 | 0.623 |
| Route ADE to original | 0.842 | 0.583 | 0.549 |
| Route closer-to-instruction rate | 0.848 | 0.816 | 0.782 |

Interpretation:

- the route-level clean subset does not improve offline MTID instruction
  response;
- MTID 10e is still the best offline model for interaction labels;
- clean-control remains useful only as a data-quality/closed-loop ablation.

Clean-control CARLA trace command:

```bash
SIMLINGO_BENCHMARK_CHECKPOINT=simlingo_training/outputs/2026_04_28_23_35_11_simlingo_internvl_driving_clean_1k/checkpoints/last.ckpt \
SIMLINGO_BENCHMARK_RESULT=outputs/benchmark_clean_control_results.json \
SIMLINGO_CARLA_DEBUG_CONTROL=1 \
SIMLINGO_CARLA_DEBUG_CONTROL_FREQ=5 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python \
  run_benchmark_local.py 2>&1 | tee benchmark_clean_control.log
```

The run was manually stopped after clear stuck behavior, so
`outputs/benchmark_clean_control_results.json` is only a partial benchmark file.

Trace summary from `benchmark_clean_control.log`:

| Metric | Value |
| --- | ---: |
| Control samples | 109 |
| Step range | 5 -> 545 |
| Speed mean | 0.599 |
| Speed max | 7.500 at step 75 |
| Full throttle samples | 96 / 109 |
| Full brake samples | 8 / 109 |
| First stuck step | 120 |
| Target distance mean / max | 37.360 / 50.966 |
| Predicted route0 norm mean | 0.192 |
| Predicted waypoint0 norm mean | 1.705 |
| Predicted waypoint0 norm last10 mean | 1.656 |

Interpretation:

- clean-control still becomes stationary despite sustained `throttle=1.0`;
- the issue is not fixed by route-level removal of slow/stuck training routes;
- next work should focus on physical contact/off-route state, controller target
  conversion, route geometry, and CARLA vehicle dynamics rather than collecting
  more data immediately.

## Collision Debug Result

Added optional CARLA collision/off-lane debug instrumentation:

- enable with `SIMLINGO_CARLA_DEBUG_COLLISION=1`;
- collision events print as `CARLA collision ...`;
- control lines now include `world_speed`, `lane_dist`, `road_id`, `lane_id`,
  `lane_type`, `ego_z`, `ego_pitch`, `ego_roll`, `collision_count`, and
  `last_collision`;
- `mtid/tools/summarize_carla_control_trace.py` summarizes collision events and
  lane/world-speed fields.

Command used:

```bash
SIMLINGO_CARLA_DEBUG_CONTROL=1 \
SIMLINGO_CARLA_DEBUG_CONTROL_FREQ=5 \
SIMLINGO_CARLA_DEBUG_COLLISION=1 \
SIMLINGO_BENCHMARK_RESULT=outputs/benchmark_mtid_collision_debug_results.json \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python \
  run_benchmark_local.py 2>&1 | tee benchmark_mtid_collision_debug.log
```

Summary:

| Metric | Value |
| --- | ---: |
| Control samples | 54 |
| Speed max | 7.200 at step 100 |
| First stuck step | 125 |
| Lane distance max | 1.368 |
| Collision events | 150 |
| Max collision | step 118, `static.prop.mesh`, intensity 10924.550 |
| Last collision | step 270, `static.prop.mesh`, intensity 143.497 |

Key evidence:

- step `100`: ego is still in `Driving` lane, but lane distance is already
  `1.368`;
- step `105`: ego is in `Parking` lane;
- step `118`: first static collision occurs with `static.prop.mesh`;
- step `125+`: ego is physically stuck with full throttle and repeated static
  collisions.

Interpretation:

- yes, the smoke-route stationary behavior is caused by a real collision/contact
  with a static mesh;
- the upstream cause is likely lateral route/controller drift into a parking
  lane before the collision, not MTID language behavior or lack of data alone.

## Planner-Steering Ablation

Hypothesis from visual/trace inspection:

- during avoidance, model-route steering moves the ego too far laterally;
- the vehicle enters a `Parking` lane / curb-side region and hits a static mesh.

Added a CARLA steering-source switch:

- default: `SIMLINGO_CARLA_STEER_SOURCE=model`;
- ablation: `SIMLINGO_CARLA_STEER_SOURCE=planner`;
- planner mode keeps steering on the route-planner target points while still
  using model-predicted speed waypoints for longitudinal control.

Command:

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

Trace summary:

| Metric | Model steering collision run | Planner steering run |
| --- | ---: | ---: |
| Control samples | 54 | 67 |
| Speed max | 7.200 | 6.400 |
| First stuck step | 125 | n/a |
| Lane distance max | 1.368 | 0.054 |
| Collision events | 150 | 0 |
| Max collision | step 118, `static.prop.mesh`, 10924.550 | n/a |

Interpretation:

- the user's observation is correct: the agent was effectively avoiding too far
  and entering the parking/curb-side region;
- planner steering prevents that specific failure in the tested segment;
- for near-term closed-loop smoke, use planner steering as a guardrail;
- for a paper-quality fix, analyze why predicted route steering drifts outside
  the drivable lane during avoidance.

## Completed Planner-Steering Smoke

Command:

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

Result JSON:

```text
outputs/benchmark_mtid_planner_steer_full_results.json
```

Bench2Drive result:

| Metric | Value |
| --- | ---: |
| Route status | Completed |
| Driving Score | 100.0 |
| Route Completion | 100.0 |
| Infraction penalty | 1.0 |
| Collisions | 0 |
| Outside route lanes | 0 |
| Route deviations | 0 |
| Agent blocked | 0 |
| Route timeout | 0 |
| Min-speed infractions | 91.143 |

Control trace summary:

| Metric | Value |
| --- | ---: |
| Control samples | 44 |
| Step range | 20 -> 880 |
| Speed mean/max | 4.091 / 6.200 |
| First stuck step | n/a |
| Lane distance mean/max | 0.025 / 0.059 |
| Collision count max | 0 |
| World speed mean/max | 4.095 / 6.171 |
| PID desired speed mean/max | 4.327 / 6.325 |

Interpretation:

- planner steering is enough to complete the short smoke route cleanly;
- the previous stationary behavior was collision/contact from lateral drift,
  not a fundamental CARLA physics issue;
- the remaining closed-loop issue is speed: the model-controlled longitudinal
  behavior is conservative enough to trigger repeated min-speed infractions.

## Runtime Speed Calibration Ablation

Added CARLA-only longitudinal calibration knobs in `team_code/agent_simlingo.py`:

```bash
SIMLINGO_CARLA_SPEED_SCALE=1.25
SIMLINGO_CARLA_MIN_DESIRED_SPEED=3.0
SIMLINGO_CARLA_MAX_DESIRED_SPEED=8.0
```

Defaults preserve previous behavior:

```text
SPEED_SCALE=1.0
MIN_DESIRED_SPEED=0.0
MAX_DESIRED_SPEED=0.0
```

The control trace now logs `desired_speed_raw`, `speed_scale`, and
`min_desired_speed`; `mtid/tools/summarize_carla_control_trace.py` summarizes
those fields.

Smoke-route ablations:

| Run | Result JSON | Game time | Speed mean/max | Collision/off-route | MinSpeedTest |
| --- | --- | ---: | ---: | ---: | ---: |
| planner steer, default speed | `outputs/benchmark_mtid_planner_steer_full_results.json` | 44.25s | 4.091 / 6.200 | 0 / 0 | fail, 77.83% console |
| planner steer, `SPEED_SCALE=1.25` | `outputs/benchmark_mtid_planner_steer_speed125_results.json` | 38.85s | 4.579 / 7.400 | 0 / 0 | fail, 90.44% console |
| planner steer, `SPEED_SCALE=1.25`, `MIN_DESIRED_SPEED=3.0`, `MAX_DESIRED_SPEED=8.0` | `outputs/benchmark_mtid_planner_steer_speed125_min3_results.json` | 35.20s | 5.220 / 8.000 | 0 / 0 | fail, 93.21% console |

Interpretation:

- planner steering remains physically stable under faster longitudinal control;
- speed scaling/flooring reduces route duration but does not pass
  `MinSpeedTest`;
- the speed issue is not solved by a simple runtime multiplier alone;
- `MinSpeedTest` is an efficiency metric in this Bench2Drive checkout, not a
  score blocker. `MinimumSpeedRouteTest._set_traffic_event()` records a
  `MIN_SPEED_INFRACTION` at each checkpoint and sets the criterion to failure,
  but `leaderboard/utils/statistics_manager.py` marks
  `TrafficEventType.MIN_SPEED_INFRACTION` as penalty type `unused`.
- route score and infraction penalty therefore stay `100.0` and `1.0` when
  min-speed is the only reported issue.
- speed calibration should remain an efficiency ablation. It is not required
  before moving from smoke to a small route subset.

## Devtest ParkingExit Trace

Command:

```bash
SIMLINGO_BENCHMARK_ROUTES=leaderboard/data/routes_devtest.xml \
SIMLINGO_CARLA_STEER_SOURCE=planner \
SIMLINGO_CARLA_DEBUG_CONTROL=1 \
SIMLINGO_CARLA_DEBUG_CONTROL_FREQ=50 \
SIMLINGO_CARLA_DEBUG_COLLISION=1 \
SIMLINGO_BENCHMARK_RESULT=outputs/benchmark_mtid_devtest_planner_results.json \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python \
  run_benchmark_local.py 2>&1 | tee benchmark_mtid_devtest_planner.log
```

The run was stopped manually after route 0 showed a stable failure pattern. The
result JSON therefore has `entry_status: Started`, `progress: [0, 2]`, and no
final route records. Use the log as a debug trace, not as an official score.

Added focused reproduction route:

```text
mtid/routes/routes_mtid_parking_exit.xml
```

This file keeps the same first two waypoints as `routes_mtid_smoke.xml`, then
adds the exact `ParkingExit_1` scenario from `routes_devtest.xml`. Use it to
debug the ParkingExit interaction without running both devtest routes.

Observed scenario:

```text
RouteScenario_0_rep0_Town12_ParkingExit_1_0
```

Control trace summary:

| Metric | Value |
| --- | ---: |
| Control snapshots | 25 |
| Step range | 50 -> 1250 |
| Speed mean/max | 0.000 / 0.000 |
| Full throttle samples | 25 / 25 |
| First stuck step | 50 |
| World speed mean/max | 0.014 / 0.035 |
| Lane distance mean/max | 0.362 / 0.863 |
| Collision events | 1188 |
| Debug collision count max | 1170 |
| Max collision | step 79, `vehicle.mercedes.coupe_2020`, intensity 4701.276 |
| Last collision | step 1267, `vehicle.mercedes.coupe_2020`, intensity 81.695 |

Representative control state:

```text
step=1250 speed=0.000 throttle=1.000 brake=0.000 steer=-0.971
lane_type=Parking road_id=529 lane_id=-2
desired_speed=1.119 collision_count=1170
```

Interpretation:

- this devtest failure is not the same as the earlier smoke-route static mesh
  collision;
- planner steering fixed the short route, but route 0 of `routes_devtest.xml`
  starts in a parking-exit interaction where ego remains in `Parking` lane and
  collides repeatedly with a real vehicle actor;
- the vehicle is not stationary because the controller refuses to accelerate:
  it is commanding full throttle, but physical contact keeps world speed near
  zero;
- `ParkingExit` intentionally teleports ego into a parking lane and spawns
  front, behind, and side vehicles. The privileged autopilot has a
  `starts_with_parking_exit` route-planner workaround, but `agent_simlingo.py`
  currently uses the simpler `RoutePlanner` path without the same adjustment;
- the next debugging target is scenario geometry and interaction handling:
  spawn pose, parked/exiting actor pose, route target direction, and whether the
  planner target path requires yielding instead of pushing through the actor.

## ParkingExit Route-Fix Reproduction

Added optional route-start handling in `team_code/agent_simlingo.py` behind:

```bash
SIMLINGO_CARLA_PARKING_EXIT_ROUTE_FIX=1
```

The fix detects when the ego starts more than `2m` away from the first global
route waypoint, prepends the current ego pose, and inserts a lane-change command
toward the route lane. The corrected version also applies the route planner's
GPS-to-CARLA coordinate offset before passing world-coordinate transforms into
`RoutePlanner`.

Finalized command:

```bash
timeout --kill-after=30s 10m bash -lc '\
SIMLINGO_BENCHMARK_ROUTES=mtid/routes/routes_mtid_parking_exit.xml \
SIMLINGO_CARLA_STEER_SOURCE=planner \
SIMLINGO_CARLA_PARKING_EXIT_ROUTE_FIX=1 \
SIMLINGO_CARLA_DEBUG_CONTROL=1 \
SIMLINGO_CARLA_DEBUG_CONTROL_FREQ=20 \
SIMLINGO_CARLA_DEBUG_COLLISION=1 \
SIMLINGO_BENCHMARK_RESULT=outputs/benchmark_mtid_parking_exit_routefix2_results.json \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python \
  run_benchmark_local.py 2>&1 | tee benchmark_mtid_parking_exit_routefix2.log'
```

Finalized leaderboard result:

| Metric | Value |
| --- | ---: |
| Route completion | 100.0 |
| Driving score | 21.6 |
| Infraction penalty | 0.216 |
| Duration game/system | 93.9s / 326.36s |
| Registered vehicle collisions | 3 |
| Min-speed infraction aggregate | 91.143 |

Control-trace summary:

| Metric | Value |
| --- | ---: |
| Control snapshots | 93 |
| Step range | 20 -> 1860 |
| Speed mean/max | 2.014 / 6.100 |
| Full throttle samples | 65 / 93 |
| Full brake samples | 12 / 93 |
| First stuck step | 1220 |
| Target distance mean/max | 19.849 / 51.868 |
| Debug collision events | 933 |
| Debug collision count max | 923 |
| Max collision | step 1210, `vehicle.mercedes.coupe_2020`, intensity 8766.942 |
| Last collision | step 1869, `vehicle.mercedes.coupe_2020`, intensity 36.355 |

Interpretation:

- the original raw-world-coordinate route fix was wrong because the route
  planner expects its converted CARLA frame. That produced target distances near
  `970m` and route deviation.
- the corrected frame-aware route fix works as a planner geometry fix: the ego
  leaves the Parking lane, enters the Driving lane, and reaches `100%` route
  completion.
- the focused route still fails because the ego collides with
  `vehicle.mercedes.coupe_2020` instead of yielding/creeping safely through the
  ParkingExit interaction.
- the next blocker is therefore interaction behavior, not route frame
  conversion. A pure route-start fix is insufficient for this scenario.

## ParkingExit Yield/Creep Diagnostic

Added an optional diagnostic guard in `team_code/agent_simlingo.py` behind:

```bash
SIMLINGO_CARLA_PARKING_EXIT_YIELD=1
SIMLINGO_CARLA_PARKING_EXIT_YIELD_STEPS=900
SIMLINGO_CARLA_PARKING_EXIT_YIELD_BRAKE_STEPS=160
SIMLINGO_CARLA_PARKING_EXIT_YIELD_DISTANCE=25
SIMLINGO_CARLA_PARKING_EXIT_CREEP_THROTTLE=0.25
```

The guard is active only when `SIMLINGO_CARLA_PARKING_EXIT_ROUTE_FIX=1` is also
enabled. It scans nearby CARLA `vehicle.*` actors, filters by true Euclidean
distance, yields during the initial merge window, creeps while still in a
Parking lane, and suppresses the existing stuck `force_move` override while a
hazard is active.

Experiments:

| Run | Result |
| --- | --- |
| `benchmark_mtid_parking_exit_yield.log` | Over-yielded to continuous merge-side traffic and behaved like a deadlock. |
| `benchmark_mtid_parking_exit_yield2.log` | Exposed a false-positive distant hazard after entering Driving lane; fixed by adding a real distance threshold. |
| `benchmark_mtid_parking_exit_yield3.log` | Avoided the initial Parking-lane collision and entered Driving lane, then stopped behind real lead traffic; when the guard expired, the original stuck/force-move behavior pushed into vehicles. |

Latest command:

```bash
timeout --kill-after=30s 10m bash -lc '\
SIMLINGO_BENCHMARK_ROUTES=mtid/routes/routes_mtid_parking_exit.xml \
SIMLINGO_CARLA_STEER_SOURCE=planner \
SIMLINGO_CARLA_PARKING_EXIT_ROUTE_FIX=1 \
SIMLINGO_CARLA_PARKING_EXIT_YIELD=1 \
SIMLINGO_CARLA_PARKING_EXIT_YIELD_STEPS=900 \
SIMLINGO_CARLA_PARKING_EXIT_YIELD_BRAKE_STEPS=160 \
SIMLINGO_CARLA_PARKING_EXIT_YIELD_DISTANCE=25 \
SIMLINGO_CARLA_DEBUG_CONTROL=1 \
SIMLINGO_CARLA_DEBUG_CONTROL_FREQ=20 \
SIMLINGO_CARLA_DEBUG_COLLISION=1 \
SIMLINGO_BENCHMARK_RESULT=outputs/benchmark_mtid_parking_exit_yield3_results.json \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python \
  run_benchmark_local.py 2>&1 | tee benchmark_mtid_parking_exit_yield3.log'
```

Trace summary:

| Metric | Value |
| --- | ---: |
| Final status | timeout, JSON `entry_status: Started` |
| Control snapshots | 129 |
| Step range | 20 -> 2580 |
| Speed mean/max | 0.164 / 4.800 |
| Full throttle samples | 84 / 129 |
| Full brake samples | 38 / 129 |
| First stuck step | 960 |
| Target distance mean/max | 36.277 / 50.108 |
| Pred route0 norm mean | 0.020 |
| Pred wps0 norm mean | 0.242 |
| PID desired speed mean/max | 1.130 / 4.563 |
| World speed mean/max | 0.195 / 4.754 |
| Lane distance mean/max | 0.346 / 1.484 |
| Debug stuck counter max | 1218 |
| Debug force_move max | 14 |
| Collision events | 1612 |
| Debug collision count max | 1599 |
| Max collision | step 939, `vehicle.ford.mustang`, intensity 8250.680 |
| Last collision | step 2592, `vehicle.mercedes.coupe_2020`, intensity 59.587 |

Interpretation:

- the early ParkingExit crash is avoidable with simple actor-aware yielding;
- the remaining failure is not throttle or CARLA physics alone: after a clean
  merge, the ego detects real lead traffic and stops;
- when the diagnostic guard expires, stuck recovery pushes the ego through the
  detected vehicle, causing repeated collisions;
- keeping the guard active longer would likely avoid collision but create an
  AgentBlocked/deadlock case;
- the next implementation should be a real lead-vehicle following/yield policy
  or additional training data for ParkingExit-style interactions, not another
  forced merge.

## ParkingExit Route Geometry Diagnostic

Added a route-fix geometry debug string in `team_code/agent_simlingo.py`. It is
printed once when `SIMLINGO_CARLA_PARKING_EXIT_ROUTE_FIX=1` activates and shows
ego lane, route lane, and immediate left/right neighbor lanes.

Short diagnostic command:

```bash
timeout --kill-after=30s 4m bash -lc '\
SIMLINGO_BENCHMARK_ROUTES=mtid/routes/routes_mtid_parking_exit.xml \
SIMLINGO_CARLA_STEER_SOURCE=planner \
SIMLINGO_CARLA_PARKING_EXIT_ROUTE_FIX=1 \
SIMLINGO_CARLA_DEBUG_CONTROL=1 \
SIMLINGO_CARLA_DEBUG_CONTROL_FREQ=100 \
SIMLINGO_BENCHMARK_RESULT=outputs/benchmark_mtid_parking_exit_geometry_results.json \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/vivu/miniconda3/envs/simlingo_rtx5070/bin/python \
  run_benchmark_local.py 2>&1 | tee benchmark_mtid_parking_exit_geometry.log'
```

Key line:

```text
ParkingExit route fix active: start_distance=3.203 lateral_offset=3.145 merge_command=CHANGELANELEFT route_min_distance=1.000 coordinate_offset=(-983.593,-0.000,-0.500) geometry=ego=road529/lane-2/Parking/yaw90.1/width2.5/left:road529/lane-1/Driving/right:road529/lane-3/Shoulder;route0=road529/lane-1/Driving/yaw90.1/width3.0/left:road529/lane1/Driving/right:road529/lane-2/Parking
```

Interpretation:

- ego starts in `road529/lane-2/Parking`;
- the intended first route lane is `road529/lane-1/Driving`;
- ego's left neighbor is exactly that route lane;
- `merge_command=CHANGELANELEFT` is therefore correct at scenario start;
- the route-fix side is not the primary bug. The remaining issue is vehicle
  interaction in/near the target lane.

## ParkingExit Merge-Steer and Static-Nudge Tests

Added hazard actor speed/lane debug to `parking_exit_yield=...` control lines.
This showed that early hazards are real traffic in `road529/lane-1/Driving`
with non-zero speed, so the initial yield is valid:

```text
step=20 parking_exit_yield=yield:...:v2.57:road529:lane-1:Driving
step=80 parking_exit_yield=yield:...:v6.12:road529:lane-1:Driving
```

Then added a merge-steer override while the guard is creeping in a Parking
lane:

```bash
SIMLINGO_CARLA_PARKING_EXIT_MERGE_STEER=0.8
```

For this ParkingExit instance, `merge_command=CHANGELANELEFT`, so Parking-lane
creep uses `steer=-0.8`. This fixes a concrete guard bug where the creep phase
could reuse planner steer and turn back into a Parking lane.

Merge-steer trace:

```text
benchmark_mtid_parking_exit_mergesteer.log
```

Result:

- ego enters and stays in `road529/lane-1/Driving`;
- no collision in the inspected window;
- summary: `31` control samples, step `20 -> 620`, collision count max `0`;
- it stops behind a stationary `vehicle.mercedes.coupe_2020` around `5.18m`
  ahead;
- the stationary actor has `v0.00` and projects to `road529/lane-1/Driving`;
- this is safer than routefix2, but it deadlocks because the route has no
  trajectory shift around the parked/blocking actor.

Also added a default-off static nudge diagnostic:

```bash
SIMLINGO_CARLA_PARKING_EXIT_NUDGE_STATIC=1
```

Nudge trace:

```text
benchmark_mtid_parking_exit_nudge.log
```

Result:

- the local nudge began colliding with `vehicle.mercedes.coupe_2020` around
  step `400+`;
- repeated collisions continued while `parking_exit_yield=static_nudge:...`;
- summary: `27` control samples, `181` collision events, max collision at step
  `369` with intensity `1117.278`;
- this mode is unsafe and should stay off unless used only for controlled
  debugging.

Conclusion: the next useful implementation is not a larger local nudge. It is
either a proper route/trajectory shift around the parked actor, modeled after
privileged planner obstacle handling, or training data that teaches the model to
perform the ParkingExit maneuver without clipping the parked vehicle.

## ParkingExit Waypoint-Hazard Diagnostic

Added a waypoint-vs-hazard diagnostic to `team_code/agent_simlingo.py`:

```bash
SIMLINGO_CARLA_DEBUG_WAYPOINT_HAZARD=1
```

When enabled together with `SIMLINGO_CARLA_DEBUG_CONTROL=1`, each `CARLA
control` line logs:

- current hazard distance, longitudinal/lateral position, speed, and static
  flag;
- minimum clearance from the hazard to `pred_route`;
- minimum clearance from the hazard to `pred_speed_wps`;
- minimum clearance from the hazard to the planner route input.

Extended `mtid/tools/summarize_carla_control_trace.py` to summarize these
clearances, including a separate static-hazard line.

Trace:

```text
benchmark_mtid_parking_exit_waypoint_hazard.log
```

Summary:

```text
Control samples: 24
Step range: 20 -> 480
Debug collision_count max: 0
Hazard debug samples: 19/24 static: 9
Model route hazard clearance mean/min: 2.417 / 0.633
Speed wps hazard clearance mean/min: 3.821 / 1.378
Planner route hazard clearance mean/min: 5.759 / 0.666
Static hazard clearance mean/min model_route: 1.197 / 0.633 speed_wps: 3.078 / 1.378 planner_route: 7.680 / 6.737
```

Important static-hazard samples:

```text
step=360 hazard=vehicle.mercedes.coupe_2020 v0.00 d5.55 model_route_hazard_min=1.736 speed_wps_hazard_min=1.378
step=380 hazard=vehicle.mercedes.coupe_2020 v0.00 d5.22 model_route_hazard_min=0.633 speed_wps_hazard_min=1.973
step=400 hazard=vehicle.mercedes.coupe_2020 v0.00 d5.22 model_route_hazard_min=0.642 speed_wps_hazard_min=1.840
```

Conclusion: this supports the user's data-quality hypothesis. After the ego
enters the Driving lane, the MTID 10e model does not produce a safe route around
the stationary Mercedes. The closest model-route clearance is only about
`0.63m`, which is far below a safe vehicle clearance. This means targeted
ParkingExit / parked-vehicle avoidance data is justified. A controller-side
route shift may still be useful as a smoke-test guardrail, but it should not be
treated as a replacement for missing training behavior.

## Known Limitations

- MTID v0 is data generation and compatibility only.
- No safety loss or architecture change is active yet.
- No new CARLA data has been collected.
- True motorcycle data is not present in the current dataset.
- Current two-wheeler labels come from bicycle actors.
- `lane_less_corridor` is a corridor-following proxy from route and road-user
  density, not true lane-less custom-scenario data.
- Local runs use `batch_size=1` and are early validation runs, not final paper
  results.

## Recommended Next Work

1. **Implement ParkingExit route/trajectory shift around the parked actor:**
   merge-steer gets ego into the Driving lane, but the local path still points
   into a stationary scenario vehicle.
2. **Keep yield behavior for moving traffic:** early hazards are valid
   non-stationary actors in the target lane, so the initial yield should remain.
3. **Re-run the short single-route ParkingExit reproduction:** use
   `mtid/routes/routes_mtid_parking_exit.xml` and require no vehicle collision,
   not just route completion.
4. **Run `routes_devtest.xml` after focused ParkingExit passes:** route 0 is
   currently the gating failure.
5. **Keep speed calibration as an efficiency ablation:** default planner
   steering is safer as the first devtest policy; `SPEED_SCALE=1.25` can be
   tested after the default devtest run.
6. **Analyze model-route drift:** compare predicted route steering against
   planner target-point steering around avoidance steps `95-120`.
7. **Decide final evaluation control policy:** planner steering can be a
   smoke-test guardrail, but paper claims should state clearly whether steering
   is model route, planner route, or a constrained/blended route.
8. Compare against regular mixed-data fine-tuning if such a same-architecture
   checkpoint becomes available.
9. Collect custom CARLA mixed-traffic data only if the paper needs stronger
   motorcycle, wrong-way, or true lane-less evidence after seeing closed-loop
   results.

## Run Summary Tool

Added after documentation cleanup:

```bash
python3 mtid/tools/summarize_training_run.py \
  simlingo_training/outputs/2026_04_27_10_33_14_simlingo_mtid_1k
```

Expected use:

- verify which metrics were last logged;
- confirm the last training step;
- confirm checkpoint files and sizes before selecting a checkpoint for
  qualitative inspection or evaluation.
