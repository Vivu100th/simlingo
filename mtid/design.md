# MTID Design Notes

## Core Idea

SimLingo Action Dreaming creates alternative ego futures for the same visual
context. MTID adds alternative futures for the surrounding actors so the model
learns interaction-aware behavior in unstructured mixed traffic.

Baseline comparison:

- Action Dreaming: ego counterfactual, actors replayed/non-reactive.
- MTID: ego counterfactual plus actor counterfactual, scored by physical safety
  and interaction risk.

## First MVP

Keep the SimLingo model architecture unchanged. The first MTID version should
only generate new dreamer-style labels compatible with `Data_Dreamer`.

MVP modes:

1. `motorcycle_cut_in`
2. `jaywalker_crossing`
3. `two_wheeler_filtering`
4. `wrong_way_two_wheeler`
5. `lane_less_corridor`
6. `dense_gap_yield`

## Physics Blocks

Ego:

```text
x = [px, py, yaw, v]
u = [steer, throttle, brake]
x_next = bicycle_model(x, u)
```

Vehicles:

```text
IDM acceleration:
a = a_max * (1 - (v / v0)^delta - (s_star / s)^2)
s_star = s0 + v * T + v * delta_v / (2 * sqrt(a_max * b))
```

Pedestrians / two-wheelers:

```text
desired_velocity + repulsive_collision_avoidance + boundary/corridor force
```

Risk:

```text
TTC = distance_along_conflict / closing_speed
risk_i = exp(-distance / sigma_d) * exp(-TTC / sigma_t) * class_weight
```

Barrier-style penalty:

```text
L_barrier = sum_t sum_i ReLU(d_safe_i^2 - d_i(t)^2)
```

## Data Contract

The generated sample should stay close to the original SimLingo dreamer JSON
format:

```text
mode
waypoints
route
rgb_path
allowed
info
route_reasoning
dreamer_instruction
instructions_templates
templates_placeholders
dreamer_answer_safety
safe_to_execute
```

Extra MTID fields can go inside `info` first to avoid breaking the dataloader.

## SimLingo Integration

MTID v0 adds one training switch:

```yaml
data_module:
  base_dataset:
    dreamer_folder: mtid_dreamer
```

If the field is absent, the dataloader still uses `dreamer`. The original
`dataset_generation/dreamer_data` files and the model architecture are not
changed in v0.
