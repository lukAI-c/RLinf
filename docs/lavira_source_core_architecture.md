# LaViRA Source-Core Architecture

## Goal

RLinf keeps batched policy inference, simulator ownership, rollout lifecycle,
metrics, and training channels. Navigation geometry is delegated to the pinned
LHX implementation from `/home/lhx/workspace/lavira-code`, branch
`feat/merged-la-va`, commit `b3d6c35067ad731a16ec096b698ddfc2f5c91af6`.

## Two Tracks

The oracle track runs the untouched LHX `ZS_Evaluator_mp.py`. It remains the
reference for native Habitat behavior and the reported baseline.

The RLinf integration track vendors a byte-identical navigation subset under
`rlinf/third_party/lavira_rft/source/`. It directly runs LHX
`Semantic_Mapping`, `FusionMapPolicy`, `FMMPlanner`, map/depth utilities, and
RepViT-SAM construction. Files under `source/` must not be edited.

`loader.py` owns Python 3.11/Torch compatibility at the import boundary.
`source_core.py` is a thin adapter for metric RGB-D, absolute world poses,
per-slot reset, and diagnostics. RLinf continues to own model batching,
on-policy token/logprob storage, Habitat RPC, Genesis backends, reward,
advantage, actor update, and metric aggregation.

## Runtime Contract

Production configuration:

```yaml
lavira_runtime:
  enabled: true
  map_backend: source
  map_device: cuda
```

`map_backend: sparse_ab` retains the former NumPy mapper only for same-frame
migration diagnostics. It is not a supported training or evaluation backend.
Unknown backend names fail at controller construction.

Each real observation runs source map fusion once, then rebuilds traversibility
once. FMM replay reuses that cache until the next observation. Production FMM
does not call `rlinf/models/embodiment/qwen_nav/lavira_map.py`.

Sensor geometry is supplied by the active simulator contract. Strict Habitat
evaluation uses `HFOV=79°` and camera height `0.88m`; Genesis can supply its
calibrated values without changing the LHX algorithm body.

## Same-Frame Audit

Set `map_visualization.save_raw_every: 1`. Raw snapshots contain exact metric
depth, semantic masks, absolute pose, and previous primitive action. Replay an
episode through source and sparse A/B maps with:

```bash
PYTHONPATH=/home/clk/workspace/RLinf \
/home/clk/miniconda3/envs/genesis-vllm/bin/python \
scripts/compare_lavira_mapper_frames.py \
  --input /path/to/lavira_maps/worker_x/env_00/episode_001 \
  --output /tmp/lavira_mapper_ab.json
```

The report contains per-frame obstacle, explored, and traversible XOR, plus a
source replay versus saved-source consistency check.

## Verification

```bash
/home/clk/miniconda3/envs/genesis-vllm/bin/python -m pytest -q \
  tests/unit_tests/test_lavira_runtime.py \
  tests/unit_tests/test_lavira_source_core.py \
  tests/unit_tests/test_habitat_qwen_contract.py \
  tests/unit_tests/test_qwen_nav_lavira_waypoint_parser.py
```

After unit verification, run episode 259 with `map_backend=source`, inspect the
recorded `bbox -> depth -> map goal -> traversible -> FMM action` chain, and
only then start the 100-episode Habitat evaluation.
