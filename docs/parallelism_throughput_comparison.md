# Parallelism and Throughput Comparison: RLinf+GenArk vs ActiveVLN

## Goal

This document summarizes the parallelism and throughput tradeoff between our current RLinf+GenArk RFT pipeline and ActiveVLN.

The main message for the figure:

> RLinf+GenArk currently has strong within-scene batch rollout throughput, while ActiveVLN has stronger scene-level diversity through a simulator pool.

## RLinf + GenArk

### Current Setup

RLinf+GenArk uses a Genesis-backed vectorized environment.

One `GenarkVecEnv` builds one Genesis scene. After the scene is built, the same worker cannot switch to another scene.

Inside that one scene, GenArk can run multiple env slots in batch:

```text
GenarkVecEnv / Genesis Scene A
├── env slot 0
├── env slot 1
├── env slot 2
├── ...
└── env slot 11
```

Current example:

```text
1 EnvWorker
1 pinned scene
12 env slots
20 LLM decisions per rollout
= 240 decision samples per rollout
```

Current properties:

```text
Scene diversity per rollout: 1 scene
Parallel env slots: 12
Strength: efficient batch rollout inside one scene
Weakness: low scene diversity if only one EnvWorker is used
```

### Scalable Setup

RLinf can increase scene diversity by launching multiple EnvWorkers.

Each EnvWorker pins to a different scene through `seed_offset`.

Example:

```text
EnvWorker 0 -> Genesis Scene A -> 12 env slots
EnvWorker 1 -> Genesis Scene B -> 12 env slots
EnvWorker 2 -> Genesis Scene C -> 12 env slots
EnvWorker 3 -> Genesis Scene D -> 12 env slots
```

The combined rollout batch becomes:

```text
4 scenes x 12 slots x 20 decisions
= 960 decision samples per rollout
```

Important limitation:

```text
One EnvWorker cannot mix multiple scenes.
Scene-level parallelism requires multiple EnvWorkers.
```

Summary:

```text
RLinf+GenArk parallelism = scene-internal batch slots x number of EnvWorkers
```

## ActiveVLN

ActiveVLN separates training from the VLN-CE environment server.

The training process talks to an external environment server through a batch client. The environment server owns a pool of simulator actors.

Example from ActiveVLN:

```text
2 GPUs
16 simulators per GPU
= 32 simulators in parallel
```

Conceptually:

```text
ActiveVLN Environment Server
├── Simulator 0  -> episode from scene A
├── Simulator 1  -> episode from scene B
├── Simulator 2  -> episode from scene C
├── ...
└── Simulator 31 -> episode from scene X
```

Each simulator can be assigned an episode from the dataset. Therefore, a rollout batch can cover many scenes.

Important nuance:

```text
32 simulators does not guarantee 32 unique scenes.
Actual scene count depends on which episodes are sampled.
```

Summary:

```text
ActiveVLN parallelism = simulator pool over dataset episodes
```

## Key Comparison

| Dimension | RLinf+GenArk Current | RLinf+GenArk Scaled | ActiveVLN |
|---|---:|---:|---:|
| Environment backend | Genesis | Genesis | Habitat-Sim / VLN-CE |
| Parallel unit | env slot inside one scene | EnvWorker x env slots | simulator actor |
| Scene switching inside one worker | No | No | Simulator can load assigned episode |
| Scene diversity per rollout | 1 scene | N EnvWorkers = N scenes | up to active simulator count |
| Parallel env count | 12 slots | e.g. 48 slots with 4 workers | 32 simulators in example |
| Main strength | batch rollout throughput inside one scene | high slot throughput plus moderate scene diversity | high scene diversity |
| Main bottleneck | scene diversity | GPU/memory cost per Genesis scene | simulator/server complexity |

## Core Message

RLinf+GenArk and ActiveVLN scale along different axes.

RLinf+GenArk is strong at:

```text
- batching many env slots inside one Genesis scene
- producing many decision samples from the same scene efficiently
- keeping env/rollout/actor inside one RLinf training pipeline
```

RLinf+GenArk needs:

```text
- multiple EnvWorkers to cover multiple scenes
- per-scene metrics to avoid hiding scene-level failures
- clean parse/stop/reward semantics so high throughput does not amplify bad samples
```

ActiveVLN is strong at:

```text
- many independent simulator actors
- broad scene/episode diversity in each rollout batch
- dynamic simulator allocation from a dataset-level episode pool
```

ActiveVLN costs:

```text
- external environment server
- simulator pool management
- HTTP/Ray communication
- more complex debugging across training and environment processes
```

## Figure Takeaway

RLinf+GenArk is currently throughput-oriented within a scene:

```text
1 EnvWorker x 1 scene x 12 slots
```

ActiveVLN is diversity-oriented across scenes:

```text
32 simulator actors x dataset episode pool
```

To close the scene diversity gap, RLinf+GenArk should scale to:

```text
N EnvWorkers x N scenes x 12 slots per scene
```

while keeping rollout samples clean and trainable.

## Practical Next Step

For our RLinf+GenArk pipeline, the first scalable target should be:

```text
2 EnvWorkers
2 pinned scenes
24 total env slots
12 slots per scene
20 decisions per rollout
= 480 decision samples per rollout
```

Then scale to:

```text
4 EnvWorkers
4 pinned scenes
48 total env slots
12 slots per scene
20 decisions per rollout
= 960 decision samples per rollout
```

This preserves GenArk's within-scene batch rollout advantage while gradually improving scene diversity.
