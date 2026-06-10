"""
Benchmark: concurrent vs sequential scene dispatch in MultiSceneBackend.

Uses fake Ray actors with configurable sleep to simulate GPU render time,
so no actual Genesis scenes are needed. Proves that Phase 2 fan-out
dispatches all K actors in parallel (wall-clock ≈ max, not sum).

Usage:
    CUDA_VISIBLE_DEVICES=4,5,6,7 python tests/benchmark_concurrent_dispatch.py

Expected output (render_delay=0.3s, K=4):
    Sequential K=4: ~1200 ms  (4 × 300ms)
    Concurrent  K=4: ~320 ms  (300ms + overhead)
    Speedup: ~3.7×
"""

import sys
import time
import types

# ---------------------------------------------------------------------------
# Bootstrap: must happen before any rlinf import
# ---------------------------------------------------------------------------

import ray

# Stub genesis (not needed for this benchmark)
if "genesis" not in sys.modules:
    sys.modules["genesis"] = types.ModuleType("genesis")

sys.path.insert(0, "/home/nvme03/lck/RLinf")

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Fake actor — simulates render latency on a real GPU process
# ---------------------------------------------------------------------------

RENDER_DELAY = 0.30   # seconds per scene per step (tune to match your real render time)
CAM_H, CAM_W = 480, 640
N_ENVS_PER_SCENE = 4

@ray.remote(num_gpus=1)
class FakeSceneActor:
    """Occupies 1 GPU, sleeps RENDER_DELAY to simulate rendering."""

    def __init__(self, scene_idx: int):
        import torch
        self._idx = scene_idx
        # Touch CUDA so Ray actually acquires the GPU
        self._dummy = torch.zeros(1, device="cuda")

    def render(self, k_s: int) -> bytes:
        import time
        time.sleep(RENDER_DELAY)
        return bytes(k_s * CAM_H * CAM_W * 3)

    def ping(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Sequential fan-out (Phase 1 equivalent)
# ---------------------------------------------------------------------------

def sequential_render(actors, k_s: int) -> float:
    t0 = time.perf_counter()
    for actor in actors:
        ray.get(actor.render.remote(k_s))
    return (time.perf_counter() - t0) * 1000  # ms


# ---------------------------------------------------------------------------
# Concurrent fan-out (Phase 2: submit-all then collect-all)
# ---------------------------------------------------------------------------

def concurrent_render(actors, k_s: int) -> float:
    t0 = time.perf_counter()
    refs = [actor.render.remote(k_s) for actor in actors]   # all submitted (non-blocking)
    ray.get(refs)                                             # collect all
    return (time.perf_counter() - t0) * 1000  # ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(K: int, repeats: int = 8, warmup: int = 3):
    print(f"\n{'='*60}")
    print(f"  K={K} scenes  |  render_delay={RENDER_DELAY*1000:.0f}ms  |  "
          f"{N_ENVS_PER_SCENE} envs/scene")
    print(f"{'='*60}")

    actors = [FakeSceneActor.remote(s) for s in range(K)]
    # Warmup: wait for all actors to be ready
    ray.get([a.ping.remote() for a in actors])
    print(f"  {K} actors ready (each on its own GPU).")

    # --- sequential ---
    seq_times = []
    for i in range(warmup + repeats):
        t = sequential_render(actors, N_ENVS_PER_SCENE)
        if i >= warmup:
            seq_times.append(t)
    seq_avg = sum(seq_times) / len(seq_times)

    # --- concurrent ---
    con_times = []
    for i in range(warmup + repeats):
        t = concurrent_render(actors, N_ENVS_PER_SCENE)
        if i >= warmup:
            con_times.append(t)
    con_avg = sum(con_times) / len(con_times)

    speedup = seq_avg / con_avg

    print(f"\n  Sequential K={K}:  {seq_avg:7.1f} ms/step  "
          f"(expected ~{RENDER_DELAY*K*1000:.0f} ms)")
    print(f"  Concurrent K={K}:  {con_avg:7.1f} ms/step  "
          f"(expected ~{RENDER_DELAY*1000:.0f} ms + overhead)")
    print(f"  Speedup:           {speedup:.2f}×  "
          f"(ideal {K:.1f}×)")

    [ray.kill(a) for a in actors]
    return seq_avg, con_avg, speedup


if __name__ == "__main__":
    ray.init(
        num_gpus=4,           # expose GPUs 4-7 (set via CUDA_VISIBLE_DEVICES externally)
        ignore_reinit_error=True,
    )
    print(f"\nRay cluster: {ray.cluster_resources()}")

    results = []
    for K in [1, 2, 3, 4]:
        s, c, sp = run(K, repeats=8, warmup=3)
        results.append((K, s, c, sp))

    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    print(f"  {'K':>3}  {'Sequential':>14}  {'Concurrent':>12}  {'Speedup':>10}")
    print(f"  {'-'*44}")
    for K, s, c, sp in results:
        print(f"  {K:>3}  {s:>12.1f}ms  {c:>10.1f}ms  {sp:>9.2f}×")

    ray.shutdown()
    print()
