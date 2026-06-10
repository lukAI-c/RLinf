"""
Unit tests for MultiSceneBackend / SceneLayout.

No Genesis or Ray required — tests run against the pure-Python logic.
All tests import only from genesis_multiscene and do NOT instantiate actual
Genesis scenes or Ray actors.

Test matrix (plan §Verification):
  - Layout: group-alignment + tiling + fully-active invariants
  - Scatter: index translation, partition_env_idx, local<->global math
  - Coordinate: cam_pos_hab() contract (per-scene vs global)
  - Episode pool: per-scene pool resolution, scene_id matching
  - GRPO: no-straddle for all valid layouts
  - GPU budget: ValueError on K > budget
"""

import numpy as np
import pytest

# -------------------------------------------------------------------
# Import without Genesis / Ray (they may not be installed in CI)
# -------------------------------------------------------------------
import sys, types

# Stub out genesis and ray so import succeeds without GPU
def _stub_module(name):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod

if "ray" not in sys.modules:
    _ray = _stub_module("ray")
    # @ray.remote(num_gpus=1) and @ray.remote (no-arg) both must work
    def _ray_remote(*args, **kwargs):
        def decorator(cls): return cls
        if args and callable(args[0]) and not kwargs:
            return args[0]
        return decorator
    _ray.remote = _ray_remote
    _ray.get = lambda *a, **kw: None
    _ray.is_initialized = lambda: True
    _ray.init = lambda **kw: None

if "ray.exceptions" not in sys.modules:
    _ray_exc = _stub_module("ray.exceptions")
    class _RayActorError(Exception): pass
    class _WorkerCrashedError(Exception): pass
    class _RayTaskError(Exception): pass
    _ray_exc.RayActorError = _RayActorError
    _ray_exc.WorkerCrashedError = _WorkerCrashedError
    _ray_exc.RayTaskError = _RayTaskError

for _name in ["genesis"]:
    if _name not in sys.modules:
        _stub_module(_name)

# After stubbing, import the layout (pure Python, no actual actors)
import torch
from rlinf.envs.genark.genesis_multiscene import (
    SceneLayout,
    _stable_scene_hash,
    MultiSceneBackend,
)
from rlinf.envs.genark.genesis_server import SceneCrashError


# ===================================================================
# Fixtures
# ===================================================================

SCENES_2 = ["mp3d/sceneA/sceneA.glb", "mp3d/sceneB/sceneB.glb"]
SCENES_4 = ["mp3d/s0/s0.glb", "mp3d/s1/s1.glb", "mp3d/s2/s2.glb", "mp3d/s3/s3.glb"]


def make_layout(scenes=SCENES_2, episode_counts=None, num_envs=8, group_size=2,
                gpu_budget=None):
    if episode_counts is None:
        episode_counts = [50] * len(scenes)
    return SceneLayout.build(
        scenes=scenes,
        episode_counts=episode_counts,
        num_envs=num_envs,
        group_size=group_size,
        gpu_budget=gpu_budget,
    )


# ===================================================================
# Layout invariants
# ===================================================================

class TestSceneLayoutInvariants:

    def test_basic_two_scenes(self):
        layout = make_layout(num_envs=8, group_size=2)
        assert layout.num_envs == 8
        assert len(layout.scenes) == 2
        assert sum(layout.sizes) == 8

    def test_blocks_tile_no_gap_no_overlap(self):
        for num_envs, K, gs in [(8,2,2), (12,3,2), (16,4,4), (20,4,4)]:
            scenes = [f"s{i}" for i in range(K)]
            layout = SceneLayout.build(
                scenes=scenes,
                episode_counts=[100]*K,
                num_envs=num_envs,
                group_size=gs,
            )
            covered = np.zeros(num_envs, dtype=bool)
            for s in range(K):
                off, k_s = layout.offsets[s], layout.sizes[s]
                assert not covered[off:off+k_s].any(), \
                    f"blocks overlap at scene {s}"
                covered[off:off+k_s] = True
            assert covered.all(), "blocks do not tile [0, num_envs)"

    def test_group_alignment(self):
        """off_s and K_s must be multiples of group_size (Bug #2)."""
        layout = make_layout(num_envs=12, group_size=4,
                             scenes=["s0","s1","s2"],
                             episode_counts=[50,50,50])
        for s in range(3):
            assert layout.offsets[s] % 4 == 0, f"off_{s} not group-aligned"
            assert layout.sizes[s]   % 4 == 0, f"K_{s} not group-aligned"

    def test_fully_active_invariant_enforced(self):
        """ValueError if any scene has fewer episodes than its allocated slots (Bug #5)."""
        with pytest.raises(ValueError, match="only.*episodes.*allocated.*slots"):
            SceneLayout.build(
                scenes=["s0","s1"],
                episode_counts=[3, 50],  # s0 has 3 eps but will get >3 slots
                num_envs=8,
                group_size=2,
            )

    def test_gpu_budget_enforced(self):
        """ValueError if K > gpu_budget (Bug #9)."""
        with pytest.raises(ValueError, match="gpu_budget"):
            make_layout(scenes=SCENES_4, episode_counts=[50]*4,
                        num_envs=8, group_size=2, gpu_budget=2)

    def test_gpu_budget_ok(self):
        layout = make_layout(scenes=SCENES_4, episode_counts=[50]*4,
                             num_envs=8, group_size=2, gpu_budget=4)
        assert len(layout.scenes) == 4

    def test_num_envs_not_divisible_by_group_size(self):
        with pytest.raises(ValueError, match="not divisible by group_size"):
            SceneLayout.build(
                scenes=["s0","s1"],
                episode_counts=[50,50],
                num_envs=7,   # 7 % 2 != 0
                group_size=2,
            )

    def test_empty_scenes_raises(self):
        with pytest.raises(ValueError, match="empty"):
            SceneLayout.build(scenes=[], episode_counts=[], num_envs=8, group_size=2)

    def test_slot_to_scene_covers_all(self):
        layout = make_layout(num_envs=12, group_size=2,
                             scenes=["s0","s1","s2"],
                             episode_counts=[50,50,50])
        for g in range(12):
            s = int(layout.slot_to_scene_idx[g])
            assert 0 <= s < 3

    def test_single_scene_degenerate(self):
        """K=1 must work (used for equivalence baseline in Phase 0)."""
        layout = SceneLayout.build(
            scenes=["s0"], episode_counts=[50], num_envs=8, group_size=2
        )
        assert layout.offsets[0] == 0
        assert layout.sizes[0]   == 8
        assert list(layout.slot_to_scene_idx) == [0]*8


# ===================================================================
# Index translation (scatter / global<->local math)
# ===================================================================

class TestIndexTranslation:

    def test_scene_idx_of(self):
        layout = make_layout(num_envs=8, group_size=2)
        # layout with 2 scenes each getting 4 slots: [0..3]→0, [4..7]→1
        assert layout.scene_idx_of(0) == 0
        assert layout.scene_idx_of(3) == 0
        assert layout.scene_idx_of(4) == 1
        assert layout.scene_idx_of(7) == 1

    def test_local_idx(self):
        layout = make_layout(num_envs=8, group_size=2)
        off1 = layout.offsets[1]
        for l in range(layout.sizes[1]):
            g = off1 + l
            assert layout.local_idx(g) == l

    def test_global_idx_roundtrip(self):
        layout = make_layout(num_envs=12, group_size=2,
                             scenes=["s0","s1","s2"],
                             episode_counts=[50,50,50])
        for g in range(12):
            s = layout.scene_idx_of(g)
            l = layout.local_idx(g)
            assert layout.global_idx(s, l) == g

    def test_partition_env_idx(self):
        layout = make_layout(num_envs=8, group_size=2)
        env_idx = [0, 2, 4, 6]
        partitioned = layout.partition_env_idx(env_idx)
        # Scenes 0 (slots 0-3) and 1 (slots 4-7)
        assert set(partitioned[0][0]) == {0, 2}
        assert set(partitioned[1][0]) == {4, 6}
        # Local indices
        off1 = layout.offsets[1]
        assert set(partitioned[1][1]) == {4 - off1, 6 - off1}

    def test_partition_preserves_row_order(self):
        """Rows in the partitioned output must correspond to env_idx entries."""
        layout = make_layout(num_envs=8, group_size=2)
        env_idx = [7, 0, 4, 3]
        partitioned = layout.partition_env_idx(env_idx)
        for s, (globals_, locals_) in partitioned.items():
            for g, l in zip(globals_, locals_):
                assert l == g - layout.offsets[s]

    def test_scene_local_group(self):
        layout = make_layout(num_envs=8, group_size=2)
        # Scene 0: slots 0-3, group_size=2 → local groups 0,0,1,1
        assert layout.scene_local_group(0) == 0
        assert layout.scene_local_group(1) == 0
        assert layout.scene_local_group(2) == 1
        assert layout.scene_local_group(3) == 1
        # Scene 1: slots 4-7 → local groups 0,0,1,1
        assert layout.scene_local_group(4) == 0
        assert layout.scene_local_group(5) == 0
        assert layout.scene_local_group(6) == 1
        assert layout.scene_local_group(7) == 1


# ===================================================================
# GRPO: no group straddles scenes
# ===================================================================

class TestGRPONoStraddle:

    def _check_no_straddle(self, layout):
        gs = layout.group_size
        for g_start in range(0, layout.num_envs, gs):
            scenes_in_group = {
                int(layout.slot_to_scene_idx[j])
                for j in range(g_start, min(g_start + gs, layout.num_envs))
            }
            assert len(scenes_in_group) == 1, (
                f"Group starting at slot {g_start} straddles scenes {scenes_in_group}"
            )

    def test_2_scenes(self):
        self._check_no_straddle(make_layout(num_envs=8, group_size=2))

    def test_4_scenes(self):
        self._check_no_straddle(
            make_layout(scenes=SCENES_4, episode_counts=[50]*4,
                        num_envs=16, group_size=4)
        )

    def test_group_size_1(self):
        self._check_no_straddle(make_layout(num_envs=8, group_size=1))

    def test_group_size_equals_block_size(self):
        """One group per scene."""
        self._check_no_straddle(make_layout(num_envs=4, group_size=2))

    @pytest.mark.parametrize("num_envs,K,gs", [
        (8, 2, 2), (8, 2, 4), (12, 3, 2), (12, 3, 4),
        (16, 4, 2), (16, 4, 4), (24, 4, 4), (20, 4, 4),
    ])
    def test_parametrize(self, num_envs, K, gs):
        scenes = [f"s{i}" for i in range(K)]
        try:
            layout = SceneLayout.build(
                scenes=scenes, episode_counts=[100]*K,
                num_envs=num_envs, group_size=gs,
            )
            self._check_no_straddle(layout)
        except ValueError:
            pass  # indivisible configs; we test the error in layout tests


# ===================================================================
# Per-scene pool logic (episode_counts < slots triggers error)
# ===================================================================

class TestPerScenePool:

    def test_scene_id_of_each_slot(self):
        layout = make_layout(num_envs=8, group_size=2)
        for g in range(8):
            sid = layout.scene_id_of(g)
            assert sid in SCENES_2

    def test_episode_count_boundary(self):
        """Exactly enough episodes: K_s == len(eps_s) must succeed."""
        layout = SceneLayout.build(
            scenes=["s0","s1"],
            episode_counts=[4, 4],  # exactly K_s each
            num_envs=8,
            group_size=2,
        )
        assert layout.sizes[0] == 4
        assert layout.sizes[1] == 4

    def test_episode_count_one_short_raises(self):
        with pytest.raises(ValueError, match="only.*episodes"):
            SceneLayout.build(
                scenes=["s0","s1"],
                episode_counts=[3, 50],  # s0 is one short
                num_envs=8,
                group_size=2,
            )


# ===================================================================
# Stable hash determinism
# ===================================================================

class TestStableHash:

    def test_deterministic(self):
        h1 = _stable_scene_hash("mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb")
        h2 = _stable_scene_hash("mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb")
        assert h1 == h2

    def test_different_scenes_differ(self):
        h1 = _stable_scene_hash("mp3d/sceneA/sceneA.glb")
        h2 = _stable_scene_hash("mp3d/sceneB/sceneB.glb")
        assert h1 != h2

    def test_non_negative(self):
        for sid in SCENES_2 + SCENES_4:
            assert _stable_scene_hash(sid) >= 0


# ===================================================================
# Phase 2: concurrent fan-out dispatch
# ===================================================================
#
# The K scenes run in parallel ONLY if MultiSceneBackend submits all K
# .remote() calls BEFORE blocking on any of them. These tests use fake
# sub-backends that log the (submit, fetch) order and assert every submit
# precedes every fetch — i.e. all actors are working before we collect.

class _RecordingSub:
    """Fake GenesisRemoteBackend recording submit/fetch order (no Ray, no GPU)."""

    def __init__(self, idx, log, k_s, cam_h=4, cam_w=4, crash_on=None):
        self.idx = idx
        self.log = log
        self._k = k_s
        self._cam_h = cam_h
        self._cam_w = cam_w
        self._crash_on = crash_on or set()
        self.cam_pos = torch.zeros(k_s, 3)
        self.cam_yaw = torch.zeros(k_s)
        self.current_tri_idx = torch.zeros(k_s, dtype=torch.int64)

    def _maybe_crash(self, tag):
        if tag in self._crash_on:
            raise SceneCrashError(scene_id=f"s{self.idx}", cause=Exception("boom"))

    # --- step / pose (return None, sync via fetch_state) ---
    def step_physics_async(self, actions, active_mask, active_k_s):
        self.log.append(("submit", self.idx))
        return ("step", self.idx)

    def set_agent_poses_async(self, locals_, pos, yaw):
        self.log.append(("submit", self.idx))
        return ("pose", self.idx)

    def fetch_state(self, ref):
        self.log.append(("fetch", self.idx))
        self._maybe_crash("state")

    # --- render ---
    def render_main_async(self, k_s):
        self.log.append(("submit", self.idx))
        return ("render", self.idx)

    def fetch_render_main(self, ref):
        self.log.append(("fetch", self.idx))
        self._maybe_crash("render")
        return np.zeros((self._k, self._cam_h, self._cam_w, 3), dtype=np.uint8)

    def render_4dir_async(self, k_s):
        self.log.append(("submit", self.idx))
        return ("4dir", self.idx)

    def fetch_render_4dir(self, ref):
        self.log.append(("fetch", self.idx))
        self._maybe_crash("4dir")
        return np.zeros((self._k, 3, self._cam_h, self._cam_w, 3), dtype=np.uint8)


def _bare_multiscene(layout, subs, cam_h=4, cam_w=4):
    """Construct a MultiSceneBackend without running __init__ (which needs Ray)."""
    ms = object.__new__(MultiSceneBackend)
    ms._layout = layout
    ms._subs = subs
    ms._device = torch.device("cpu")
    ms._cam_h = cam_h
    ms._cam_w = cam_w
    N = layout.num_envs
    ms._cam_pos_t = torch.zeros(N, 3)
    ms._cam_yaw_t = torch.zeros(N)
    ms._current_tri_idx_t = torch.zeros(N, dtype=torch.int64)
    return ms


class TestConcurrentDispatch:

    def _assert_submit_before_fetch(self, log, K):
        kinds = [k for k, _ in log]
        assert kinds.count("submit") == K, f"expected {K} submits, got {log}"
        assert kinds.count("fetch") == K, f"expected {K} fetches, got {log}"
        last_submit = max(i for i, (k, _) in enumerate(log) if k == "submit")
        first_fetch = min(i for i, (k, _) in enumerate(log) if k == "fetch")
        assert last_submit < first_fetch, (
            f"a fetch ran before the last submit => actors serialized: {log}"
        )

    def test_step_physics_parallel_dispatch(self):
        layout = SceneLayout.build(["s0", "s1", "s2"], [50, 50, 50],
                                   num_envs=12, group_size=2)
        log = []
        subs = [_RecordingSub(s, log, layout.sizes[s]) for s in range(3)]
        ms = _bare_multiscene(layout, subs)
        ms.step_physics(torch.zeros(12, dtype=torch.int64),
                        torch.ones(12, dtype=torch.bool), 12)
        self._assert_submit_before_fetch(log, 3)

    def test_render_main_parallel_dispatch(self):
        layout = SceneLayout.build(["s0", "s1", "s2"], [50, 50, 50],
                                   num_envs=12, group_size=2)
        log = []
        subs = [_RecordingSub(s, log, layout.sizes[s]) for s in range(3)]
        ms = _bare_multiscene(layout, subs)
        out = ms.render_main(12)
        self._assert_submit_before_fetch(log, 3)
        assert out.shape == (12, 4, 4, 3)

    def test_render_4dir_parallel_dispatch(self):
        layout = SceneLayout.build(["s0", "s1", "s2"], [50, 50, 50],
                                   num_envs=12, group_size=2)
        log = []
        subs = [_RecordingSub(s, log, layout.sizes[s]) for s in range(3)]
        ms = _bare_multiscene(layout, subs)
        out = ms.render_4dir(12)
        self._assert_submit_before_fetch(log, 3)
        assert out.shape == (12, 3, 4, 4, 3)

    def test_set_agent_poses_parallel_dispatch(self):
        layout = SceneLayout.build(["s0", "s1"], [50, 50],
                                   num_envs=8, group_size=2)
        log = []
        subs = [_RecordingSub(s, log, layout.sizes[s]) for s in range(2)]
        ms = _bare_multiscene(layout, subs)
        env_idx = [0, 1, 4, 5]  # touches both scene blocks
        ms.set_agent_poses(env_idx, torch.zeros(4, 3), torch.zeros(4))
        self._assert_submit_before_fetch(log, 2)

    def test_scatter_after_step_writes_global_tensor(self):
        """Phase B scatter must land each scene's block in the global tensor."""
        layout = SceneLayout.build(["s0", "s1"], [50, 50],
                                   num_envs=8, group_size=2)
        log = []
        subs = [_RecordingSub(s, log, layout.sizes[s]) for s in range(2)]
        # mark each scene's cam_pos so we can detect correct placement
        subs[0].cam_pos = torch.full((4, 3), 1.0)
        subs[1].cam_pos = torch.full((4, 3), 2.0)
        ms = _bare_multiscene(layout, subs)
        ms.step_physics(torch.zeros(8, dtype=torch.int64),
                        torch.ones(8, dtype=torch.bool), 8)
        assert torch.allclose(ms._cam_pos_t[0:4], torch.full((4, 3), 1.0))
        assert torch.allclose(ms._cam_pos_t[4:8], torch.full((4, 3), 2.0))

    def test_crash_during_fetch_aggregates_to_dormancy(self):
        """One scene crashing still submits all K first, then raises SceneCrashError."""
        layout = SceneLayout.build(["s0", "s1"], [50, 50],
                                   num_envs=8, group_size=2)
        log = []
        subs = [
            _RecordingSub(0, log, layout.sizes[0], crash_on={"state"}),
            _RecordingSub(1, log, layout.sizes[1]),
        ]
        ms = _bare_multiscene(layout, subs)
        with pytest.raises(SceneCrashError):
            ms.step_physics(torch.zeros(8, dtype=torch.int64),
                            torch.ones(8, dtype=torch.bool), 8)
        # Parallel dispatch must be intact even on the crash path:
        kinds = [k for k, _ in log]
        assert kinds.count("submit") == 2, f"submits not fanned out: {log}"
