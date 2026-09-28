"""Contract tests for the isolated Habitat Qwen evaluation bridge."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from rlinf.data.embodied_io_struct import EnvOutput
from rlinf.envs.habitat.qwen_remote_env import (
    HabitatQwenRemoteEnv,
    _AtomicSceneQueue,
)
from rlinf.models.embodiment.qwen_nav.grounded_sam import (
    GroundedSAMRefineResult,
    GroundedSAMServicePool,
    GroundedSAMWaypointRefiner,
)
from rlinf.models.embodiment.qwen_nav.prompts import (
    build_source_stop_check_parts,
    build_stop_check_text,
)
from scripts.aggregate_aligned_nav_metrics import aggregate_rows
from scripts.grounded_sam_server import dispatch_request, serve_connection
from scripts.habitat_qwen_server import HabitatQwenBridge


def _one_slot_observation(instruction="go forward", active=True):
    return {
        "main_images": np.zeros((1, 2, 3, 3), dtype=np.uint8),
        "extra_view_images": np.zeros((1, 3, 2, 3, 3), dtype=np.uint8),
        "wrist_images": np.ones((1, 4, 2, 3, 1), dtype=np.float32),
        "states": np.zeros((1, 4), dtype=np.float32),
        "scan_images": np.zeros((1, 12, 2, 3, 3), dtype=np.uint8),
        "scan_depth_images": np.zeros((1, 12, 2, 3, 1), dtype=np.float32),
        "scan_states": np.zeros((1, 12, 3), dtype=np.float32),
        "scan_valid": np.zeros((1,), dtype=bool),
        "episode_active": np.asarray([active], dtype=bool),
        "task_descriptions": [instruction],
    }


def test_pool_merge_keeps_flat_instructions_and_active_mask():
    merged = HabitatQwenBridge._merge_observations(
        [_one_slot_observation("first"), _one_slot_observation("", False)]
    )
    assert merged["task_descriptions"] == ["first", ""]
    assert merged["episode_active"].tolist() == [True, False]
    assert merged["main_images"].shape[0] == 2


def test_remote_adapter_rejects_nested_instruction_contract():
    observation = _one_slot_observation()
    observation["task_descriptions"] = [["nested"]]
    with pytest.raises(TypeError, match="flat list"):
        HabitatQwenRemoteEnv._to_tensor_obs(observation)


def test_scan_and_activity_contract_reaches_policy_facing_env_output():
    remote_obs = HabitatQwenRemoteEnv._to_tensor_obs(_one_slot_observation())
    policy_obs = EnvOutput(obs=remote_obs).to_dict()["obs"]
    expected = {
        "main_images", "extra_view_images", "wrist_images", "states",
        "task_descriptions", "scan_images", "scan_depth_images",
        "scan_states", "scan_valid", "episode_active", "episode_ids", "trial_ids",
        "scene_ids", "simulator_positions",
    }
    assert set(policy_obs) == expected
    assert policy_obs["scan_images"].shape == (1, 12, 2, 3, 3)
    assert policy_obs["scan_valid"].dtype == torch.bool
    assert policy_obs["episode_active"].tolist() == [True]


def test_atomic_scene_queue_claims_each_scene_once(tmp_path):
    manifest = tmp_path / "queue.json"
    state = tmp_path / "state.json"
    manifest.write_text(json.dumps({
        "schema_version": 2,
        "slots_per_worker": 2,
        "scene_jobs": [
            {"scene_id": "scene_a", "episode_ids": ["1", "2", "3"]},
            {"scene_id": "scene_b", "episode_ids": ["4"]},
        ],
    }))
    state.write_text(json.dumps({"next_scene": 0, "claims": []}))
    queue_a = _AtomicSceneQueue(manifest, state)
    queue_b = _AtomicSceneQueue(manifest, state)

    assert queue_a.claim(0)[1]["scene_id"] == "scene_a"
    assert queue_b.claim(1)[1]["scene_id"] == "scene_b"
    assert queue_a.claim(0) is None
    claims = json.loads(state.read_text())["claims"]
    assert [row["chunk_index"] for row in claims] == [0, 1]


def test_dynamic_scene_worker_finishes_scene_before_claiming_next():
    class _Queue:
        episode_ids = ["1", "2", "3", "4"]

        def __init__(self):
            self.jobs = iter([
                (0, {"scene_id": "a", "episode_ids": ["1", "2", "3"]}),
                (1, {"scene_id": "b", "episode_ids": ["4"]}),
            ])

        def claim(self, _rank):
            return next(self.jobs, None)

    env = HabitatQwenRemoteEnv.__new__(HabitatQwenRemoteEnv)
    env.num_envs = 2
    env._worker_rank = 0
    env._scene_queue = _Queue()
    env._queue_exhausted = False
    env._current_chunk_index = None
    env._current_scene_episodes = []
    env._current_scene_offset = 0
    env._current_scene_batch = 0
    env._all_assignments = []
    env._elapsed_steps = np.zeros(2, dtype=np.int32)
    env._done = np.zeros(2, dtype=bool)
    env._is_start = True
    resets = []

    def _rpc(method, payload):
        assert method == "reset"
        resets.append(list(payload["episode_ids"]))
        return {"obs": HabitatQwenBridge._merge_observations([
            _one_slot_observation("active" if value else "", bool(value))
            for value in payload["episode_ids"]
        ])}

    env._rpc = _rpc
    env._reset_next_scene_chunk()
    env._reset_next_scene_chunk()
    env._reset_next_scene_chunk()
    env._reset_next_scene_chunk()

    assert resets == [["1", "2"], ["3", ""], ["4", ""], ["", ""]]
    assert [episode for episode, _ in env._all_assignments] == ["1", "2", "3", "4"]
    assert env._queue_exhausted is True


def test_habitat_bridge_inactive_padding_observation_is_dormant():
    bridge = HabitatQwenBridge.__new__(HabitatQwenBridge)
    bridge._args = SimpleNamespace(height=2, width=3)
    bridge._envs = [None]
    bridge._done = np.asarray([True])
    bridge._terminal_observation_emitted = np.asarray([True])
    bridge._elapsed = np.asarray([0], dtype=np.int32)
    bridge._scan_observations = [None]

    obs = bridge._observation(0)
    assert not bool(obs["episode_active"])
    assert obs["task_descriptions"] == ""
    assert not obs["main_images"].any()


def test_reset_trial_identity_never_reuses_ids():
    env = HabitatQwenRemoteEnv.__new__(HabitatQwenRemoteEnv)
    env.num_envs = 2
    env._episode_ids = ["259", "259"]
    env._reset_count = 0
    env._episode_records = {}
    env._elapsed_steps = np.zeros(2, dtype=np.int32)
    env._done = np.zeros(2, dtype=bool)
    env._is_start = True
    calls = []

    def _rpc(method, payload):
        calls.append((method, payload))
        return {"obs": HabitatQwenBridge._merge_observations([
            _one_slot_observation(), _one_slot_observation()
        ])}

    env._rpc = _rpc
    env.reset()
    first = list(env._trial_ids)
    env.reset()
    second = list(env._trial_ids)
    assert set(first).isdisjoint(second)
    assert calls[0][1]["trial_ids"] == first
    assert calls[1][1]["trial_ids"] == second


def test_repeated_episode_metrics_are_ordered_by_trial_identity():
    env = HabitatQwenRemoteEnv.__new__(HabitatQwenRemoteEnv)
    env._episode_ids = ["259", "259"]
    env._trial_ids = ["trial_0", "trial_1"]
    env._episode_records = {}
    rows = [
        {"episode_id": "259", "trial_id": "trial_1", "success": 0.0},
        {"episode_id": "259", "trial_id": "trial_0", "success": 1.0},
    ]
    env._rpc = lambda *_args, **_kwargs: {"records": rows}

    ordered = env.get_episode_metrics()
    assert [row["trial_id"] for row in ordered] == ["trial_0", "trial_1"]
    assert [row["success"] for row in ordered] == [1.0, 0.0]


def test_shared_rollout_advances_shard_and_preserves_all_metrics():
    env = HabitatQwenRemoteEnv.__new__(HabitatQwenRemoteEnv)
    env.num_envs = 1
    env._worker_rank = 2
    env._shared_rollout = True
    env._episode_pool = ["10", "11"]
    env._episode_cursor = 0
    env._episode_ids = []
    env._trial_ids = []
    env._all_assignments = []
    env._episode_records = {}
    env._reset_count = 0
    env._elapsed_steps = np.zeros(1, dtype=np.int32)
    env._done = np.zeros(1, dtype=bool)
    env._is_start = True
    calls = []

    def _rpc(method, payload):
        calls.append((method, payload))
        if method == "reset":
            return {"obs": _one_slot_observation()}
        return {
            "records": [
                {
                    "episode_id": episode_id,
                    "trial_id": trial_id,
                    "success": float(episode_id == "11"),
                }
                for episode_id, trial_id in env._all_assignments
            ]
        }

    env._rpc = _rpc
    env.reset()
    env.reset()

    assert [call[1]["episode_ids"] for call in calls[:2]] == [["10"], ["11"]]
    assert env.expected_episode_metrics_count == 2
    records = env.get_episode_metrics()
    assert [row["episode_id"] for row in records] == ["10", "11"]
    assert len({row["trial_id"] for row in records}) == 2


def test_shared_rollout_rejects_reset_after_shard_exhaustion():
    env = HabitatQwenRemoteEnv.__new__(HabitatQwenRemoteEnv)
    env.num_envs = 1
    env._worker_rank = 0
    env._shared_rollout = True
    env._episode_pool = ["10"]
    env._episode_cursor = 1

    with pytest.raises(RuntimeError, match="episode pool exhausted"):
        env.reset()


def test_metric_files_separate_trials_from_episode_average(tmp_path):
    bridge = HabitatQwenBridge.__new__(HabitatQwenBridge)
    bridge._metrics_path = str(tmp_path)
    bridge._records = [
        {
            "scene_id": "scene", "episode_id": "259", "trial_id": "trial_0",
            "success": 1.0, "distance_to_goal": 1.0,
        },
        {
            "scene_id": "scene", "episode_id": "259", "trial_id": "trial_1",
            "success": 0.0, "distance_to_goal": 5.0,
        },
    ]

    bridge._write_metrics()
    trials = json.loads((tmp_path / "all_episode_metrics.json").read_text())
    summary = json.loads((tmp_path / "per_episode_summary.json").read_text())
    average = json.loads((tmp_path / "avg_metrics.json").read_text())
    assert len(trials) == 2
    assert summary == [{
        "scene_id": "scene", "episode_id": "259", "num_trials": 2,
        "success": 0.5, "distance_to_goal": 3.0,
    }]
    assert average["success"] == 0.5


def test_full_aggregate_keeps_trial_rows_and_trial_weighted_average(tmp_path):
    rows = [
        {"scene_id": "s", "episode_id": "1", "trial_id": "a", "success": 1.0},
        {"scene_id": "s", "episode_id": "1", "trial_id": "b", "success": 1.0},
        {"scene_id": "s", "episode_id": "2", "trial_id": "c", "success": 0.0},
    ]
    aggregate_rows(rows, tmp_path, expected_episodes=2)
    trials = json.loads((tmp_path / "all_episode_metrics.json").read_text())
    episodes = json.loads((tmp_path / "per_episode_summary.json").read_text())
    average = json.loads((tmp_path / "avg_metrics.json").read_text())
    assert len(trials) == 3
    assert len(episodes) == 2
    assert average["success"] == pytest.approx(2.0 / 3.0)
    assert not (tmp_path / "all_episode_trials.json").exists()


def test_grounded_sam_matches_lhx_color_and_instance_fusion_contract():
    seen = {}

    class _Dino:
        def predict_with_classes(self, image, classes, **_kwargs):
            seen["dino_image"] = image.copy()
            return SimpleNamespace(
                xyxy=np.asarray([[0, 0, 2, 2], [1, 1, 3, 3]], dtype=np.float32),
                class_id=np.asarray([0, 0]),
            )

    refiner = GroundedSAMWaypointRefiner.__new__(GroundedSAMWaypointRefiner)
    refiner.grounding_dino_model = _Dino()
    refiner.box_threshold = 0.25
    refiner.text_threshold = 0.25
    refiner.max_box_area_ratio = 0.95
    refiner._segment = lambda image, _boxes: (
        seen.setdefault("sam_image", image.copy()),
        np.asarray([
            [[1, 1, 0], [1, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 1, 1], [0, 1, 1]],
        ], dtype=np.float32),
    )[1]
    rgb = np.zeros((3, 3, 3), dtype=np.uint8)
    rgb[0, 0] = [10, 20, 30]

    masks = refiner.segment_classes(rgb, ["chair"])
    assert seen["dino_image"][0, 0].tolist() == [30, 20, 10]
    assert seen["sam_image"][0, 0].tolist() == [10, 20, 30]
    assert masks["chair"][1, 1] == 2.0


def test_grounded_sam_batch_compatibility_uses_ordered_single_image_calls():
    refiner = GroundedSAMWaypointRefiner.__new__(GroundedSAMWaypointRefiner)
    segment_calls = []
    refine_calls = []

    def _segment(image, classes):
        segment_calls.append((int(image[0, 0, 0]), tuple(classes)))
        return {classes[0]: np.ones((1, 1), dtype=np.float32)}

    def _refine(image, target, target_region, **kwargs):
        refine_calls.append(
            (
                int(image[0, 0, 0]),
                target,
                target_region,
                tuple(kwargs["grounding_classes"]),
                kwargs["source_lhx"],
            )
        )
        return GroundedSAMRefineResult(detected=True, label=target)

    refiner.segment_classes = _segment
    refiner.refine = _refine
    images = [
        np.full((1, 1, 3), value, dtype=np.uint8)
        for value in (1, 2)
    ]

    segmented = refiner.segment_classes_batch([
        {"image_rgb": images[0], "classes": ["chair"]},
        {"image_rgb": images[1], "classes": ["door"]},
    ])
    refined = refiner.refine_batch([
        {
            "image_rgb": images[0],
            "target": "chair",
            "target_region": "left",
            "grounding_classes": ["chair"],
            "source_lhx": True,
        },
        {
            "image_rgb": images[1],
            "target": "door",
            "target_region": "right",
            "grounding_classes": ["door"],
            "source_lhx": False,
        },
    ])

    assert segment_calls == [(1, ("chair",)), (2, ("door",))]
    assert refine_calls == [
        (1, "chair", "left", ("chair",), True),
        (2, "door", "right", ("door",), False),
    ]
    assert [next(iter(row)) for row in segmented] == ["chair", "door"]
    assert [result.label for result in refined] == ["chair", "door"]


def test_grounded_sam_waypoint_uses_target_and_lhx_largest_box_fallback():
    seen = {}

    class _Dino:
        def predict_with_classes(self, image, classes, **_kwargs):
            seen["classes"] = list(classes)
            seen["image"] = image.copy()
            return SimpleNamespace(
                xyxy=np.asarray(
                    [[1, 1, 3, 3], [0, 0, 9, 8]], dtype=np.float32
                ),
                confidence=None,
                class_id=np.asarray([0, 0]),
            )

    refiner = GroundedSAMWaypointRefiner.__new__(GroundedSAMWaypointRefiner)
    refiner.grounding_dino_model = _Dino()
    refiner.box_threshold = 0.25
    refiner.text_threshold = 0.25
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    rgb[0, 0] = [10, 20, 30]

    result = refiner.refine(rgb, "red rug")

    assert seen["classes"] == ["red rug"]
    assert seen["image"][0, 0].tolist() == [30, 20, 10]
    assert result.detected
    assert result.point_2d is None
    assert result.bbox_2d == [0.0, 0.0, 900.0, 800.0]


def test_grounded_sam_source_lhx_uses_global_highest_confidence_scene_box():
    class _Dino:
        def predict_with_classes(self, image, classes, **_kwargs):
            assert classes == [
                "stone steps", "stairs", "stairway", "staircase", "steps"
            ]
            return SimpleNamespace(
                xyxy=np.asarray(
                    [[60, 10, 95, 90], [0, 0, 100, 100]], dtype=np.float32
                ),
                confidence=np.asarray([0.95, 0.99], dtype=np.float32),
                class_id=np.asarray([0, 1]),
            )

    refiner = GroundedSAMWaypointRefiner.__new__(GroundedSAMWaypointRefiner)
    refiner.grounding_dino_model = _Dino()
    refiner.box_threshold = 0.25
    refiner.text_threshold = 0.25
    refiner.waypoint_scene_box_area_ratio = 0.65
    refiner.waypoint_scene_edge_margin_ratio = 0.02
    refiner.reject_scene_region_source_waypoint = False

    result = refiner.refine(
        np.zeros((100, 100, 3), dtype=np.uint8),
        "stone steps",
        "left",
        grounding_classes=[
            "stone steps", "stairs", "stairway", "staircase", "steps"
        ],
        source_lhx=True,
    )

    assert result.detected
    assert result.label == "stairs"
    assert result.confidence == pytest.approx(0.99)
    assert result.bbox_2d == [0.0, 0.0, 1000.0, 1000.0]
    assert result.fallback_reason == ""


def test_grounded_sam_robust_source_rejects_scene_box():
    class _Dino:
        def predict_with_classes(self, image, classes, **_kwargs):
            return SimpleNamespace(
                xyxy=np.asarray([[0, 0, 100, 100]], dtype=np.float32),
                confidence=np.asarray([0.99], dtype=np.float32),
                class_id=np.asarray([0]),
            )

    refiner = GroundedSAMWaypointRefiner.__new__(GroundedSAMWaypointRefiner)
    refiner.grounding_dino_model = _Dino()
    refiner.box_threshold = 0.25
    refiner.text_threshold = 0.25
    refiner.waypoint_scene_box_area_ratio = 0.65
    refiner.waypoint_scene_edge_margin_ratio = 0.02
    refiner.reject_scene_region_source_waypoint = True

    result = refiner.refine(
        np.zeros((100, 100, 3), dtype=np.uint8),
        "hallway",
        "any",
        grounding_classes=["hallway"],
        source_lhx=True,
    )

    assert not result.detected
    assert result.bbox_2d is None
    assert result.fallback_reason == "scene_region_bbox"


def test_grounded_sam_service_pool_routes_slots_and_restores_job_order():
    barrier = threading.Barrier(4)

    class _Client:
        def __init__(self, service_id):
            self.service_id = service_id
            self.payloads = []

        def request(self, method, payload):
            assert method == "segment_classes_batch"
            self.payloads.append(payload)
            barrier.wait(timeout=2.0)
            return [
                {"service": self.service_id, "token": job["token"]}
                for job in payload
            ]

    pool = GroundedSAMServicePool.__new__(GroundedSAMServicePool)
    pool.slots_per_service = 2
    pool._clients = [_Client(i) for i in range(4)]
    pool._executor = ThreadPoolExecutor(max_workers=4)
    jobs = [
        {"env_i": env_i, "token": f"job-{env_i}"}
        for env_i in (6, 0, 4, 2, 7, 1, 5, 3)
    ]

    results = pool.segment_classes_batch(jobs)
    pool._executor.shutdown(wait=True)

    assert [row["token"] for row in results] == [
        job["token"] for job in jobs
    ]
    assert [row["service"] for row in results] == [3, 0, 2, 1, 3, 0, 2, 1]
    assert [
        [job["token"] for job in client.payloads[0]]
        for client in pool._clients
    ] == [
        ["job-0", "job-1"],
        ["job-2", "job-3"],
        ["job-4", "job-5"],
        ["job-6", "job-7"],
    ]


def test_grounded_sam_server_dispatch_preserves_batch_order():
    class _Model:
        def __init__(self):
            self.batch_calls = []

        def segment_classes(self, _image, classes):
            return {classes[0]: np.ones((2, 2), dtype=np.float32)}

        def refine(self, _image, target):
            return GroundedSAMRefineResult(
                detected=True, label=target, confidence=0.75
            )

        def segment_classes_batch(self, payload):
            self.batch_calls.append(("segment", len(payload)))
            return [self.segment_classes(job["image_rgb"], job["classes"]) for job in payload]

        def refine_batch(self, payload):
            self.batch_calls.append(("refine", len(payload)))
            return [
                self.refine(job["image_rgb"], job["target"])
                for job in payload
            ]

    model = _Model()
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    masks, should_stop = dispatch_request(
        model,
        "segment_classes_batch",
        [
            {"image_rgb": image, "classes": ["chair"]},
            {"image_rgb": image, "classes": ["door"]},
        ],
    )
    refined, _ = dispatch_request(
        model,
        "refine_batch",
        [
            {"image_rgb": image, "target": "chair"},
            {"image_rgb": image, "target": "door"},
        ],
    )

    assert not should_stop
    assert [next(iter(row)) for row in masks] == ["chair", "door"]
    assert [row["label"] for row in refined] == ["chair", "door"]
    assert model.batch_calls == [("segment", 2), ("refine", 2)]


def test_grounded_sam_server_serializes_model_across_client_connections():
    class _Connection:
        def __init__(self):
            self.requests = [
                {"method": "segment_classes_batch", "payload": []},
                {"method": "close", "payload": None},
            ]
            self.responses = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class _Model:
        def __init__(self):
            self.guard = threading.Lock()
            self.active = 0
            self.max_active = 0

        def segment_classes_batch(self, _payload):
            with self.guard:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.03)
            with self.guard:
                self.active -= 1
            return []

    def recv(connection):
        if not connection.requests:
            raise ConnectionError
        return connection.requests.pop(0)

    def send(connection, payload):
        connection.responses.append(payload)

    model = _Model()
    model_lock = threading.Lock()
    connections = [_Connection(), _Connection()]
    threads = [
        threading.Thread(
            target=serve_connection,
            args=(connection, model, model_lock, recv, send),
        )
        for connection in connections
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert model.max_active == 1
    assert all(connection.responses[-1]["result"] == {"closed": True}
               for connection in connections)


def test_stop_check_template_renders_literal_json_schema():
    text = build_stop_check_text(
        "wait by the exercise equipment", target="exercise equipment"
    )
    assert '"analysis":' in text
    assert '"decision": "STOP" or "CONTINUE"' in text
    assert text.count("<image>") == 4


def test_source_stop_check_parts_preserve_interleaved_view_boundary():
    prefix, suffix = build_source_stop_check_parts(
        "wait by the exercise equipment", target="exercise equipment"
    )
    assert "Current Views:" in prefix
    assert '"analysis":' in suffix
    assert "{current_views}" not in prefix + suffix
