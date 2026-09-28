from rlinf.envs.genark.episode_assignment import FixedEpisodeAssignment
from rlinf.envs.genark.hard_pool import build_hard_pool_manifest
from rlinf.envs.genark.terminal_navigation_score import classify_hard_pool_bucket


def test_hard_pool_bucket_order_is_stable_across_repeated_calls():
    kwargs = dict(
        clean_stop_count=2,
        terminal_score_std=0.4,
        max_final_euclidean_dtg_m=3.5,
    )
    labels = [classify_hard_pool_bucket(**kwargs) for _ in range(8)]
    assert labels == ["mixed_support"] * 8


def test_assignment_cursor_resumes_exactly():
    episodes = [
        {
            "episode_id": str(episode_id),
            "scene_id": "mp3d/Z6MFQCViBuw/Z6MFQCViBuw.glb",
            "instruction": {"instruction_text": f"go {episode_id}"},
        }
        for episode_id in (586, 824, 1301)
    ]
    assignment = FixedEpisodeAssignment(
        episodes,
        [
            {"episode_id": "586"},
            {"episode_id": "824"},
            {"episode_id": "1301"},
            {"episode_id": "586"},
        ],
        scene_id="mp3d/Z6MFQCViBuw/Z6MFQCViBuw.glb",
    )
    first, _ = assignment.next_episode()
    second, _ = assignment.next_episode()
    state = assignment.state_dict()

    restored = FixedEpisodeAssignment(
        episodes,
        [
            {"episode_id": "586"},
            {"episode_id": "824"},
            {"episode_id": "1301"},
            {"episode_id": "586"},
        ],
        scene_id="mp3d/Z6MFQCViBuw/Z6MFQCViBuw.glb",
    )
    restored.load_state_dict(state)
    third, _ = restored.next_episode()
    fourth, new_pass = restored.next_episode()

    assert first["episode_id"] == "586"
    assert second["episode_id"] == "824"
    assert third["episode_id"] == "1301"
    assert fourth["episode_id"] == "586"
    assert new_pass is False
    assert restored.state_dict()["cursor"] == 4


def test_assignment_set_global_step_zero_does_not_rewind_primed_cursor():
    episodes = [
        {
            "episode_id": str(episode_id),
            "scene_id": "mp3d/Z6MFQCViBuw/Z6MFQCViBuw.glb",
            "instruction": {"instruction_text": f"go {episode_id}"},
        }
        for episode_id in (586, 824, 1301)
    ]
    groups = [{"episode_id": "586"}, {"episode_id": "824"}, {"episode_id": "1301"}]
    assignment = FixedEpisodeAssignment(
        episodes, groups, scene_id="mp3d/Z6MFQCViBuw/Z6MFQCViBuw.glb"
    )
    first, _ = assignment.next_episode()
    assignment.set_global_step(0)
    second, _ = assignment.next_episode()
    assert first["episode_id"] == "586"
    assert second["episode_id"] == "824"


def test_fresh_resume_at_step_six_starts_at_group_six():
    episodes = [
        {
            "episode_id": str(episode_id),
            "scene_id": "mp3d/Z6MFQCViBuw/Z6MFQCViBuw.glb",
            "instruction": {"instruction_text": f"go {episode_id}"},
        }
        for episode_id in (586, 824, 1301, 1133, 576, 469)
    ]
    groups = [{"episode_id": ep} for ep in ("586", "824", "1301", "1133", "576", "469") * 2]
    assignment = FixedEpisodeAssignment(
        episodes, groups, scene_id="mp3d/Z6MFQCViBuw/Z6MFQCViBuw.glb"
    )
    assignment.set_global_step(6)
    first, _ = assignment.next_episode()
    assert first["episode_id"] == "586"
    assert assignment.state_dict()["cursor"] == 7


def test_hard_pool_builder_requires_five_trials_and_is_deterministic():
    rows = []
    for trial in range(5):
        rows.append(
            {
                "scene_id": "mp3d/Z6MFQCViBuw/Z6MFQCViBuw.glb",
                "episode_id": "824",
                "distance_to_goal": 1.2 + 0.1 * trial,
                "termination_cause": "missed_stop",
                "clean_stop_success": 0.0,
                "start_distance_to_goal": 6.5,
            }
        )
    kwargs = dict(
        source_policy_checkpoint="/ckpt/checkpoint-1200",
        source_policy_fingerprint="sha256-test",
        distance_floor_m=2.0,
        clean_stop_bonus=1.0,
        success_distance=3.0,
        required_trials=5,
    )
    left = build_hard_pool_manifest(rows, **kwargs)
    right = build_hard_pool_manifest(list(reversed(rows)), **kwargs)
    assert left == right
    assert left["episodes"][0]["bucket"] == "flat_missed_stop"
    assert left["episodes"][0]["in_training_pool"] is False

    try:
        build_hard_pool_manifest(rows[:4], **kwargs)
    except ValueError as exc:
        assert "expected 5" in str(exc)
    else:
        raise AssertionError("builder must reject the wrong trial count")
