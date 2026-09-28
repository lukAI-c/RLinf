from rlinf.workers.rollout.utils import derive_rollout_sampling_seed


def _seed(**overrides):
    identity = {
        "base_seed": 42,
        "global_step": 0,
        "worker_rank": 0,
        "request_index": 0,
        "collection_offset": 0,
    }
    identity.update(overrides)
    return derive_rollout_sampling_seed(**identity)


def test_rollout_sampling_seed_is_reproducible_and_rank_distinct():
    seeds = [_seed(worker_rank=rank) for rank in range(4)]

    assert len(set(seeds)) == 4
    assert seeds == [_seed(worker_rank=rank) for rank in range(4)]


def test_rollout_sampling_seed_changes_across_request_step_and_retry():
    seeds = {
        _seed(),
        _seed(request_index=1),
        _seed(global_step=1),
        _seed(collection_offset=1),
    }

    assert len(seeds) == 4
    assert all(0 <= seed < (1 << 31) - 1 for seed in seeds)
