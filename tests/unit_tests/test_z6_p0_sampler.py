"""P0 densify sampling: train-level episodes_file must beat OpenNav-100 init_params."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rlinf.envs.genark.genark_env import (
    EpisodeBalancer,
    _load_episodes,
    resolve_train_episodes_file,
)

REPO = Path(__file__).resolve().parents[2]
Z6_P0 = REPO / "examples/embodiment/config/OpenNav_z6_p0_train.json"
OPENNAV100 = "/home/clk/workspace/genark/data/datasets/OpenNav_R2R-CE_100_bertidx.json"
EVAL_IDS = frozenset({"207", "432", "550", "559", "705", "586"})
OPENNAV5 = frozenset({"824", "1301", "1133", "576", "469"})


def _sample_ids(episodes: list[dict], n: int = 24) -> list[str]:
    rng = np.random.default_rng(0)
    bal = EpisodeBalancer(episodes, rng)
    ids = []
    for _ in range(n):
        ep, _ = bal.next_episode()
        ids.append(str(ep["episode_id"]))
    return ids


def test_resolve_train_episodes_file_prefers_top_level_over_init_params():
    cfg = SimpleNamespace(
        episodes_file=str(Z6_P0),
        init_params=SimpleNamespace(episodes_file=OPENNAV100),
    )
    assert resolve_train_episodes_file(cfg) == str(Z6_P0)


def test_init_params_opennav100_plus_blocklist_collapses_to_five():
    cfg = SimpleNamespace(
        episodes_file=None,
        init_params=SimpleNamespace(episodes_file=OPENNAV100),
    )
    path = resolve_train_episodes_file(cfg)
    eps = [
        e
        for e in _load_episodes(path)
        if "Z6MFQCViBuw" in str(e.get("scene_id", ""))
        and str(e.get("episode_id")) not in EVAL_IDS
    ]
    ids = {str(e["episode_id"]) for e in eps}
    assert ids == OPENNAV5


def test_z6_p0_json_has_more_than_five_train_ids():
    eps = [
        e
        for e in _load_episodes(str(Z6_P0))
        if str(e.get("episode_id")) not in EVAL_IDS
    ]
    ids = {str(e["episode_id"]) for e in eps}
    assert len(ids) > 8
    assert ids.isdisjoint(EVAL_IDS)
    assert not ids.issubset(OPENNAV5)


def test_balancer_on_z6_p0_leaves_the_five_opennav_ids():
    cfg = SimpleNamespace(
        episodes_file=str(Z6_P0),
        init_params=SimpleNamespace(episodes_file=OPENNAV100),
    )
    path = resolve_train_episodes_file(cfg)
    eps = [
        e
        for e in _load_episodes(path)
        if str(e.get("episode_id")) not in EVAL_IDS
    ]
    sampled = _sample_ids(eps, n=24)
    unique = set(sampled)
    assert unique.isdisjoint(EVAL_IDS)
    assert not unique.issubset(OPENNAV5)
    extra = sorted(unique - OPENNAV5, key=lambda x: int(x) if x.isdigit() else x)
    assert extra, sampled
