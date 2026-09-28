#!/usr/bin/env python3
"""Analyze DITA shadow-judge scores and frozen Qwen termination features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute binary ROC AUC using average ranks for tied scores."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = labels == 1
    n_positive = int(positives.sum())
    n_negative = int((~positives).sum())
    if not n_positive or not n_negative:
        raise ValueError("ROC AUC requires both classes")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    rank_sum = ranks[positives].sum()
    return float(
        (rank_sum - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def _load_records(root: Path) -> dict[str, np.ndarray]:
    paths = sorted(root.rglob("shadow_rank*.npz"))
    if not paths:
        raise FileNotFoundError(f"no shadow_rank*.npz files below {root}")
    chunks: dict[str, list[np.ndarray]] = {}
    for path in paths:
        with np.load(path) as data:
            for key in data.files:
                chunks.setdefault(key, []).append(np.asarray(data[key]))
    required = {
        "hidden",
        "probability",
        "oracle_within_radius",
        "episode_id",
    }
    missing = required.difference(chunks)
    if missing:
        raise ValueError(f"shadow artifacts missing fields: {sorted(missing)}")
    return {key: np.concatenate(values, axis=0) for key, values in chunks.items()}


def _ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(1, labels.size)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (probabilities >= lower) & (
            probabilities <= upper if upper == 1.0 else probabilities < upper
        )
        if not selected.any():
            continue
        value += selected.sum() / total * abs(
            probabilities[selected].mean() - labels[selected].mean()
        )
    return float(value)


def _readiness(auc_samples: np.ndarray) -> float:
    mapped = np.clip((auc_samples - 0.55) / 0.30, 0.0, 1.0)
    return float(mapped.mean())


def _group_bootstrap_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    samples = []
    for _ in range(repeats):
        drawn = rng.choice(unique_groups, size=unique_groups.size, replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in drawn])
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size < 2:
            continue
        samples.append(_roc_auc(sampled_labels, scores[indices]))
    if not samples:
        raise ValueError("group bootstrap produced no two-class resamples")
    return np.asarray(samples, dtype=np.float64)


def _probe_oof(
    hidden: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    unique_groups = np.unique(groups)
    n_splits = min(5, unique_groups.size)
    if n_splits < 2:
        raise ValueError("linear probe requires at least two episode groups")
    rng = np.random.default_rng(seed)
    shuffled_groups = unique_groups.copy()
    rng.shuffle(shuffled_groups)
    folds = np.array_split(shuffled_groups, n_splits)
    predictions = np.full(labels.shape, np.nan, dtype=np.float64)
    for valid_groups in folds:
        valid_idx = np.flatnonzero(np.isin(groups, valid_groups))
        train_idx = np.flatnonzero(~np.isin(groups, valid_groups))
        if np.unique(labels[train_idx]).size < 2:
            continue
        train_hidden = hidden[train_idx].astype(np.float64)
        valid_hidden = hidden[valid_idx].astype(np.float64)
        mean = train_hidden.mean(axis=0)
        scale = train_hidden.std(axis=0)
        scale[scale < 1e-6] = 1.0
        train_hidden = (train_hidden - mean) / scale
        valid_hidden = (valid_hidden - mean) / scale
        train_design = np.concatenate(
            [train_hidden, np.ones((train_hidden.shape[0], 1))], axis=1
        )
        valid_design = np.concatenate(
            [valid_hidden, np.ones((valid_hidden.shape[0], 1))], axis=1
        )
        class_counts = np.bincount(labels[train_idx], minlength=2).astype(float)
        sample_weights = 0.5 / class_counts[labels[train_idx]]
        weighted_design = train_design * np.sqrt(sample_weights[:, None])
        weighted_targets = labels[train_idx] * np.sqrt(sample_weights)
        # The Qwen hidden dimension is much larger than the number of sampled
        # decisions. Solve ridge regression in the sample-space dual to avoid a
        # dense hidden_dim x hidden_dim system.
        dual = np.linalg.solve(
            weighted_design @ weighted_design.T
            + np.eye(weighted_design.shape[0], dtype=np.float64),
            weighted_targets,
        )
        weights = weighted_design.T @ dual
        raw_scores = valid_design @ weights
        predictions[valid_idx] = 1.0 / (1.0 + np.exp(-raw_scores))
    if not np.isfinite(predictions).all():
        raise ValueError(
            "episode-disjoint probe could not produce predictions for every sample"
        )
    return predictions


def analyze(root: Path, *, bootstrap_repeats: int, seed: int) -> dict:
    data = _load_records(root)
    labels = data["oracle_within_radius"].astype(np.int64).reshape(-1)
    probabilities = data["probability"].astype(np.float64).reshape(-1)
    groups = data["episode_id"].astype(np.int64).reshape(-1)
    hidden = data["hidden"]
    finite = np.isfinite(probabilities) & np.isfinite(hidden).all(axis=1)
    finite &= groups >= 0
    labels = labels[finite]
    probabilities = probabilities[finite]
    groups = groups[finite]
    hidden = hidden[finite]
    if "trial_id" in data and "decision_index" in data:
        trial_ids = data["trial_id"].reshape(-1)[finite]
        decision_indices = data["decision_index"].reshape(-1)[finite]
        identities = np.stack(
            [groups, trial_ids.astype(np.int64), decision_indices.astype(np.int64)],
            axis=1,
        )
        _, unique_indices = np.unique(identities, axis=0, return_index=True)
        unique_indices.sort()
        labels = labels[unique_indices]
        probabilities = probabilities[unique_indices]
        groups = groups[unique_indices]
        hidden = hidden[unique_indices]
    if np.unique(labels).size != 2:
        raise ValueError(
            "shadow dataset needs both within-radius and outside-radius decisions"
        )

    predictions = probabilities >= 0.5
    shadow_auc = _roc_auc(labels, probabilities)
    shadow_boot = _group_bootstrap_auc(
        labels,
        probabilities,
        groups,
        repeats=bootstrap_repeats,
        seed=seed,
    )
    probe_scores = _probe_oof(hidden, labels, groups, seed=seed)
    probe_auc = _roc_auc(labels, probe_scores)
    probe_boot = _group_bootstrap_auc(
        labels,
        probe_scores,
        groups,
        repeats=bootstrap_repeats,
        seed=seed + 1,
    )
    per_episode = {}
    for episode_id in np.unique(groups):
        selected = groups == episode_id
        episode_labels = labels[selected]
        episode_probabilities = probabilities[selected]
        episode_result = {
            "samples": int(selected.sum()),
            "positive_samples": int(episode_labels.sum()),
            "mean_probability_terminal": (
                float(episode_probabilities[episode_labels == 1].mean())
                if (episode_labels == 1).any()
                else None
            ),
            "mean_probability_nonterminal": (
                float(episode_probabilities[episode_labels == 0].mean())
                if (episode_labels == 0).any()
                else None
            ),
        }
        episode_result["shadow_auc"] = (
            _roc_auc(episode_labels, episode_probabilities)
            if np.unique(episode_labels).size == 2
            else None
        )
        per_episode[str(int(episode_id))] = episode_result

    result = {
        "samples": int(labels.size),
        "positive_samples": int(labels.sum()),
        "negative_samples": int((labels == 0).sum()),
        "unique_episodes": int(np.unique(groups).size),
        "shadow_judge": {
            "auc": shadow_auc,
            "auc_ci95": np.quantile(shadow_boot, [0.025, 0.975]).tolist(),
            "readiness": _readiness(shadow_boot),
            "accuracy": float((labels == predictions).mean()),
            "balanced_accuracy": float(
                0.5
                * (
                    predictions[labels == 1].mean()
                    + (~predictions[labels == 0]).mean()
                )
            ),
            "precision": float(
                labels[predictions].mean() if predictions.any() else 0.0
            ),
            "terminal_recall": float(predictions[labels == 1].mean()),
            "specificity": float((~predictions[labels == 0]).mean()),
            "brier": float(np.mean((probabilities - labels) ** 2)),
            "ece_10bin": _ece(labels, probabilities),
            "mean_probability_terminal": float(probabilities[labels == 1].mean()),
            "mean_probability_nonterminal": float(
                probabilities[labels == 0].mean()
            ),
        },
        "linear_probe": {
            "auc": probe_auc,
            "auc_ci95": np.quantile(probe_boot, [0.025, 0.975]).tolist(),
            "readiness": _readiness(probe_boot),
        },
        "readout_gap_probe_minus_shadow_auc": probe_auc - shadow_auc,
        "per_episode": per_episode,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()
    result = analyze(
        args.artifact_dir,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
