#!/usr/bin/env python3
"""Recompute Robostral terminal scores from a completed RFT train.log.

Reads Euclidean ``distance_to_goal`` from historical ``[GenArk][ep-diag]``
lines. Does not run Genesis and does not modify the log directory.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rlinf.envs.genark.terminal_navigation_score import (
    DEFAULT_CLEAN_STOP_BONUS,
    DEFAULT_DISTANCE_FLOOR_M,
    TERMINAL_GRPO_STD_EPS,
    classify_replay_group,
    compute_terminal_grpo_outcome_advantages,
    compute_terminal_navigation_score,
    diagnostic_is_clean_stop,
)

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
EP_DIAG_RE = re.compile(
    r"\[GenArk\]\[ep-diag\]\s+"
    r"env=(?P<env>\d+)\s+"
    r"ep=(?P<ep>\S+)\s+"
    r"cause=(?P<cause>\S+)\s+"
    r"success_type=(?P<success_type>\S+)\s+"
    r"steps=(?P<steps>\d+)\s+"
    r"parse_fail=(?P<parse_fail>\d+)\s+"
    r"start_dtg=(?P<start_dtg>[-0-9.]+)m\s+"
    r"final_dtg=(?P<final_dtg>[-0-9.]+)m\s+"
    r"stop_dtg=(?P<stop_dtg>[-0-9.]+)m\s+"
    r"min_dtg=(?P<min_dtg>[-0-9.]+)m\s+"
    r"best_prog=(?P<best_prog>[-0-9.]+)m"
)
GROUP_DIAG_RE = re.compile(
    r"\[GRPO\]\[group-diag\]\s+"
    r"stage=(?P<stage>\d+)\s+"
    r"group=(?P<group>\d+)\s+"
    r"ep=(?P<ep>\S+)\s+"
    r"reward_mean=(?P<reward_mean>[-0-9.]+)\s+"
    r"reward_std=(?P<reward_std>[-0-9.]+)\s+"
    r".*?"
    r"success_count=(?P<success_count>\d+)\s+"
    r"all_failure=(?P<all_failure>\d+)"
)


def _strip_ansi(line: str) -> str:
    return ANSI_RE.sub("", line)


def parse_train_log(log_path: Path) -> list[dict]:
    pending_by_env: dict[int, dict] = {}
    groups: list[dict] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = _strip_ansi(raw)
            ep_match = EP_DIAG_RE.search(line)
            if ep_match:
                row = ep_match.groupdict()
                env_i = int(row["env"])
                pending_by_env[env_i] = {
                    "env_id": env_i,
                    "episode_id": str(row["ep"]),
                    "termination_cause": row["cause"],
                    "success_type": row["success_type"],
                    "steps": int(row["steps"]),
                    "start_dtg": float(row["start_dtg"]),
                    "distance_to_goal": float(row["final_dtg"]),
                    "stop_dtg": float(row["stop_dtg"]),
                    "min_dtg": float(row["min_dtg"]),
                    "best_dtg_progress": float(row["best_prog"]),
                }
                continue
            group_match = GROUP_DIAG_RE.search(line)
            if not group_match:
                continue
            members = [pending_by_env[i] for i in sorted(pending_by_env) if i in pending_by_env]
            if len(members) < 2:
                continue
            groups.append(
                {
                    "stage": int(group_match.group("stage")),
                    "group": int(group_match.group("group")),
                    "logged_episode_id": str(group_match.group("ep")),
                    "logged_success_count": int(group_match.group("success_count")),
                    "logged_all_failure": int(group_match.group("all_failure")),
                    "logged_reward_std": float(group_match.group("reward_std")),
                    "members": [dict(row) for row in members],
                }
            )
            pending_by_env = {}
    return groups


def score_group(
    group: dict,
    *,
    distance_floor_m: float,
    clean_stop_bonus: float,
    success_distance: float,
    group_size: int,
) -> dict:
    members = group["members"]
    if len(members) != group_size:
        raise ValueError(
            f"group ep={group.get('logged_episode_id')} has "
            f"{len(members)} members, expected {group_size}"
        )
    episode_ids = {str(row["episode_id"]) for row in members}
    if len(episode_ids) != 1:
        raise ValueError(f"mixed episode IDs in one group: {sorted(episode_ids)}")
    scores = []
    clean_stops = []
    dtgs = []
    causes = []
    for row in members:
        dtg = float(row["distance_to_goal"])
        if not math.isfinite(dtg):
            raise ValueError(f"non-finite DTG in episode {row['episode_id']} env={row['env_id']}")
        clean = diagnostic_is_clean_stop(row, success_distance=success_distance)
        score = compute_terminal_navigation_score(
            dtg,
            clean,
            distance_floor_m=distance_floor_m,
            clean_stop_bonus=clean_stop_bonus,
        )
        scores.append(score)
        clean_stops.append(clean)
        dtgs.append(dtg)
        causes.append(str(row["termination_cause"]))
        row["clean_stop"] = clean
        row["terminal_score"] = score
    score_t = torch.tensor(scores, dtype=torch.float32)
    advantages = compute_terminal_grpo_outcome_advantages(
        score_t,
        group_size,
        episode_ids=[row["episode_id"] for row in members],
    )
    if not torch.isfinite(advantages).all():
        raise ValueError("non-finite terminal-GRPO advantage")
    score_std = float(score_t.std(unbiased=True).item()) if score_t.numel() > 1 else 0.0
    k = int(sum(clean_stops))
    label = classify_replay_group(
        scores,
        clean_stops,
        dtgs,
        distance_floor_m=distance_floor_m,
    )
    return {
        "episode_id": next(iter(episode_ids)),
        "logged_episode_id": group["logged_episode_id"],
        "logged_success_count": group["logged_success_count"],
        "n": len(members),
        "clean_stop_count": k,
        "wrong_stop_count": sum(cause == "stop" and not clean for cause, clean in zip(causes, clean_stops)),
        "proximity_without_stop_count": sum(
            (not clean) and dtg < success_distance and cause != "stop"
            for clean, dtg, cause in zip(clean_stops, dtgs, causes)
        ),
        "score_mean": float(score_t.mean().item()),
        "score_std": score_std,
        "score_min": float(score_t.min().item()),
        "score_max": float(score_t.max().item()),
        "final_dtg_mean": float(sum(dtgs) / len(dtgs)),
        "final_dtg_min": float(min(dtgs)),
        "final_dtg_max": float(max(dtgs)),
        "zero_std": score_std < TERMINAL_GRPO_STD_EPS,
        "nonzero_advantage": bool((advantages.abs() > 0).any().item()),
        "advantage_mean": float(advantages.mean().item()),
        "label": label,
        "members": members,
        "advantages": [float(v) for v in advantages.tolist()],
    }


def summarize(scored: list[dict]) -> dict:
    k0 = [row for row in scored if row["clean_stop_count"] == 0]
    return {
        "n_groups": len(scored),
        "k0_groups": len(k0),
        "k0_nonzero_score_std": sum(not row["zero_std"] for row in k0),
        "navigation_informative": sum(row["label"] == "navigation_informative" for row in scored),
        "mixed_stop_support": sum(row["label"] == "mixed_stop_support" for row in scored),
        "flat_missed_stop": sum(row["label"] == "flat_missed_stop" for row in scored),
        "uninformative": sum(row["label"] == "uninformative" for row in scored),
        "nonfinite": 0,
        "episode_586_groups": [
            {
                "clean_stop_count": row["clean_stop_count"],
                "score_std": row["score_std"],
                "label": row["label"],
            }
            for row in scored
            if str(row["episode_id"]) == "586"
        ],
        "by_episode": _by_episode(scored),
    }


def _by_episode(scored: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in scored:
        ep = str(row["episode_id"])
        bucket = out.setdefault(
            ep,
            {
                "n_groups": 0,
                "k0_groups": 0,
                "k0_nonzero_score_std": 0,
                "mixed_stop_support": 0,
                "labels": [],
            },
        )
        bucket["n_groups"] += 1
        bucket["labels"].append(row["label"])
        if row["clean_stop_count"] == 0:
            bucket["k0_groups"] += 1
            if not row["zero_std"]:
                bucket["k0_nonzero_score_std"] += 1
        if row["label"] == "mixed_stop_support":
            bucket["mixed_stop_support"] += 1
    return out


def render_markdown(summary: dict, scored: list[dict], args: argparse.Namespace) -> str:
    lines = [
        "# Robostral Terminal-Score Offline Replay",
        "",
        f"- Source log: `{args.log_dir}`",
        f"- Distance contract: Habitat-coordinate Euclidean `distance_to_goal`",
        f"- Score: `-max({args.distance_floor_m}, DTG) + {args.clean_stop_bonus} * clean_stop`",
        f"- Historical missed-STOP truncation is preserved (read from logs only)",
        f"- Groups parsed: **{summary['n_groups']}**",
        "",
        "## Gate 6.2 Acceptance",
        "",
        f"- Old `K=0/8` groups: **{summary['k0_groups']}**",
        f"- Of those, nonzero terminal-score std: **{summary['k0_nonzero_score_std']}** (report-only, not a pass/fail target)",
        f"- `navigation_informative`: **{summary['navigation_informative']}**",
        f"- `mixed_stop_support`: **{summary['mixed_stop_support']}**",
        f"- `flat_missed_stop`: **{summary['flat_missed_stop']}**",
        f"- `uninformative`: **{summary['uninformative']}**",
        f"- NaN/Inf: **{summary['nonfinite']}**",
        "",
        "### Episode 586",
        "",
    ]
    if summary["episode_586_groups"]:
        for i, row in enumerate(summary["episode_586_groups"], start=1):
            lines.append(
                f"- group {i}: K={row['clean_stop_count']}/8 "
                f"score_std={row['score_std']:.4f} label={row['label']}"
            )
    else:
        lines.append("- no episode 586 groups parsed")
    lines.extend(["", "### Per episode", "", "| episode | groups | K=0 | K=0 nonzero std | mixed STOP |", "|---|---:|---:|---:|---:|"])
    for ep, bucket in sorted(summary["by_episode"].items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0]):
        lines.append(
            f"| {ep} | {bucket['n_groups']} | {bucket['k0_groups']} | "
            f"{bucket['k0_nonzero_score_std']} | {bucket['mixed_stop_support']} |"
        )
    lines.extend(["", "### Group detail", "", "| # | episode | K | score mean | score std | DTG mean | label | nonzero A |", "|---|---|---:|---:|---:|---:|---|---|"])
    for i, row in enumerate(scored, start=1):
        lines.append(
            f"| {i} | {row['episode_id']} | {row['clean_stop_count']}/{row['n']} | "
            f"{row['score_mean']:.3f} | {row['score_std']:.4f} | "
            f"{row['final_dtg_mean']:.3f} | {row['label']} | "
            f"{int(row['nonzero_advantage'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "log_dir",
        type=Path,
        nargs="?",
        default=Path("logs/20260816-205104-sft1200-episode-tier-curriculum-rloo-rft"),
    )
    parser.add_argument("--distance-floor-m", type=float, default=DEFAULT_DISTANCE_FLOOR_M)
    parser.add_argument("--clean-stop-bonus", type=float, default=DEFAULT_CLEAN_STOP_BONUS)
    parser.add_argument("--success-distance", type=float, default=3.0)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    log_dir = args.log_dir if args.log_dir.is_absolute() else REPO / args.log_dir
    log_path = log_dir / "train.log" if log_dir.is_dir() else log_dir
    if not log_path.is_file():
        raise FileNotFoundError(log_path)

    groups = parse_train_log(log_path)
    scored = [
        score_group(
            group,
            distance_floor_m=args.distance_floor_m,
            clean_stop_bonus=args.clean_stop_bonus,
            success_distance=args.success_distance,
            group_size=args.group_size,
        )
        for group in groups
    ]
    summary = summarize(scored)
    payload = {"summary": summary, "groups": scored}
    report = render_markdown(summary, scored, args)
    print(report)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
