#!/usr/bin/env python3
"""Compare Gate A.5 frozen evals: checkpoint-1200 vs global_step_6."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from rlinf.envs.genark.terminal_navigation_score import (
    compute_terminal_navigation_score,
    diagnostic_is_clean_stop,
)

EPISODES = ("586", "824", "1301", "1133", "576", "469")


def _load_trials(log_dir: Path) -> list[dict]:
    for name in ("all_episode_trials.json", "all_episode_metrics.json"):
        path = log_dir / name
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if name == "all_episode_trials.json":
            return list(payload.get("trials") or [])
        # Fall back: some dumps only keep episode averages.
        raise ValueError(f"{path} is episode-averaged; need all_episode_trials.json")
    # Parse eval.log / train.log ep-diag as last resort.
    log_path = log_dir / "eval.log"
    if not log_path.is_file():
        log_path = log_dir / "train.log"
    if not log_path.is_file():
        raise FileNotFoundError(f"No trial dump or log under {log_dir}")
    return _parse_ep_diag(log_path)


def _parse_ep_diag(path: Path) -> list[dict]:
    import re

    ansi = re.compile(r"\x1b\[[0-9;]*m")
    pat = re.compile(
        r"\[GenArk\]\[ep-diag\]\s+env=(?P<env>\d+)\s+ep=(?P<ep>\S+)\s+"
        r"cause=(?P<cause>\S+)\s+success_type=(?P<stype>\S+)\s+steps=(?P<steps>\d+)\s+"
        r".*?start_dtg=(?P<start>[-0-9.]+)m\s+final_dtg=(?P<dtg>[-0-9.]+)m\s+"
        r"min_dtg=(?P<min_dtg>[-0-9.]+)m"
    )
    rows = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pat.search(ansi.sub("", raw))
        if not match:
            continue
        rows.append(
            {
                "episode_id": match.group("ep"),
                "termination_cause": match.group("cause"),
                "success_type": match.group("stype"),
                "distance_to_goal": float(match.group("dtg")),
                "start_distance_to_goal": float(match.group("start")),
                "min_dtg": float(match.group("min_dtg")),
                "steps_taken": int(match.group("steps")),
            }
        )
    return rows


def _summarize(
    trials: list[dict],
    *,
    success_distance: float = 3.0,
    episodes: tuple[str, ...] | None = None,
) -> dict:
    wanted = tuple(episodes) if episodes is not None else EPISODES
    by_ep: dict[str, list[dict]] = defaultdict(list)
    for row in trials:
        ep = str(row.get("episode_id"))
        if ep in wanted:
            by_ep[ep].append(row)
    episodes = {}
    for ep in wanted:
        rows = by_ep.get(ep, [])
        scores = []
        dtgs = []
        min_dtgs = []
        clean = 0
        wrong = 0
        proximity = 0
        stop_false = 0
        oracle = 0
        for row in rows:
            dtg = float(row.get("distance_to_goal", row.get("final_dtg", math.nan)))
            is_clean = diagnostic_is_clean_stop(row, success_distance=success_distance)
            # Always recompute the endpoint-only score so older dumps that
            # stored a clean-STOP bonus remain comparable.
            score = compute_terminal_navigation_score(
                dtg, is_clean, clean_stop_bonus=0.0
            )
            scores.append(score)
            dtgs.append(dtg)
            min_dtgs.append(float(row.get("min_dtg", dtg)))
            cause = str(row.get("termination_cause", row.get("success_type", "")))
            if is_clean:
                clean += 1
            elif cause == "stop" or str(row.get("success_type")) == "wrong_stop":
                wrong += 1
            if str(row.get("success_type")) == "proximity" or (
                not is_clean
                and cause != "stop"
                and math.isfinite(dtg)
                and dtg < success_distance
            ):
                if not is_clean:
                    proximity += 1
            if cause != "stop":
                stop_false += 1
            if math.isfinite(dtg) and dtg < success_distance:
                oracle += 1
        n = len(rows)
        score_std = 0.0
        if n > 1:
            mean = sum(scores) / n
            score_std = math.sqrt(sum((s - mean) ** 2 for s in scores) / (n - 1))
        episodes[ep] = {
            "n_trials": n,
            "clean_stop_count": clean,
            "wrong_stop_count": wrong,
            "proximity_without_stop_count": proximity,
            "stop_false_count": stop_false,
            "oracle_success_count": oracle,
            "clean_sr": clean / n if n else 0.0,
            "oracle_sr": oracle / n if n else 0.0,
            "oracle_clean_gap": (oracle - clean) / n if n else 0.0,
            "mean_final_dtg": sum(dtgs) / n if n else math.nan,
            "min_final_dtg": min(dtgs) if dtgs else math.nan,
            "mean_min_dtg": sum(min_dtgs) / n if n else math.nan,
            "terminal_score_mean": sum(scores) / n if n else math.nan,
            "terminal_score_std": score_std,
        }
    return episodes


def _delta(left: float, right: float) -> float:
    if any(not math.isfinite(v) for v in (left, right)):
        return math.nan
    return right - left


def compare(base: dict, updated: dict) -> dict:
    per_episode = {}
    totals = {
        "clean_stop_count": [0, 0],
        "wrong_stop_count": [0, 0],
        "proximity_without_stop_count": [0, 0],
        "stop_false_count": [0, 0],
        "oracle_success_count": [0, 0],
        "n_trials": [0, 0],
        "score_sum": [0.0, 0.0],
        "dtg_sum": [0.0, 0.0],
    }
    episode_order = tuple(base.keys())
    for ep in episode_order:
        a = base[ep]
        b = updated[ep]
        per_episode[ep] = {
            "base": a,
            "step6": b,
            "delta": {
                "clean_stop_count": b["clean_stop_count"] - a["clean_stop_count"],
                "wrong_stop_count": b["wrong_stop_count"] - a["wrong_stop_count"],
                "proximity_without_stop_count": (
                    b["proximity_without_stop_count"] - a["proximity_without_stop_count"]
                ),
                "stop_false_count": b["stop_false_count"] - a["stop_false_count"],
                "oracle_clean_gap": _delta(a["oracle_clean_gap"], b["oracle_clean_gap"]),
                "mean_final_dtg": _delta(a["mean_final_dtg"], b["mean_final_dtg"]),
                "terminal_score_mean": _delta(
                    a["terminal_score_mean"], b["terminal_score_mean"]
                ),
            },
        }
        for key in (
            "clean_stop_count",
            "wrong_stop_count",
            "proximity_without_stop_count",
            "stop_false_count",
            "oracle_success_count",
            "n_trials",
        ):
            totals[key][0] += a[key]
            totals[key][1] += b[key]
        if math.isfinite(a["terminal_score_mean"]):
            totals["score_sum"][0] += a["terminal_score_mean"] * a["n_trials"]
            totals["score_sum"][1] += b["terminal_score_mean"] * b["n_trials"]
        if math.isfinite(a["mean_final_dtg"]):
            totals["dtg_sum"][0] += a["mean_final_dtg"] * a["n_trials"]
            totals["dtg_sum"][1] += b["mean_final_dtg"] * b["n_trials"]
    n0, n1 = totals["n_trials"]
    answers = {
        "terminal_score_improved": (
            totals["score_sum"][1] / max(n1, 1) > totals["score_sum"][0] / max(n0, 1)
        ),
        "clean_stop_decreased": totals["clean_stop_count"][1] < totals["clean_stop_count"][0],
        "closer_but_more_stop_false": (
            totals["dtg_sum"][1] / max(n1, 1) < totals["dtg_sum"][0] / max(n0, 1)
            and totals["stop_false_count"][1] > totals["stop_false_count"][0]
            and totals["clean_stop_count"][1] <= totals["clean_stop_count"][0]
        ),
        "episode_1133_local_shift": (
            _local_shift(per_episode["1133"]) if "1133" in per_episode else None
        ),
    }
    return {
        "per_episode": per_episode,
        "totals": {
            "base_clean_stop": totals["clean_stop_count"][0],
            "step6_clean_stop": totals["clean_stop_count"][1],
            "base_score_mean": totals["score_sum"][0] / max(n0, 1),
            "step6_score_mean": totals["score_sum"][1] / max(n1, 1),
            "base_dtg_mean": totals["dtg_sum"][0] / max(n0, 1),
            "step6_dtg_mean": totals["dtg_sum"][1] / max(n1, 1),
            "base_stop_false": totals["stop_false_count"][0],
            "step6_stop_false": totals["stop_false_count"][1],
            "base_oracle": totals["oracle_success_count"][0],
            "step6_oracle": totals["oracle_success_count"][1],
            "n_trials": [n0, n1],
        },
        "answers": answers,
    }


def _local_shift(row: dict) -> dict:
    delta = row["delta"]
    return {
        "dtg_improved": bool(delta["mean_final_dtg"] < -0.1),
        "clean_stop_dropped": bool(delta["clean_stop_count"] < 0),
        "stop_false_increased": bool(delta["stop_false_count"] > 0),
        "score_improved": bool(delta["terminal_score_mean"] > 0.05),
        "likely_nav_not_stop": bool(
            delta["mean_final_dtg"] < -0.1
            and delta["clean_stop_count"] <= 0
            and delta["stop_false_count"] >= 0
        ),
    }


def render(result: dict, base_dir: Path, step_dir: Path) -> str:
    answers = result["answers"]
    totals = result["totals"]
    lines = [
        "# Robostral Gate A.5 Frozen Behavioral Check",
        "",
        f"- Base: `{base_dir}` (checkpoint-1200)",
        f"- Updated: `{step_dir}` (Gate A global_step_6)",
        f"- Protocol: Z6 episodes {', '.join(result['per_episode'])}; 5 trials each; temperature 1.0; missed-STOP continuation",
        "",
        "## Answers",
        "",
        f"1. **global_step_6 improved terminal score?** "
        f"{'Yes' if answers['terminal_score_improved'] else 'No'} "
        f"({totals['base_score_mean']:.3f} → {totals['step6_score_mean']:.3f})",
        f"2. **Did it lower clean STOP?** "
        f"{'Yes' if answers['clean_stop_decreased'] else 'No'} "
        f"({totals['base_clean_stop']} → {totals['step6_clean_stop']} / {totals['n_trials'][0]})",
        f"3. **Closer but more stop=false?** "
        f"{'Yes' if answers['closer_but_more_stop_false'] else 'No'} "
        f"(DTG {totals['base_dtg_mean']:.3f} → {totals['step6_dtg_mean']:.3f}, "
        f"stop=false {totals['base_stop_false']} → {totals['step6_stop_false']})",
        f"4. **1133 large-grad local shift?** "
        f"{json.dumps(answers['episode_1133_local_shift']) if answers['episode_1133_local_shift'] is not None else 'n/a'}",
        "",
        "## Per episode",
        "",
        "| ep | K_base | K_step6 | DTG_base | DTG_step6 | S_base | S_step6 | prox_base | prox_step6 | gap_base | gap_step6 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ep in result["per_episode"]:
        row = result["per_episode"][ep]
        a, b = row["base"], row["step6"]
        lines.append(
            f"| {ep} | {a['clean_stop_count']}/{a['n_trials']} | "
            f"{b['clean_stop_count']}/{b['n_trials']} | "
            f"{a['mean_final_dtg']:.3f} | {b['mean_final_dtg']:.3f} | "
            f"{a['terminal_score_mean']:.3f} | {b['terminal_score_mean']:.3f} | "
            f"{a['proximity_without_stop_count']} | {b['proximity_without_stop_count']} | "
            f"{a['oracle_clean_gap']:.2f} | {b['oracle_clean_gap']:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_log_dir", type=Path)
    parser.add_argument("step6_log_dir", type=Path)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--episodes",
        default=",".join(EPISODES),
        help="Comma-separated episode IDs to score",
    )
    args = parser.parse_args()
    episode_ids = tuple(item.strip() for item in args.episodes.split(",") if item.strip())
    result = compare(
        _summarize(_load_trials(args.base_log_dir), episodes=episode_ids),
        _summarize(_load_trials(args.step6_log_dir), episodes=episode_ids),
    )
    text = render(result, args.base_log_dir, args.step6_log_dir)
    print(text)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
        args.report.with_suffix(".json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
