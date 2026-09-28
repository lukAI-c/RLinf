#!/usr/bin/env python3
"""Build the endpoint-only training report from train.log plus frozen evals."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_robostral_gate_a5 import compare, render, _load_trials, _summarize

ANSI = re.compile(r"\x1b\[[0-9;]*m")
GROUP = re.compile(
    r"\[GRPO\]\[group-diag\].*?ep=(?P<ep>\S+)\s+.*?success_count=(?P<k>\d+)\s+"
    r"all_failure=(?P<fail>\d+)\s+wrong_stop=(?P<wrong>\d+)\s+no_stop=(?P<nostop>\d+)\s+"
    r"proximity=(?P<prox>\d+)"
)
EP_DIAG = re.compile(
    r"\[GenArk\]\[ep-diag\].*?ep=(?P<ep>\S+)\s+cause=(?P<cause>\S+)\s+"
    r"success_type=(?P<stype>\S+)\s+steps=(?P<steps>\d+)\s+"
    r".*?final_dtg=(?P<dtg>[-0-9.]+)m"
)
METRIC = re.compile(
    r"train/terminal_score_mean=(?P<smean>[-0-9.e+]+).*?"
    r"train/terminal_score_std=(?P<sstd>[-0-9.e+]+).*?"
    r"train/in_goal_range_count=(?P<inrange>[-0-9.e+]+)|"
    r"actor/grad_norm=(?P<grad>[-0-9.e+]+).*?actor/approx_kl=(?P<kl>[-0-9.e+]+)|"
    r"actor/skipped_zero_advantage=(?P<skip>[-0-9.e+]+)"
)


def _strip(line: str) -> str:
    return ANSI.sub("", line)


def parse_train(log_path: Path) -> dict:
    groups = []
    pending = []
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for raw in text.splitlines():
        line = _strip(raw)
        ep = EP_DIAG.search(line)
        if ep:
            pending.append(ep.groupdict())
            continue
        g = GROUP.search(line)
        if g:
            members = pending[-8:] if len(pending) >= 8 else pending[:]
            dtgs = [float(m["dtg"]) for m in members]
            in_range = sum(d < 3.0 for d in dtgs)
            groups.append(
                {
                    "episode_id": g.group("ep"),
                    "clean_stop": int(g.group("k")),
                    "wrong_stop": int(g.group("wrong")),
                    "no_stop": int(g.group("nostop")),
                    "proximity": int(g.group("prox")),
                    "mean_dtg": sum(dtgs) / len(dtgs) if dtgs else None,
                    "in_goal_range": in_range,
                    "n": len(members),
                }
            )
            pending = []
    return {"n_groups": len(groups), "groups": groups}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_dir", type=Path)
    parser.add_argument("sft_eval_dir", type=Path)
    parser.add_argument("step6_eval_dir", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    train_log = args.train_dir / "train.log"
    train = parse_train(train_log) if train_log.is_file() else {"n_groups": 0, "groups": []}
    result = compare(
        _summarize(_load_trials(args.sft_eval_dir)),
        _summarize(_load_trials(args.step6_eval_dir)),
    )
    eval_md = render(result, args.sft_eval_dir, args.step6_eval_dir)
    lines = [
        "# Endpoint-Only Terminal GRPO Report",
        "",
        "Training objective: `S = -max(2, final Euclidean DTG)`. "
        "Clean STOP is logged only. STOP tokens are off the PPO mask.",
        "",
        f"- Train log: `{args.train_dir}`",
        f"- Frozen SFT eval: `{args.sft_eval_dir}`",
        f"- Frozen step-6 eval: `{args.step6_eval_dir}`",
        "",
        "## On-policy groups during training",
        "",
        "| step | episode | K clean | in-range (<3m) | mean DTG | proximity | wrong STOP |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for i, g in enumerate(train["groups"], start=1):
        mean_dtg = f"{g['mean_dtg']:.2f}" if g["mean_dtg"] is not None else "-"
        lines.append(
            f"| {i} | {g['episode_id']} | {g['clean_stop']}/{g['n']} | "
            f"{g['in_goal_range']}/{g['n']} | {mean_dtg} | {g['proximity']} | {g['wrong_stop']} |"
        )
    lines.extend(["", eval_md.replace("# Robostral Gate A.5 Frozen Behavioral Check", "## Frozen eval vs checkpoint-1200", 1)])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.report.with_suffix(".json").write_text(
        json.dumps({"train": train, "eval": result}, indent=2), encoding="utf-8"
    )
    print(args.report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
