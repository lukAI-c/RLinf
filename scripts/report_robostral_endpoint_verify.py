#!/usr/bin/env python3
"""Assemble held-out + repeat verification from frozen eval directories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_robostral_gate_a5 import (
    _load_trials,
    _summarize,
    compare,
    render,
)


def _avg(log_dir: Path) -> dict:
    path = log_dir / "avg_metrics.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _pair(title: str, base_dir: Path, other_dir: Path, episodes: tuple[str, ...]) -> str:
    result = compare(
        _summarize(_load_trials(base_dir), episodes=episodes),
        _summarize(_load_trials(other_dir), episodes=episodes),
    )
    body = render(result, base_dir, other_dir)
    avg_b, avg_o = _avg(base_dir), _avg(other_dir)
    extra = [
        "",
        "### avg_metrics",
        "",
        f"- nDTW: {avg_b.get('eval/ndtw', float('nan')):.4f} → {avg_o.get('eval/ndtw', float('nan')):.4f}",
        f"- path length: {avg_b.get('eval/path_length', float('nan')):.2f} → {avg_o.get('eval/path_length', float('nan')):.2f}",
        f"- SPL: {avg_b.get('eval/spl', float('nan')):.3f} → {avg_o.get('eval/spl', float('nan')):.3f}",
        f"- env success: {avg_b.get('eval/success', float('nan')):.3f} → {avg_o.get('eval/success', float('nan')):.3f}",
        "",
    ]
    return f"## {title}\n\n" + body.replace("# Robostral Gate A.5 Frozen Behavioral Check\n\n", "") + "\n".join(extra)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--step6", type=Path, required=True)
    parser.add_argument("--repeat1", type=Path, required=True)
    parser.add_argument("--repeat2", type=Path, required=True)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    episodes = tuple(item.strip() for item in args.episodes.split(",") if item.strip())
    parts = [
        "# Endpoint-only repeatability and held-out verification",
        "",
        "Held-out Z6 episodes plus anti-forgetting anchor 586. "
        "Training repeats exclude 586. Reward remains `S=-max(2, final DTG)`.",
        "",
        f"- Episodes: {', '.join(episodes)}",
        f"- SFT held-out: `{args.sft}`",
        f"- Existing endpoint step6 held-out: `{args.step6}`",
        f"- Repeat 1 held-out: `{args.repeat1}`",
        f"- Repeat 2 held-out: `{args.repeat2}`",
        "",
        _pair("Existing endpoint step-6 vs SFT (held-out)", args.sft, args.step6, episodes),
        _pair("Repeat 1 vs SFT (held-out)", args.sft, args.repeat1, episodes),
        _pair("Repeat 2 vs SFT (held-out)", args.sft, args.repeat2, episodes),
    ]
    text = "\n".join(parts) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
