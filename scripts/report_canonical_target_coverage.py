#!/usr/bin/env python3
"""Report frequency/type coverage of the reviewed canonical target aliases."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from rlinf.models.embodiment.qwen_nav.canonical_targets import normalize_target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="log root containing decision.json files")
    args = parser.parse_args()

    counts: Counter[str] = Counter()
    for path in args.root.rglob("decision.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        raw = payload.get("raw_target", payload.get("target", ""))
        if isinstance(raw, str) and raw.strip():
            counts[" ".join(raw.strip().lower().split())] += 1

    total = sum(counts.values())
    covered_rows = 0
    covered_types = 0
    for raw, count in counts.items():
        canonical, _status = normalize_target(raw)
        if canonical is not None:
            covered_rows += count
            covered_types += 1

    print(json.dumps({
        "rows": total,
        "unique_raw_types": len(counts),
        "frequency_coverage": covered_rows / total if total else 0.0,
        "type_coverage": covered_types / len(counts) if counts else 0.0,
        "uncovered_by_frequency": [
            {"target": raw, "count": count}
            for raw, count in counts.most_common()
            if normalize_target(raw)[0] is None
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

