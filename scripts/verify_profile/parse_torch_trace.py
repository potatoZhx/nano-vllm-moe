#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def parse_trace(path: Path, top: int) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    events = data.get("traceEvents", [])
    by_cat: dict[str, float] = collections.defaultdict(float)
    by_name: dict[str, float] = collections.defaultdict(float)
    name_count: collections.Counter[str] = collections.Counter()
    cat_count: collections.Counter[str] = collections.Counter()

    for event in events:
        cat = str(event.get("cat", ""))
        name = str(event.get("name", ""))
        cat_count[cat] += 1
        dur_us = float(event.get("dur") or 0.0)
        if dur_us <= 0:
            continue
        by_cat[cat] += dur_us
        by_name[name] += dur_us
        name_count[name] += 1

    lines = [
        f"# Torch Trace Summary: `{path}`",
        "",
        f"- event_count: `{len(events)}`",
        "",
        "## Categories",
        "",
        "| category | count | duration ms |",
        "|---|---:|---:|",
    ]
    for cat, dur_us in sorted(by_cat.items(), key=lambda item: item[1], reverse=True)[:top]:
        lines.append(f"| `{cat}` | {cat_count[cat]} | {dur_us / 1000.0:.3f} |")

    lines.extend(
        [
            "",
            "## Names",
            "",
            "| name | count | duration ms |",
            "|---|---:|---:|",
        ]
    )
    for name, dur_us in sorted(by_name.items(), key=lambda item: item[1], reverse=True)[:top]:
        safe_name = name.replace("|", "\\|")
        lines.append(f"| `{safe_name}` | {name_count[name]} | {dur_us / 1000.0:.3f} |")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a PyTorch Chrome trace exported by verify profiling.")
    parser.add_argument("trace", type=Path)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    text = parse_trace(args.trace, args.top)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
