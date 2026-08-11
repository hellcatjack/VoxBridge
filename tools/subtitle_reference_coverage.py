#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from voxbridge.debug.subtitle_reference_coverage import analyze_reference_coverage


def _load_jsonl(path: Path) -> list[dict]:
    events = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                events.append(json.loads(line))
    return events


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare committed WebSocket subtitles with a YouTube json3 reference timeline.",
    )
    parser.add_argument("--reference-json3", required=True)
    parser.add_argument("--events-jsonl", required=True)
    parser.add_argument("--duration-sec", type=float, default=None)
    args = parser.parse_args()

    report = analyze_reference_coverage(
        Path(args.reference_json3),
        _load_jsonl(Path(args.events_jsonl)),
        duration_sec=args.duration_sec,
    )
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
