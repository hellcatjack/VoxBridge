#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


TEXT_POOL_RE = re.compile(r"text_pool\s+(\{.*\})\s*$")


def _parse_text_pool_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = TEXT_POOL_RE.search(line.strip())
            if not m:
                continue
            raw = m.group(1)
            try:
                row = json.loads(raw)
            except Exception:
                continue
            if str(row.get("topic", "")) != "text_pool":
                continue
            rows.append(row)
    return rows


def _group_rows(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, int], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        ws_id = str(row.get("ws_id", "unknown"))
        segment_id = int(row.get("segment_id", 0) or 0)
        grouped[(ws_id, segment_id)].append(row)
    return grouped


def _summarize(grouped: Dict[Tuple[str, int], List[Dict[str, Any]]]) -> str:
    lines: List[str] = []
    lines.append(f"groups={len(grouped)}")
    for (ws_id, segment_id), rows in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        rows_sorted = sorted(rows, key=lambda r: int(r.get("seq", 0) or 0))
        generating = sum(1 for r in rows_sorted if str(r.get("phase", "")) == "generating")
        solidified = sum(1 for r in rows_sorted if str(r.get("phase", "")) == "solidified")
        last_event = str(rows_sorted[-1].get("event", "")) if rows_sorted else ""
        lines.append(
            f"[{ws_id}] segment={segment_id} rows={len(rows_sorted)} generating={generating} "
            f"solidified={solidified} last_event={last_event}"
        )
        for row in rows_sorted[-5:]:
            lines.append(
                "  - "
                f"seq={int(row.get('seq', 0) or 0)} event={row.get('event', '')} "
                f"phase={row.get('phase', '')} chars={int(row.get('text_chars', 0) or 0)} "
                f"reason={row.get('reason', '')}"
            )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize text_pool trace events from backend log.")
    p.add_argument("--log", required=True, help="Path to backend log file")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log_path = Path(args.log).expanduser()
    rows = _parse_text_pool_rows(log_path)
    grouped = _group_rows(rows)
    print(_summarize(grouped))


if __name__ == "__main__":
    main()

