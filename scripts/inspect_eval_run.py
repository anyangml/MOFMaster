#!/usr/bin/env python3
"""Inspect an analyzer_eval output directory.

Usage:
  uv run python scripts/inspect_eval_run.py data/evals/run_YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _last_ai_message(output: dict) -> str | None:
    msgs = output.get("messages") or []
    for msg in reversed(msgs):
        if isinstance(msg, dict) and msg.get("type") == "ai":
            content = msg.get("content")
            if isinstance(content, str):
                return content
    return None


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: inspect_eval_run.py <run_dir>")

    run_dir = Path(sys.argv[1])
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Missing {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    failures = []
    for item in summary:
        if item.get("plan") in (None, []) or item.get("approved") in (None, False):
            failures.append(item)

    print(f"Run: {run_dir}")
    print(f"Total cases: {len(summary)}")
    print(f"Failures/non-plans: {len(failures)}")

    for item in failures:
        raw_path = Path(item["raw"])
        print("\n==", item["case_id"], "==")
        print("title:", item.get("title"))
        print("plan:", item.get("plan"))
        print("approved:", item.get("approved"))
        print("expectation:", item.get("expectation"))
        print("raw:", raw_path)

        try:
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
        except Exception as e:
            print("raw_read_error:", str(e))
            continue

        if "error" in raw:
            print("server_error:", raw["error"])
            continue

        output = raw.get("output") or {}
        print("rejection_count:", output.get("_rejection_count"))

        last_ai = _last_ai_message(output)
        if last_ai:
            preview = last_ai if len(last_ai) <= 600 else last_ai[:600] + "…"
            print("last_ai_preview:", preview.replace("\n", "\\n"))
        else:
            print("last_ai_preview: <none>")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
