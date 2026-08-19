#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the Phase 1 continuous-reading exit criterion.")
    parser.add_argument("--database", type=Path, default=Path("data/hydrovision.sqlite3"))
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--hours", type=float, default=24)
    args = parser.parse_args()

    with sqlite3.connect(args.database) as db:
        rows = db.execute(
            "SELECT ts FROM performance_reading ORDER BY ts ASC, reading_id ASC"
        ).fetchall()
    timestamps = [datetime.fromisoformat(row[0]) for row in rows]
    if len(timestamps) < 2:
        print("FAIL: fewer than two performance readings are stored")
        return 1

    coverage_seconds = (timestamps[-1] - timestamps[0]).total_seconds()
    allowed_gap = 2 * args.interval_seconds
    gaps = [
        (left, right, (right - left).total_seconds())
        for left, right in zip(timestamps, timestamps[1:])
        if (right - left).total_seconds() > allowed_gap
    ]
    if coverage_seconds < args.hours * 3600:
        print(f"FAIL: coverage is {coverage_seconds / 3600:.2f}h; need {args.hours:.2f}h")
        return 1
    if gaps:
        for left, right, seconds in gaps:
            print(f"FAIL: unexpected {seconds:.0f}s gap from {left.isoformat()} to {right.isoformat()}")
        return 1

    print(
        f"PASS: {len(timestamps)} readings cover {coverage_seconds / 3600:.2f}h "
        f"with no gap over {allowed_gap}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
