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
    parser.add_argument("--nameplate-mw", type=float, default=75)
    args = parser.parse_args()

    with sqlite3.connect(args.database) as db:
        rows = db.execute(
            """
            SELECT ts, theoretical_mw, gap_pct
            FROM performance_reading
            ORDER BY ts ASC, reading_id ASC
            """
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
    missing_phase_two = sum(row[1] is None or row[2] is None for row in rows)
    if missing_phase_two:
        print(f"FAIL: {missing_phase_two} readings are missing theoretical_mw or gap_pct")
        return 1
    implausible = [row[1] for row in rows if not 0 <= row[1] <= args.nameplate_mw]
    if implausible:
        print(
            f"FAIL: {len(implausible)} theoretical values are outside "
            f"0..{args.nameplate_mw} MW"
        )
        return 1

    print(
        f"PASS: {len(timestamps)} readings cover {coverage_seconds / 3600:.2f}h "
        f"with complete Phase 2 values, no gap over {allowed_gap}s, and all "
        f"theoretical output within 0..{args.nameplate_mw} MW"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
