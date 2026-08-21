#!/usr/bin/env python3
"""Run or inspect matched-condition work-order attribution verification."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.store import Store  # noqa: E402
from backend.verification import AttributionVerificationService, VerificationSettings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(os.getenv("HYDROVISION_DATA_DIR", ROOT / "data")),
    )
    parser.add_argument("command", choices=("run-due", "monitor", "backtest"))
    args = parser.parse_args()
    service = AttributionVerificationService(
        Store(args.data_dir / "hydrovision.sqlite3"), VerificationSettings.from_env(),
    )
    if args.command == "run-due":
        result = service.run_due()
    elif args.command == "monitor":
        result = service.monitoring()
    else:
        result = service.backtest_manual_history()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
