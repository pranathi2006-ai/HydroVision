#!/usr/bin/env python3
"""Train, compare, and explicitly promote Phase 6 attribution models."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.learned_attribution import (  # noqa: E402
    LearnedAttributionService,
    LearnedAttributionSettings,
    LearnedAttributionTrainer,
)
from backend.store import Store  # noqa: E402


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.getenv("HYDROVISION_DATA_DIR", ROOT / "data")),
    )
    subcommands = command.add_subparsers(dest="command", required=True)
    subcommands.add_parser("status", help="Show feedback and current shadow/active versions.")
    subcommands.add_parser("train", help="Train a new version and save it in shadow status.")
    subcommands.add_parser("retrain-if-due", help="Run the same safe check used by the scheduler.")
    compare = subcommands.add_parser("compare", help="Compare a shadow version with its rule baseline.")
    compare.add_argument("model_id")
    promote = subcommands.add_parser("promote", help="Human-approved promotion after a clear shadow win.")
    promote.add_argument("model_id")
    promote.add_argument("--approved-by", required=True)
    promote.add_argument("--confirm", required=True, help="Must be exactly: PROMOTE <model_id>")
    promote.add_argument("--notes")
    return command


def main() -> int:
    args = parser().parse_args()
    store = Store(args.data_dir / "hydrovision.sqlite3")
    settings = LearnedAttributionSettings.from_env()
    trainer = LearnedAttributionTrainer(store, settings)
    service = LearnedAttributionService(store, settings)
    if args.command == "status":
        result = {
            "confirmed_outcomes": store.confirmed_attribution_count(),
            "shadow": store.loss_model("shadow"),
            "active": store.loss_model("active"),
            "scheduler_can_promote": False,
        }
    elif args.command == "train":
        result = trainer.train()
    elif args.command == "retrain-if-due":
        result = trainer.train_if_due()
    elif args.command == "compare":
        result = service.compare_shadow(args.model_id)
    else:
        result = service.promote(
            args.model_id,
            approved_by=args.approved_by,
            confirmation=args.confirm,
            approval_notes=args.notes,
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, PermissionError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
