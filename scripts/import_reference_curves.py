#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.performance import PerformanceSettings  # noqa: E402
from backend.reference_curves import (  # noqa: E402
    PerformanceCalculationService,
    PerformanceCurveModel,
    import_reference_curves,
)
from backend.store import Store  # noqa: E402


def main() -> int:
    defaults = PerformanceSettings.from_env()
    parser = argparse.ArgumentParser(
        description="Replace static turbine/gate/loss reference curves and recalculate all readings."
    )
    parser.add_argument("curve_path", type=Path, help="OEM JSON file or directory containing the three CSV files")
    parser.add_argument("--database", type=Path, default=ROOT / "data" / "hydrovision.sqlite3")
    parser.add_argument("--dataset-name", required=True, help="OEM document/revision identifier")
    parser.add_argument("--unit-id", default=defaults.unit_id)
    parser.add_argument("--nameplate-mw", type=float, default=defaults.nameplate_capacity_mw)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Mark this dataset as non-OEM demo data; it will be refused with RealSourceAdapter",
    )
    args = parser.parse_args()

    store = Store(args.database)
    import_reference_curves(
        store,
        args.curve_path,
        dataset_name=args.dataset_name,
        is_demo=args.demo,
    )
    service = PerformanceCalculationService(
        store,
        PerformanceCurveModel(store, args.unit_id),
        nameplate_capacity_mw=args.nameplate_mw,
    )
    summary = service.backfill(overwrite=True)
    print(
        f"Imported {args.dataset_name!r}; recalculated {summary['updated']} of "
        f"{summary['processed']} readings ({summary['failed']} failed)."
    )
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
