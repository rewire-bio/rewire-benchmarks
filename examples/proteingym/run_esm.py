"""Offline small public-model example. Requires locally prepared ProteinGym data."""
import argparse
from pathlib import Path

from rewirebench import prepare, run
from rewirebench.adapters.esm import ESM2Adapter
from rewirebench.protocols.proteingym import PROTOCOL_ID


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Extracted official DMS CSV directory")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Local pinned ESM-2 8M .pt")
    parser.add_argument("--assay", default="AMFR_HUMAN_Tsuboyama_2023_4G3O")
    parser.add_argument("--limit", default=10, type=int, help="Smoke rows; not a full assay score")
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    args = parser.parse_args()
    data = prepare(PROTOCOL_ID, source=args.data, output=args.output / "prepared",
                   assay_ids=[args.assay], limit=args.limit)
    adapter = ESM2Adapter(args.checkpoint)
    report = run(data, adapter, output=args.output / "evaluation",
                 model={"name": "ESM-2 8M masked marginal example",
                        "training_overlap": "unreported"})
    print(f"{report['scope']} run; see {args.output / 'evaluation' / 'report.json'}")
    print("This is not a complete ProteinGym evaluation or a reproduced published result.")


if __name__ == "__main__":
    main()
