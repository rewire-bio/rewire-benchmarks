"""Two synthetic substitutions: inference plumbing only, with no fitness labels."""
import argparse
import json

from rewirebench.adapters.esm import ESM2Adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    adapter = ESM2Adapter(args.checkpoint)
    sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANL"
    predictions = adapter.predict([
        {"id": "synthetic::M1A", "wild_type_sequence": sequence, "mutant": "M1A"},
        {"id": "synthetic::K2A", "wild_type_sequence": sequence, "mutant": "K2A"},
    ])
    print(json.dumps({"kind": "synthetic_inference_smoke", "benchmark_result": False,
                      "predictions": predictions, "provenance": adapter.provenance}, indent=2))


if __name__ == "__main__":
    main()
