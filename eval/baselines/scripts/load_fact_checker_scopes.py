#!/usr/bin/env python3
"""Save the fact-check scopes of all IFCN signatories and EFCSN members, as the VeriTaS database has
them now, to a JSON file for run_benchmark.py --blocked-scopes. E.g.:
    python -m eval.baselines.scripts.load_fact_checker_scopes dev/q3_baselines/fact_checker_scopes.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from baselines.common.reviews import signatory_scopes  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path, help="The JSON file to write, a list of \"host\" or \"host/path\" scopes")
    args = parser.parse_args()

    scopes = signatory_scopes()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scopes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {len(scopes)} fact-check scopes to {args.output}")


if __name__ == "__main__":
    main()
