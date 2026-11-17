#!/usr/bin/env python3
"""Write a paired stage/scene report from preserved terminal run evidence."""

import argparse
from pathlib import Path

from abstract_stack.benchmark_report import write_comparison

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--baseline", type=Path, required=True)
parser.add_argument("--tuned", type=Path, required=True)
parser.add_argument("--output", type=Path, default=Path("outputs/stocking-report"))
parser.add_argument("--search", type=Path, help="Hill-climb search.json with settings/budget")
args = parser.parse_args()
result = write_comparison(args.baseline, args.tuned, args.output, search=args.search)
for label in ("baseline", "tuned"):
    report = result[label]
    print(
        f"{label}: {report['clean_successes']}/{report['scheduled_scenes']} clean scenes; "
        f"stage counts {report['stages']}/{report['scheduled_objects']} objects"
    )
