#!/usr/bin/env python3
"""Collect named, completed campaigns from terminal evidence into a compact report."""

import argparse
import json
from pathlib import Path

import numpy as np

from abstract_stack.benchmark_report import campaign_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", action="append", required=True, metavar="LABEL=MANIFEST")
    parser.add_argument("--search", type=Path)
    parser.add_argument("--development-probes", type=Path)
    parser.add_argument("--pair", action="append", default=[], metavar="BEFORE=AFTER")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = {}
    for specification in args.campaign:
        label, separator, path = specification.partition("=")
        if not separator or not label or label in reports:
            parser.error("Each campaign needs a unique LABEL=MANIFEST")
        report = campaign_report(path)
        if report["completed_scenes"] != report["scheduled_scenes"]:
            parser.error(f"Campaign {label} has unfinished scheduled scenes")
        completion_times = []
        for row in report["runs"]:
            # A native/process failure may leave a partial JSON file. Preserve
            # campaign_report's unknown row rather than reparsing it as evidence.
            if row["verification_status"] == "unknown":
                continue
            directory = Path(row["directory"])
            summary_path = directory / "summary.json"
            if not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text())
            row["final_product_poses"] = summary["final_product_poses"]
            row["stage_evaluation"] = summary["stage_evaluation"]
            row["interventions"] = summary["metrics"]["intervention_count"]
            manifest = json.loads((directory / "initial-scene/manifest.json").read_text())
            row["target_errors_xy_m"] = {
                product["name"]: (
                    np.asarray(summary["final_product_poses"][product["name"]][:2])
                    - np.asarray(product["target"][:2])
                ).tolist()
                for product in manifest["products"]
            }
            if row["clean_success"]:
                completion_times.append(row["sim_seconds"])
        report["mean_success_sim_seconds"] = (
            float(np.mean(completion_times)) if completion_times else None
        )
        reports[label] = report
    evidence = {"campaigns": reports}
    pairs = []
    for pair in args.pair:
        before, separator, after = pair.partition("=")
        if not separator or before not in reports or after not in reports:
            parser.error("Each pair must name two supplied campaign labels")
        first, second = reports[before]["runs"], reports[after]["runs"]
        if [r["seed"] for r in first] != [r["seed"] for r in second]:
            parser.error(f"Unmatched scheduled seeds in {pair}")
        checks = {}
        for key in ("initial_conditions_sha256", "settled_state_sha256"):
            checks[key] = all(
                a.get(key) and a.get(key) == b.get(key) for a, b in zip(first, second, strict=True)
            )
            if not checks[key]:
                parser.error(f"Missing or mismatched {key} in {pair}")
        pairs.append({"before": before, "after": after, "scenes": len(first), **checks})
    evidence["matched_pairs"] = pairs
    if args.development_probes:
        evidence["development_probes"] = json.loads(args.development_probes.read_text())
    if args.search:
        evidence["compliance_search"] = {
            "source": str(args.search),
            "record": json.loads(args.search.read_text()),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
