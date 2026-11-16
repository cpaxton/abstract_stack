#!/usr/bin/env python3
"""Extract small, source-linked development summaries without dropping failed runs."""

import argparse
import json
from pathlib import Path


def summarize(directory):
    row = {"run": str(directory), "verification_status": "unknown", "physical_success": None}
    try:
        manifest = json.loads((directory / "initial-scene/manifest.json").read_text())
        row.update(
            seed=manifest["seed"],
            product_count=len(manifest["products"]),
            robot_revision=manifest["robot_source"]["revision"],
            versions=manifest["versions"],
        )
        summary = json.loads((directory / "summary.json").read_text())
        row.update(
            verification_status="verified",
            physical_success=summary["physical_success"],
            failure=summary["failure"],
            seed=manifest["seed"],
            product_count=summary["metrics"]["product_count"],
            stocked_count=summary["metrics"]["stocked_count"],
            elapsed_sim_time=summary["metrics"]["elapsed_sim_time"],
            interventions=summary["metrics"]["intervention_count"],
            peak_forces_N=summary.get("peak_forces_N"),
            robot_revision=manifest["robot_source"]["revision"],
            versions=manifest["versions"],
        )
        row["physical_gates"] = {
            p["name"]: {
                "lift": any(
                    e.get("verified_lift") for e in summary["events"] if e["product"] == p["name"]
                ),
                "back_contact": any(
                    e.get("back_contact_verified")
                    for e in summary["events"]
                    if e["product"] == p["name"]
                ),
                "release": any(
                    e.get("verified_release")
                    for e in summary["events"]
                    if e["product"] == p["name"]
                ),
            }
            for p in manifest["products"]
        }
        if "video" in summary:
            row["video"] = {k: summary["video"][k] for k in ("fps", "encoded_frames", "cameras")}
    except (OSError, ValueError, KeyError) as error:
        row["artifact_error"] = str(error)
        failure = directory / "process_failure.json"
        if failure.exists():
            row["process_failure"] = json.loads(failure.read_text())
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "scope": "Selected development artifacts, not held-out evaluation or full tuning history",
        "runs": [summarize(path) for path in args.runs],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
