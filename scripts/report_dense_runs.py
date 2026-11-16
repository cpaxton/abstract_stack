#!/usr/bin/env python3
"""Summarize retained dense development trials without inventing process outcomes."""

import argparse
import hashlib
import json
from pathlib import Path

from abstract_stack.campaign import summarize_run
from abstract_stack.evaluation import reported_manipulations


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_object(path):
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def collect(directory):
    process = directory / "process_result.json"
    outcome = json.loads(process.read_text()) if process.exists() else {}
    run = summarize_run(directory, outcome.get("returncode"), video=False)
    valid = run["artifacts_valid"] and run["returncode"] in (0, 2)
    summary_path = directory / "summary.json"
    summary = read_object(summary_path)
    manifest_path = directory / "initial-scene/manifest.json"
    manifest = read_object(manifest_path)
    config = manifest.get("config") or {}
    grade = reported_manipulations(run)
    stages = grade["stages"] if valid else dict.fromkeys(grade["stages"], 0)
    return {
        "directory": str(directory),
        "returncode": run["returncode"],
        "verified_terminal_outcome": valid,
        "clean_success": valid and grade["clean_success"],
        "scheduled_manipulations": (
            len(grade["names"])
            if "scheduled_pick_products" in run
            else config.get("product_count", run.get("product_count", 0))
            - config.get("initially_stocked", 0)
        ),
        "stages": stages,
        "failure": run.get("failure"),
        "artifact_error": run.get("artifact_error"),
        "sim_seconds": run.get("elapsed_sim_time"),
        "incidents": (run.get("stage_evaluation") or {}).get("incidents"),
        "config": manifest.get("config"),
        "policy_source": manifest.get("policy_source"),
        "robot_source": manifest.get("robot_source"),
        "versions": manifest.get("versions"),
        "controller": summary.get("controller"),
        "tracking_config": summary.get("tracking_config"),
        "contact_protocol": summary.get("contact_protocol"),
        "packing": summary.get("metrics", {}).get("packing"),
        "peak_forces_N": summary.get("peak_forces_N"),
        "recorded_carton_contact_peak_N": summary.get("controller_evidence", {}).get(
            "peak_carton_carton_normal_N"
        ),
        "wrist_compensation": summary.get("controller_evidence", {}).get("wrist_load_feedforward"),
        "artifact_sha256": {
            name: digest(directory / name)
            for name in ("summary.json", "rollout_start.npz", "rollout_terminal.npz", "trace.npz")
            if (directory / name).exists()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True, help="Markdown/JSON output prefix")
    args = parser.parse_args()
    rows = [collect(path) for path in args.directories]
    report = {
        "kind": "development_trials_with_different_policies_and_initial_conditions",
        "generalization_success_rate": None,
        "definitions": {
            "free_space": "total unused lateral row width, including both wall gaps",
            "scheduled": "unprefilled cartons only; prefilled cartons remain monitored",
            "unknown": "missing verified process exit earns zero successful stages",
            "overlap": "projected envelopes: <=0.2 mm individual AND total negative gap",
            "scope": "native actuator/contact physics; privileged world state; no RL",
            "historical_forces": (
                "early development trials used per-contact-point carton peaks; "
                "current trials sum per carton pair; source fingerprints identify each variant"
            ),
        },
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Dense insertion development trials",
        "",
        "These are successive development trials with different policies or initial conditions. "
        "They are not a frozen-policy multi-scene success-rate study. Full videos and traces "
        "remain under ignored `outputs/`; the companion JSON pins sources and artifacts.",
        "",
        "Each row has ten cartons. Prefilled cartons earn no pick/place/retreat credit. "
        "All products are monitored at each physics step for transient tips and drops. "
        "A successful row requires all scheduled stages and the whole-shelf packing dwell.",
        "",
        "| Trial | Total free mm | Prefilled | Clean | Pick/place/final/retreat | Outcome |",
        "|---|---:|---:|---|---|---|",
    ]
    for row in rows:
        config = row["config"] or {}
        stages = "/".join(str(value) for value in row["stages"].values())
        outcome = row["failure"] or ("complete" if row["clean_success"] else "unverified")
        free = 1000 * config.get("row_free_space", 0)
        clean = "yes" if row["clean_success"] else "no"
        lines.append(
            f"| {Path(row['directory']).name} | {free:g} | "
            f"{config.get('initially_stocked', '?')} | {clean} | "
            f"{stages} / {row['scheduled_manipulations']} scheduled | {outcome} |"
        )
    lines += ["", "See [the dense task guide](../dense_stocking.md) for reproduction and limits."]
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
