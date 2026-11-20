#!/usr/bin/env python3
"""Sample recorded stocking checkpoints with explicit outcome provenance."""

import argparse
import hashlib
import json
import random
from pathlib import Path

SKILLS = {
    "grasp-lift": {"grasp_lift"},
    "transfer-seat": {"transfer", "shelf_place"},
    "supported-release": {"front_release"},
    "side-registration": {"free_landing", "side_registration"},
    "insert": {"front_insertion", "front_squaring_complete", "front_square_deferred"},
    "push-to-contact": {"contact_finish", "contact_refinement"},
    "row-press": {"stocking_milestone"},
    "retreat": {"release_retreat", "withdrawal_planning"},
}
FAILURE_PHASES = {
    "grasp-lift": {
        "grasp_precheck",
        "grasp_candidate",
        "grasp_staging",
        "grasp_transit",
        "pregrasp",
        "approach",
        "close",
        "lift",
        "grasp_hold",
    },
    "transfer-seat": {
        "carry_clear",
        "carry_front_clear",
        "loaded_workspace_plan",
        "loaded_stage_plan",
        "align_shelf",
        "transfer",
        "place",
        "seat",
    },
    "supported-release": {
        "stage_align",
        "release_unload",
        "release_front",
        "release_front_separate",
        "staging_clear",
        "free_landing_rejected",
    },
    "side-registration": {
        "lateral_stage",
        "lateral_plan",
        "lateral_orient_high",
        "lateral_approach",
        "lateral_push",
        "lateral_unload",
        "lateral_clear",
    },
    "insert": {
        "front_square_plan",
        "front_square_clear",
        "pusher_plan",
        "pusher_base_plan",
        "pusher_approach",
        "pusher_preshape",
        "finger_push",
    },
    "push-to-contact": {"push", "finger_push", "contact_refine", "loaded_push_stall"},
    "row-press": {
        "tangential_stage",
        "tangential_plan",
        "tangential_orient_high",
        "tangential_approach",
        "front_tangential_push",
        "tangential_unload",
        "tangential_clear",
    },
    "retreat": {"release", "release_forecast", "unloaded_clear_forecast", "clear"},
}


def catalog(root, skill, *, terminal_only=False):
    for index in sorted(root.rglob("skill-entries/index.json")):
        episode = index.parent.parent
        entries = [
            json.loads(path.read_text()) for path in sorted(index.parent.glob("*/entry.json"))
        ]
        start = episode.parent / "input_ref.json"
        restored_entry = None
        if start.is_file():
            reference = json.loads(start.read_text())
            if reference.get("checkpoint"):
                checkpoint = Path(reference["checkpoint"])
                source = checkpoint.parent / "entry.json"
                manifest = checkpoint / "manifest.json"
                if (
                    source.is_file()
                    and manifest.is_file()
                    and (
                        hashlib.sha256(manifest.read_bytes()).hexdigest() == reference.get("sha256")
                    )
                ):
                    restored_entry = json.loads(source.read_text())
                    entries.insert(0, restored_entry)
        failure = {}
        for entry in entries:
            product = entry["active_product"]
            if entry["skill"] == "motion_failure":
                failure[product] = entry
            elif entry["skill"] == "stocking_milestone":
                failure.pop(product, None)
        for offset, entry in enumerate(entries):
            if terminal_only:
                if (
                    entry["skill"] != "motion_failure"
                    or entry["case"].get("failed_phase") not in FAILURE_PHASES[skill]
                ):
                    continue
            elif entry["skill"] not in SKILLS[skill]:
                continue
            product = entry["active_product"]
            terminal = entry if terminal_only else failure.get(product)
            outcome = "unverified"
            if terminal_only:
                outcome = "terminal_failure"
            elif terminal and terminal["sim_time_s"] >= entry["sim_time_s"]:
                outcome = "followed_by_failure"
            elif any(
                later["active_product"] == product
                and later["skill"] == "stocking_milestone"
                and later["sim_time_s"] >= entry["sim_time_s"]
                for later in entries
            ):
                outcome = "followed_by_milestone"
            failure_case = terminal["case"] if outcome.endswith("_failure") else {}
            checkpoint = episode / Path(entry["entry_file"]).parent / entry["checkpoint"]
            if entry is restored_entry:
                checkpoint = Path(reference["checkpoint"])
            manifest = checkpoint / "manifest.json"
            if manifest.is_file():
                yield {
                    "checkpoint": str(checkpoint),
                    "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    "entry_skill": entry["skill"],
                    "evidence_episode": str(episode),
                    "restored_entry": entry is restored_entry,
                    "active_product": product,
                    "reset_seed": entry["reset_seed"],
                    "sim_time_s": entry["sim_time_s"],
                    "completed_before": len(entry["completed_products"])
                    if "completed_products" in entry
                    else None,
                    "outcome": outcome,
                    "later_entries": list(
                        dict.fromkeys(
                            later["skill"]
                            for later in entries[offset + 1 :]
                            if later["active_product"] == product
                        )
                    ),
                    "failure": failure_case,
                    "failure_checkpoint": (
                        str(episode / Path(terminal["entry_file"]).parent / terminal["checkpoint"])
                        if failure_case
                        else None
                    ),
                }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("skill", choices=SKILLS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--failed", action="store_true")
    mode.add_argument("--terminal", action="store_true", help="List stopped recovery states")
    args = parser.parse_args()
    if args.count is not None and args.count < 1:
        parser.error("--count must be positive")
    cases = list(catalog(args.root, args.skill, terminal_only=args.terminal))
    if args.failed:
        cases = [
            case
            for case in cases
            if case["outcome"] == "followed_by_failure"
            and case["failure"].get("failed_phase") in FAILURE_PHASES[args.skill]
        ]
    if args.count is not None:
        randomizer = random.Random(args.seed)
        randomizer.shuffle(cases)
        # Retain every recorded outcome in the catalog; sample distinct inputs.
        by_manifest = {}
        for case in cases:
            by_manifest.setdefault(case["manifest_sha256"], case)
        cases = list(by_manifest.values())[: args.count]
    print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
