#!/usr/bin/env python3
"""Assemble causal option decisions and measured returns from curriculum agent runs."""

import argparse
import json
from pathlib import Path

import numpy as np

from abstract_stack.checkpoint_manifest import sha256_file
from abstract_stack.skill_curriculum import ACTIONS


def completed_attempts(run, report, unfinished):
    for case in report["cases"]:
        default = case.get("case_id", f"{case['skill']}-{case['seed']}")
        case_dir = (run / case.get("result", f"{default}/skill-result.json")).parent
        rewind = case_dir / "rewind"
        directories = (case_dir, rewind) if rewind.is_dir() else (case_dir,)
        for directory in directories:
            breadcrumb = dict(
                run=str(run),
                case=case.get("case_id"),
                directory=str(directory),
                outcome=case["outcome"],
            )
            ledger_path = directory / "attempts.json"
            if not ledger_path.is_file():
                unfinished.append(breadcrumb)
                continue
            ledger = json.loads(ledger_path.read_text())
            if ledger["next_action"] is not None:
                unfinished.append(breadcrumb)
            for attempt in ledger["attempts"]:
                yield directory / attempt["artifacts"], attempt, ledger["initial_state_sha256"]


def build(runs, output):
    rows, columns, unfinished = [], {}, []
    skills = tuple(ACTIONS)
    max_actions = max(map(len, ACTIONS.values()))
    for run in runs:
        report_path = run / "report.json"
        report = json.loads(report_path.read_text())
        for directory, attempt, state_digest in completed_attempts(run, report, unfinished):
            with np.load(directory / "decision.npz", allow_pickle=False) as before:
                decision = dict(before)
            with np.load(directory / "outcome.npz", allow_pickle=False) as after:
                outcome = dict(after)
            skill, action = attempt["skill"], attempt["action"]
            expected = ACTIONS[skill].index(action)
            if (
                int(decision["action"]) != expected
                or not decision["action_mask"][expected]
                or float(outcome["reward"]) != float(attempt["success"])
            ):
                raise ValueError("Recorded transition disagrees with the skill or measured gate")
            mask = decision.pop("action_mask")
            row = {
                **decision,
                **outcome,
                "skill": skills.index(skill),
                "action_mask": np.pad(mask, (0, max_actions - len(mask))),
            }
            for name, value in row.items():
                columns.setdefault(name, []).append(value)
            rows.append(
                dict(
                    source_id=report["source_id"],
                    run=str(run),
                    case_id=attempt["case_id"],
                    lineage=attempt["lineage"],
                    skill=skill,
                    action=action,
                    outcome=attempt["outcome"],
                    failure=attempt["failure"],
                    state_sha256=state_digest,
                    artifacts=str(directory),
                )
            )
    if not rows or any(len(values) != len(rows) for values in columns.values()):
        raise ValueError("Need complete, schema-compatible skill option transitions")
    output.mkdir(parents=True, exist_ok=False)
    arrays = output / "transitions.npz"
    np.savez_compressed(arrays, **{name: np.stack(values) for name, values in columns.items()})
    metadata = dict(
        schema_version=1,
        kind="privileged_skill_option_transitions",
        skills=skills,
        actions=ACTIONS,
        rows=rows,
        unfinished=unfinished,
        transitions_sha256=sha256_file(arrays),
        training_scope="one high-level action per reset; group training splits by lineage",
    )
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return dict(
        rows=len(rows),
        verified=sum(row["outcome"] == "verified" for row in rows),
        unfinished=len(unfinished),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.run, args.output)))


if __name__ == "__main__":
    main()
