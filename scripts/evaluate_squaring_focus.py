#!/usr/bin/env python3
"""Bounded, frozen-source left/right neighbor squaring development suite."""

import argparse
import hashlib
import json
import math
import os
import runpy
import sys
from pathlib import Path


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=root / "configs/squaring_focus.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    cases = config["cases"]
    if not (
        1 <= len(cases) <= 8
        and 1 <= config["replans"] <= 40
        and 0 < config["case_wall_seconds"] <= 90
        and 0 < config["normal_target_N"] < 25
        and len({c["name"] for c in cases}) == len(cases)
        and all(
            c["name"].replace("-", "").isalnum()
            and c["side"] in ("left", "right")
            and 1 <= c["candidates"] <= 16
            and 0 <= c["wrist_freedom_deg"] <= 20
            for c in cases
        )
    ):
        parser.error("Keep case names, settings and compute budgets bounded")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    from abstract_stack.source_store import freeze_source

    source = freeze_source(
        root,
        paths=[
            "src/abstract_stack",
            "scripts/evaluate_squaring_focus.py",
            "scripts/probe_compliant_lookahead.py",
            "scripts/run_frozen_replanning.py",
            "configs/squaring_focus.json",
            "uv.lock",
            "pyproject.toml",
        ],
    )
    source.write_reference(args.output / "source_ref.json")
    frozen = source.runtime_root
    write(args.output / "config.json", config)
    write(
        args.output / "manifest.json",
        dict(
            source_revision=source.git["revision"],
            source_content_id=source.content_id,
            git=source.git,
            config=config,
            checkpoint=str(args.checkpoint.resolve()),
            fresh_task_credit=0,
            source_sha256={name: record["sha256"] for name, record in source.files.items()},
            input_sha256={
                name: hashlib.sha256((args.output / name).read_bytes()).hexdigest()
                for name in ("source_ref.json", "config.json")
            },
            limits="Unchanged native model, 25 N force guards, 0.15 degree source yaw gate",
        ),
    )
    supervise = runpy.run_path(str(frozen / "scripts/run_frozen_replanning.py"))["supervise"]
    rows = []
    for case in cases:
        directory = args.output / case["name"]
        command = [
            sys.executable,
            str(frozen / "scripts/probe_compliant_lookahead.py"),
            "--checkpoint",
            str(args.checkpoint.resolve()),
            "--output",
            str(directory),
            "--reset-neighbor",
            case["side"],
            "--sweep-side",
            case["side"],
            "--normal-target",
            str(config["normal_target_N"]),
            "--wrist-freedom-deg",
            str(case["wrist_freedom_deg"]),
            "--candidates",
            str(case["candidates"]),
            "--shortlist",
            str(min(4, case["candidates"])),
            "--horizon",
            "0.2",
            "--execute-prefix",
            "0.1",
            "--require-dwell",
            "--save-checkpoint",
            "--replans",
            str(config["replans"]),
            "--wall-seconds",
            str(config["case_wall_seconds"]),
        ]
        result = supervise(
            command,
            cwd=root,
            environment=dict(
                os.environ,
                PYTHONPATH=str(frozen / "src"),
                PYTHONFAULTHANDLER="1",
                PYTHONDONTWRITEBYTECODE="1",
                ABSTRACT_STACK_SOURCE_STORE=str(source.archive.parent),
                ABSTRACT_STACK_SOURCE_REFERENCE=str(args.output / "source_ref.json"),
            ),
            log=args.output / f"{case['name']}.log",
            wall_seconds=config["case_wall_seconds"] + 5,
        )
        row = dict(
            case=case["name"],
            settings=case,
            outcome="unknown",
            process=result,
            task_success=False,
            fresh_task_credit=0,
            pick_credit=0,
        )
        if result["returncode"] == 0 and (directory / "summary.json").exists():
            summary = json.loads((directory / "summary.json").read_text())
            continuation = summary["continuation"]
            rotation = continuation["active_rotation_world"]
            row.update(
                outcome="completed_fixture_probe",
                final_yaw_deg=math.degrees(math.atan2(rotation[1][0], rotation[0][0])),
                failure=continuation["failure"],
                displacement_xyz_m=continuation["displacement_xyz_m"],
                simulated_seconds=continuation["simulated_seconds"],
                peak_forces_N=continuation["peak_forces_N"],
                incidents=continuation["incidents"],
                squared_dwell_verified=continuation["backstop_dwell"]["verified"],
                reset_fixture=summary["reset_fixture"],
            )
        rows.append(row)
        write(
            args.output / "report.json",
            dict(
                kind="squaring_development_fixtures",
                scheduled_cases=len(cases),
                results=rows,
                fresh_task_credit=0,
                task_success=False,
            ),
        )
    lines = [
        "# Environmental squaring development",
        "",
        "Reset fixtures: one dynamic neighbor, eight cartons on table, active already on shelf.",
        "Zero pick/fresh episode credit. All scheduled processes retained, including unknowns.",
        "Peaks start at fixture reset and include the captured instantaneous load; "
        "prior history is separate.",
        "",
        "| Case | Outcome | Final yaw (deg) | Pair peak (N) | Squared dwell |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {row['outcome']} | {row.get('final_yaw_deg', 'unknown')} | "
            f"{row.get('peak_forces_N', {}).get('carton_pair_N', 'unknown')} | "
            f"{row.get('squared_dwell_verified', False)} |"
        )
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    return 0 if all(r["outcome"] != "unknown" for r in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
