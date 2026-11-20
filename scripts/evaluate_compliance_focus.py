#!/usr/bin/env python3
"""Compare held targets and compliant prefixes on a fixed development checkpoint bank."""

import argparse
import hashlib
import json
import os
import runpy
import shutil
import sys
from pathlib import Path


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replans", type=int, default=2)
    parser.add_argument("--case-wall-seconds", type=float, default=30)
    args = parser.parse_args()
    if not 1 <= args.replans <= 5 or not 0 < args.case_wall_seconds <= 60:
        parser.error("Keep each development comparison to 1–5 prefixes and <=60 seconds")
    root = Path(__file__).resolve().parents[1]
    args.bank, args.output = args.bank.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    bank = json.loads((args.bank / "bank.json").read_text())
    if bank["failures"] or len(bank["frames"]) != bank["scheduled_frames"]:
        parser.error("Evaluate a complete source-replayed focus bank")
    frozen = args.output / "frozen"
    shutil.copytree(
        root / "src", frozen / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    (frozen / "scripts").mkdir()
    for name in ("probe_compliant_lookahead.py", "run_frozen_replanning.py"):
        shutil.copy2(root / "scripts" / name, frozen / "scripts" / name)
    shutil.copy2(root / "uv.lock", frozen / "uv.lock")
    write(
        args.output / "manifest.json",
        dict(
            bank=str(args.bank),
            bank_sha256=hashlib.sha256((args.bank / "bank.json").read_bytes()).hexdigest(),
            source_sha256={
                p.relative_to(frozen).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in frozen.rglob("*")
                if p.is_file()
            },
            scheduled_cases=2 * len(bank["frames"]),
            replans=args.replans,
            case_wall_seconds=args.case_wall_seconds,
            fresh_task_credit=0,
            force_peak_scope="cumulative source-checkpoint history plus the executed prefix",
        ),
    )
    supervise = runpy.run_path(str(frozen / "scripts/run_frozen_replanning.py"))["supervise"]
    rows = []
    for frame in bank["frames"]:
        for mode, candidates, shortlist in (("held", 1, 1), ("compliant", 8, 4)):
            directory = args.output / f"{frame['name']}-{mode}"
            command = [
                sys.executable,
                str(frozen / "scripts/probe_compliant_lookahead.py"),
                "--checkpoint",
                str(args.bank / frame["checkpoint"]),
                "--output",
                str(directory),
                "--candidates",
                str(candidates),
                "--shortlist",
                str(shortlist),
                "--horizon",
                "0.4",
                "--execute-prefix",
                "0.1",
                "--require-dwell",
                "--square-feedback",
                "--replans",
                str(args.replans),
                "--wall-seconds",
                str(args.case_wall_seconds),
            ]
            result = supervise(
                command,
                cwd=root,
                environment=dict(
                    os.environ, PYTHONPATH=str(frozen / "src"), PYTHONFAULTHANDLER="1"
                ),
                log=args.output / f"{frame['name']}-{mode}.log",
                wall_seconds=args.case_wall_seconds + 5,
            )
            row = dict(
                frame=frame["name"],
                mode=mode,
                outcome="unknown",
                process=result,
                fresh_task_credit=0,
                task_success=False,
            )
            summary = directory / "summary.json"
            if result["returncode"] == 0 and summary.exists():
                data = json.loads(summary.read_text())["continuation"]
                rotation = data["active_rotation_world"]
                # Standard-library atan2 keeps the campaign supervisor native-free.
                import math

                yaw = math.degrees(math.atan2(rotation[1][0], rotation[0][0]))
                row.update(
                    outcome="completed_continuation",
                    simulated_seconds=data["simulated_seconds"],
                    advance_m=data["progress_minus_x_m"],
                    final_yaw_degrees=yaw,
                    absolute_yaw_reduction_degrees=abs(frame["yaw_degrees"]) - abs(yaw),
                    peak_forces_N=data["peak_forces_N"],
                    incidents=data["incidents"],
                    controller_failure=data["failure"],
                    squared_dwell_verified=data["backstop_dwell"]["verified"],
                )
            rows.append(row)
            write(
                args.output / "report.json",
                dict(
                    kind="fixed_development_checkpoint_comparison",
                    scheduled_cases=2 * len(bank["frames"]),
                    results=rows,
                    fresh_task_credit=0,
                    task_success=False,
                ),
            )
    text = [
        "# Fixed development checkpoint comparison",
        "",
        "Zero fresh task credit; these are restored failure states, not new stocking episodes.",
        "Force peaks include the inherited checkpoint history; they are not prefix-only peaks.",
        "",
        "| Frame | Mode | Process | Yaw reduction (deg) | "
        "Cumulative robot/product peak (N) | Squared dwell |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        delta = row.get("absolute_yaw_reduction_degrees")
        load = row.get("peak_forces_N", {}).get("robot_product_N")
        text.append(
            f"| {row['frame']} | {row['mode']} | {row['outcome']} | "
            f"{delta if delta is not None else 'unknown'} | "
            f"{load if load is not None else 'unknown'} | "
            f"{row.get('squared_dwell_verified', False)} |"
        )
    (args.output / "report.md").write_text("\n".join(text) + "\n")
    return 0 if all(r["outcome"] != "unknown" for r in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
