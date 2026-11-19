#!/usr/bin/env python3
"""Evaluate frozen teachers on declared DR cases, retaining every scheduled stage."""

import argparse
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def grade_result(result, names, reset_seed):
    """Require the scheduled objects, reset identity and strict stage booleans."""
    evaluation = result.get("stage_evaluation", {})
    evidence = evaluation.get("products", {})
    if (
        set(names) != set(result.get("scheduled_pick_products", []))
        or len(names) != len(result.get("scheduled_pick_products", []))
        or not set(names) <= set(evidence)
        or result.get("metrics", {}).get("seed") != reset_seed
    ):
        return dict(outcome="invalid_scheduled_object_evidence", clean_success=False)
    products = {
        name: {
            stage: evidence[name].get(stage) is True
            for stage in ("pick", "place", "final_position", "retreat")
        }
        for name in names
    }
    return dict(
        outcome="completed_teacher_episode",
        failure=result.get("failure"),
        products=products,
        clean_success=result.get("task_success") is True
        and result.get("physical_success") is True
        and evaluation.get("no_knockovers") is True
        and all(all(p.values()) for p in products.values()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "heldout"), default="development")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--case-wall-seconds", type=float, default=180)
    parser.add_argument("--policy-source-ref", type=Path)
    args = parser.parse_args()
    if not 1 <= args.limit <= 100 or not 0 < args.case_wall_seconds <= 600:
        parser.error("Bound cases and per-case wall time")
    if args.split == "heldout" and args.policy_source_ref is None:
        parser.error("Heldout evaluation requires an explicit previously frozen policy source ref")
    from abstract_stack.source_store import freeze_source, resolve_source_reference

    root = Path(__file__).resolve().parents[1]
    args.suite, args.output = args.suite.resolve(), args.output.resolve()
    index_path = args.suite / args.split / "index.json"
    index = json.loads(index_path.read_text())
    if index["split"] != args.split:
        raise ValueError("Suite index split mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = (
        resolve_source_reference(
            args.policy_source_ref, cache=root / "outputs/cache/source_runtime"
        )
        if args.policy_source_ref
        else freeze_source(
            root,
            paths=[
                "src/abstract_stack",
                "scripts/run_replanning.py",
                "scripts/run_frozen_replanning.py",
                "scripts/evaluate_domain_suite.py",
                "pyproject.toml",
                "uv.lock",
            ],
        )
    )
    frozen.write_reference(args.output / "source_ref.json")
    runtime = frozen.runtime_root
    supervise = runpy.run_path(str(runtime / "scripts/run_frozen_replanning.py"))["supervise"]
    write(
        args.output / "manifest.json",
        dict(
            suite_index=str(index_path),
            suite_index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
            source_content_id=frozen.content_id,
            evaluation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            policy_profile="stocking_cli_baseline_v1",
            git=frozen.git,
            case_wall_seconds=args.case_wall_seconds,
            scheduled_cases=len(index["cases"]),
            attempt_budget=args.limit,
        ),
    )
    rows = [
        dict(
            case_id=c["case_id"],
            outcome="not_run",
            process=None,
            scheduled_products=c["scheduled_product_names"],
            products={
                name: dict.fromkeys(("pick", "place", "final_position", "retreat"), False)
                for name in c["scheduled_product_names"]
            },
            clean_success=False,
        )
        for c in index["cases"]
    ]

    def report():
        write(
            args.output / "report.json",
            dict(
                kind="frozen_policy_dr_evaluation",
                split=args.split,
                controller_source_id=frozen.content_id,
                scheduled_scenes=len(rows),
                scheduled_objects=sum(len(r["scheduled_products"]) for r in rows),
                attempted_scenes=sum(r["outcome"] != "not_run" for r in rows),
                clean_scene_successes=sum(r["clean_success"] for r in rows),
                stage_totals={
                    stage: sum(p[stage] for r in rows for p in r["products"].values())
                    for stage in ("pick", "place", "final_position", "retreat")
                },
                evaluation_complete=all(r["outcome"] != "not_run" for r in rows),
                cases=rows,
                collection_scope="Diagnostic teacher evaluation; no automatic training eligibility",
            ),
        )

    report()
    for case, row in zip(index["cases"][: args.limit], rows, strict=False):
        directory = args.output / case["case_id"]
        directory.mkdir()
        inputs = {}
        for key in ("scene", "context"):
            path = (args.suite / case[key]).resolve()
            if (
                not path.is_relative_to(args.suite)
                or hashlib.sha256(path.read_bytes()).hexdigest() != case[f"{key}_sha256"]
            ):
                raise ValueError("DR case changed or escaped the suite")
            inputs[key] = path
        context = json.loads(inputs["context"].read_text())
        if context["split"] != args.split:
            raise ValueError("Case split differs from scheduled suite")
        if case["teacher_adapter"] is None:
            row.update(
                outcome="unsupported_teacher_domain",
                reason="Current teacher adapter does not cover this declared family",
            )
            report()
            continue
        if case["teacher_adapter"] == "receding_dense":
            command = [
                sys.executable,
                str(runtime / "scripts/run_replanning.py"),
                "--scene-config",
                str(inputs["scene"]),
                "--seed",
                str(context["reset_seed"]),
                "--output-dir",
                str(directory / "episode"),
                "--loaded-route",
                "--admit-loaded-route",
                "--table-front",
                "--base-reposition",
                "--finger-load-compensation",
                "--bounded-contact-ik",
                "--contact-recovery",
                "--payload-load-compensation",
                "--square-backstop",
                "--split-insertion",
            ]
        else:
            command = [
                sys.executable,
                "-m",
                "abstract_stack.cli",
                "script",
                "--config",
                str(inputs["scene"]),
                "--seed",
                str(context["reset_seed"]),
                "--output-dir",
                str(directory / "episode"),
                "--guarded-approach",
            ]
        process = supervise(
            command,
            cwd=root,
            environment=dict(
                os.environ,
                PYTHONPATH=str(runtime / "src"),
                PYTHONDONTWRITEBYTECODE="1",
                PYTHONFAULTHANDLER="1",
                ABSTRACT_STACK_SOURCE_REFERENCE=str(args.output / "source_ref.json"),
                ABSTRACT_STACK_SOURCE_STORE=str(frozen.archive.parent),
            ),
            log=directory / "run.log",
            wall_seconds=args.case_wall_seconds,
        )
        row.update(process=process, outcome="process_failure_or_unknown")
        write(directory / "process.json", process)
        report()
        summary = directory / "episode/summary.json"
        if process["returncode"] in (0, 2) and not process["timeout"] and summary.exists():
            try:
                row.update(
                    grade_result(
                        json.loads(summary.read_text()),
                        row["scheduled_products"],
                        context["reset_seed"],
                    )
                )
                row["clean_success"] = row["clean_success"] and process["returncode"] == 0
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                row.update(outcome="invalid_summary", clean_success=False)
        report()
    print(
        json.dumps(
            dict(
                output=str(args.output),
                scheduled_scenes=len(rows),
                attempted_scenes=sum(r["outcome"] != "not_run" for r in rows),
                clean_scene_successes=sum(r["clean_success"] for r in rows),
                heldout=args.split == "heldout",
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
