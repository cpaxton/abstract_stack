"""Audit legacy diagnostics without fabricating causal training pairs."""

import hashlib
import json
from pathlib import Path

import numpy as np


def audit_legacy_bundle(directory):
    directory = Path(directory)
    report = dict(
        kind="legacy_diagnostics_not_bc_dataset",
        source=str(directory.resolve()),
        training_ready=False,
        fresh_task_credit=0,
        verification_status="unknown",
        verified_task_success=None,
        reported_task_success=None,
        trace_rows=0,
        missing_for_bc=[
            "control-rate pre-action proprioception",
            "per-decision nominal teacher reference sequence",
            "raw timestamped sensor observations",
            "explicit native action names, modes and units",
            "pre-action / applied / next-observation alignment",
        ],
        sources_sha256={},
    )
    summary_path = directory / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        report["reported_task_success"] = summary.get("task_success")
        report["reported_failure"] = summary.get("failure")
        process_path = directory.parent / "process.json"
        if process_path.is_file():
            process = json.loads(process_path.read_text())
            success = summary.get("task_success")
            if (
                type(success) is bool
                and type(process.get("returncode")) is int
                and process.get("returncode") in (0, 2)
                and process.get("timeout") is False
                and success == (process["returncode"] == 0)
            ):
                report.update(
                    verification_status="terminal_and_process_agree", verified_task_success=success
                )
    trace_path = directory / "trace.npz"
    if trace_path.is_file():
        with np.load(trace_path, allow_pickle=False) as trace:
            samples = trace["samples"]
            controls = trace["controls"]
            report.update(
                trace_rows=len(samples),
                applied_control_shape=list(controls.shape),
                observation_timing="post-physics-step",
                action_semantics="applied compensated servo controls",
                nominal_action_sequence_available=False,
            )
            if len(samples) > 1:
                report["median_physics_dt_s"] = float(np.median(np.diff(samples[:, 0])))
    for name in ("summary.json", "trace.npz", "rollout_start.npz", "rollout_terminal.npz"):
        path = directory / name
        if path.is_file():
            report["sources_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return report


def import_legacy_for_bc(directory):
    audit = audit_legacy_bundle(directory)
    raise ValueError(
        "Legacy bundle is diagnostic-only; missing " + ", ".join(audit["missing_for_bc"])
    )
