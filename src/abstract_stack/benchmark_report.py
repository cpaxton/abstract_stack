"""Reconstruct a benchmark report from terminal evidence, retaining all failures."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from abstract_stack.campaign import summarize_run

STAGES = ("pick", "place", "final_position", "retreat")


def campaign_report(path):
    path = Path(path)
    campaign = json.loads(path.read_text())
    rows = []
    controllers, provenance = [], {}
    totals = dict.fromkeys(STAGES, 0)
    scheduled = len(campaign["seeds"]) * campaign["objects"]
    clean, physical, no_knock = 0, 0, 0
    for entry in campaign["runs"]:
        run = summarize_run(entry["directory"], entry["returncode"], video=campaign["video"])
        evaluation = run.get("stage_evaluation") or {}
        counts = {k: 0 for k in STAGES}
        valid = run["artifacts_valid"] and run["returncode"] in (0, 2)
        if valid:
            for product in evaluation.get("products", {}).values():
                for k in STAGES:
                    counts[k] += int(product[k])
        physical += int(valid and run["physical_success"] is True)
        clean_run = (
            valid
            and run["physical_success"] is True
            and evaluation.get("no_knockovers") is True
            and all(counts[k] == campaign["objects"] for k in STAGES)
        )
        clean += int(clean_run)
        no_knock += int(valid and evaluation.get("no_knockovers") is True)
        for k in STAGES:
            totals[k] += counts[k]
        rows.append(
            {
                "seed": entry["seed"],
                "returncode": run["returncode"],
                "verification_status": run["verification_status"],
                "process_failure": run.get("process_failure"),
                "clean_success": bool(clean_run),
                "physical_success": run["physical_success"],
                "stages": counts,
                "incidents": evaluation.get("incidents"),
                "sim_seconds": run.get("elapsed_sim_time"),
                "peak_forces_N": run.get("peak_forces_N"),
                "failure": run.get("failure"),
                "directory": run["directory"],
            }
        )
        bundle_path = Path(run["directory"]) / "initial-scene/manifest.json"
        if bundle_path.exists():
            bundle = json.loads(bundle_path.read_text())
            source = bundle.get("robot_source") or {}
            provenance = {
                "robot": bundle["robot"],
                "model_revision": source.get("revision"),
                "versions": bundle["versions"],
                "physics_timestep": bundle["config"]["timestep"],
                "frame_skip": bundle["config"]["frame_skip"],
            }
            rows[-1]["scene_config"] = bundle["config"]
            initial = {
                key: bundle[key]
                for key in ("config", "initial_qpos", "initial_ctrl", "products", "model")
            }
            initial["robot_revision"] = source.get("revision")
            initial["mujoco_version"] = bundle["versions"]["mujoco"]
            rows[-1]["initial_conditions_sha256"] = hashlib.sha256(
                json.dumps(initial, sort_keys=True).encode()
            ).hexdigest()
        rollout_start = Path(run["directory"]) / "rollout_start.npz"
        if rollout_start.exists():
            try:
                with np.load(rollout_start) as state:
                    if "state" in state and "state_spec" in state:
                        payload = str(int(state["state_spec"])).encode() + b"\0"
                        payload += state["state"].tobytes()
                        rows[-1]["settled_state_sha256"] = hashlib.sha256(payload).hexdigest()
            except (OSError, ValueError) as error:
                rows[-1]["start_state_error"] = str(error)
        if run["artifacts_valid"]:
            summary = json.loads((Path(run["directory"]) / "summary.json").read_text())
            if summary["controller"] not in controllers:
                controllers.append(summary["controller"])
            with np.load(Path(run["directory"]) / "trace.npz") as trace:
                samples = trace["samples"]
                if len(samples) > 1:
                    dt = float(np.median(np.diff(samples[:, 0])))
                    column = summary["trace_columns"].index("hand_normal_N")
                    rows[-1]["hand_normal_impulse_Ns"] = float(samples[:, column].sum() * dt)
    n = len(campaign["seeds"])
    z, p = 1.96, clean / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {
        "campaign": str(path),
        "policy_revision": campaign.get("policy_revision")
        or (campaign.get("policy_source") or {}).get("revision"),
        "policy_source": campaign.get("policy_source"),
        "program_switches": {
            k: campaign.get(k)
            for k in ("contact_aware", "guarded_approach", "fixture_avoidance", "thumb_preshape")
        },
        "control_program": campaign.get("control_program"),
        "controllers": controllers,
        "provenance": provenance,
        "control_overrides": campaign.get("control_overrides", {}),
        "force_profile": campaign.get("force_profile"),
        "force_profile_settings": campaign.get("force_profile_settings"),
        "episode_wall_timeout_seconds": campaign.get("episode_wall_timeout_seconds"),
        "scheduled_scenes": n,
        "completed_scenes": len(rows),
        "scheduled_objects": scheduled,
        "clean_successes": clean,
        "physical_successes": physical,
        "unknown_scenes": sum(r["verification_status"] == "unknown" for r in rows) + n - len(rows),
        "no_knockover_scenes": no_knock,
        "success_rate": clean / n,
        "wilson_95_interval": [center - radius, center + radius],
        "stages": totals,
        "runs": rows,
        "definitions": {
            "pick": "Verified physical lift, hand contact and table unloading",
            "place": "Measured shelf support before the supported slide",
            "final_position": (
                "Back-contact/upright dwell, then unloaded release satisfying "
                "unchanged task geometry/velocity/support conditions"
            ),
            "retreat": (
                "Release and clear completed; current and previously placed cartons "
                "stable for task dwell, no product tipped/dropped during retreat"
            ),
            "clean_success": (
                "Process exited normally, complete task, all four stages, no "
                "product tipped >45 degrees or dropped below 0.25m at any "
                "monitored physics step"
            ),
            "unreached": (
                "Counted as zero in the scheduled-object denominator; not silently omitted"
            ),
        },
    }


def write_comparison(baseline, tuned, output, *, search=None):
    before, after = campaign_report(baseline), campaign_report(tuned)
    if any(r["completed_scenes"] != r["scheduled_scenes"] for r in (before, after)):
        raise ValueError("Finish every scheduled scene before writing a paired comparison")
    if [r["seed"] for r in before["runs"]] != [r["seed"] for r in after["runs"]]:
        raise ValueError("Paired comparison requires the same ordered scene seeds")
    if before["scheduled_objects"] != after["scheduled_objects"]:
        raise ValueError("Paired comparison requires the same product count")
    unverified_pairs = []
    unverified_settled_pairs = []
    for b, a in zip(before["runs"], after["runs"], strict=True):
        if not b.get("initial_conditions_sha256") or not a.get("initial_conditions_sha256"):
            unverified_pairs.append(b["seed"])
        elif b["initial_conditions_sha256"] != a["initial_conditions_sha256"]:
            raise ValueError(f"Seed {b['seed']} has different scene/model/initial conditions")
        if not b.get("settled_state_sha256") or not a.get("settled_state_sha256"):
            unverified_settled_pairs.append(b["seed"])
        elif b["settled_state_sha256"] != a["settled_state_sha256"]:
            raise ValueError(f"Seed {b['seed']} has different settled integration state")
    result = {
        "baseline": before,
        "tuned": after,
        "unverified_initial_condition_pairs": unverified_pairs,
        "unverified_settled_state_pairs": unverified_settled_pairs,
    }
    if search is not None:
        result["search"] = json.loads(Path(search).read_text())
        audit = Path(search).with_name("search-audit.json")
        if audit.exists():
            result["search_audit"] = json.loads(audit.read_text())
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Scripted shelf-stocking evaluation",
        "",
        "Native MABEL MuJoCo physics, privileged world state, "
        f"{before['scheduled_objects'] // before['scheduled_scenes']} cartons per scene. "
        "No RL, live pose writes, attachments, external object forces or interventions.",
        "",
        "| Metric | Baseline | Selected controller |",
        "|---|---:|---:|",
    ]
    for label, key, denominator in [
        ("Complete scenes, all stages, no knock-overs", "clean_successes", "scheduled_scenes"),
        ("Geometric full-shelf success", "physical_successes", "scheduled_scenes"),
        ("Unknown outcomes (no stage credit)", "unknown_scenes", "scheduled_scenes"),
        ("Observed runs without recorded tip/drop", "no_knockover_scenes", "scheduled_scenes"),
    ]:
        lines.append(
            f"| {label} | {before[key]}/{before[denominator]} | {after[key]}/{after[denominator]} |"
        )
    for key in STAGES:
        lines.append(
            f"| {key.replace('_', ' ').capitalize()} | "
            f"{before['stages'][key]}/{before['scheduled_objects']} | "
            f"{after['stages'][key]}/{after['scheduled_objects']} |"
        )
    for report in (before, after):
        successes = [r["sim_seconds"] for r in report["runs"] if r["clean_success"]]
        report["mean_success_sim_seconds"] = float(np.mean(successes)) if successes else None
        peaks = [(r["peak_forces_N"] or {}).get("hand", 0) for r in report["runs"]]
        report["maximum_hand_normal_N"] = max(peaks, default=0)

    def seconds(value):
        return f"{value:.2f}" if value is not None else "n/a"

    lines.append(
        f"| Mean time of successful scenes (sim s) | "
        f"{seconds(before['mean_success_sim_seconds'])} | "
        f"{seconds(after['mean_success_sim_seconds'])} |"
    )
    lines.append(
        f"| Maximum measured hand-normal force (N) | "
        f"{before['maximum_hand_normal_N']:.2f} | {after['maximum_hand_normal_N']:.2f} |"
    )
    lines += [
        "",
        "All scheduled objects remain in each stage's denominator, including "
        "objects unreached after an earlier failure. Stage gates are measured events; "
        "complete-scene success additionally requires every gate and no transient knock-over.",
        "No-tip/drop counts cover observed execution only; an aborted prefix does not "
        "establish a completed retreat, recovery, or post-abort stability.",
        "",
        "| Seed | Baseline pick/place/final/retreat | Tuned stages | Baseline / tuned result |",
        "|---|---|---|---|",
    ]
    for b, a in zip(before["runs"], after["runs"], strict=True):

        def counts(run):
            return "/".join(str(run["stages"][k]) for k in STAGES)

        lines.append(
            f"| {b['seed']} | {counts(b)} | {counts(a)} | "
            f"{'pass' if b['clean_success'] else b['failure'] or 'stage gate failed'} / "
            f"{'pass' if a['clean_success'] else a['failure'] or 'stage gate failed'} |"
        )
    lines += ["", "Stage definitions:", ""]
    lines += [f"- **{k.replace('_', ' ')}:** {v}." for k, v in before["definitions"].items()]
    lines += [
        "",
        "Selected explicit controller overrides: "
        f"`{json.dumps(after['control_overrides'], sort_keys=True)}`.",
        "",
        "The paired scene configurations, initial robot/object states, model revision "
        "and MuJoCo version have matching SHA-256 fingerprints for "
        f"{before['scheduled_scenes'] - len(unverified_pairs)} pairs; "
        f"{len(unverified_pairs)} lack sufficient initial-state evidence.",
        "Controller source revisions and effective mode switches are recorded in the JSON.",
        "",
        "This is a narrow, ten-scene simulation benchmark, not a generalization claim. "
        "The JSON companion retains failures, timings, force peaks and incident evidence.",
        "",
    ]
    if any(r["seed"] in (0, 1) for r in before["runs"]):
        lines += [
            "Seeds 0 and 1 appeared in earlier development. This public suite "
            "is regression validation, not a blind held-out test.",
            "",
        ]
    if "search" in result:
        study = result["search"]
        lines += [
            f"Development seeds: `{study['development_seeds']}`. "
            f"Search budget: {len(study['trials'])} evaluated configurations, "
            "including the default; each runs on every development seed.",
            "",
            "| Trial | Overrides | Recorded score | Accepted |",
            "|---|---|---|---|",
        ]
        for trial in study["trials"]:
            audit_rows = result.get("search_audit", {}).get("trials", [])
            score = next(
                (r["score"] for r in audit_rows if r["trial"] == trial["trial"]), trial["score"]
            )
            lines.append(
                f"| {trial['trial']} | `{json.dumps(trial['controls'])}` | "
                f"`{score}` | {trial['accepted']} |"
            )
        lines.append("")
    lines += [
        "Gross hand-normal impulse is integrated over the physics samples in the JSON. "
        "Successful-time means are conditional on completion; all failure timings remain "
        "in the per-scene data.",
        "",
    ]
    output.with_suffix(".md").write_text("\n".join(lines))
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    return result
