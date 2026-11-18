#!/usr/bin/env python3
"""Seeded native arm/head fixture execution, without networking or human actuation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np

from abstract_stack.ik import rotation_error
from abstract_stack.scene import ShelfConfig
from abstract_stack.teleop.calibration import Pose
from abstract_stack.teleop.retarget_arms import ALL_NAMES, ArmHeadRetargeter
from abstract_stack.teleop.sim import MujocoTeleopAdapter


def rotate(pose, axis, angle):
    quaternion, matrix = np.empty(4), np.empty(9)
    mujoco.mju_axisAngle2Quat(quaternion, np.asarray(axis, dtype=float), angle)
    mujoco.mju_quat2Mat(matrix, quaternion)
    mujoco.mju_mat2Quat(quaternion, (matrix.reshape(3, 3) @ pose.rotation).ravel())
    return Pose(pose.position_m, tuple(quaternion))


def errors(measured, goals):
    return {
        side: dict(
            position_error_m=None
            if side == "head"
            else float(
                np.linalg.norm(np.asarray(measured[side].position_m) - goals[side].position_m)
            ),
            orientation_error_rad=float(
                np.linalg.norm(rotation_error(measured[side].rotation, goals[side].rotation))
            ),
        )
        for side in goals
    }


def contact_peak(sim):
    # Owner-thread read-only diagnostics. The adapter is never shared with solvers.
    force = np.empty(6)
    peak = 0.0
    for index in range(sim._env.data.ncon):
        mujoco.mj_contactForce(sim._env.model, sim._env.data, index, force)
        peak = max(peak, float(abs(force[0])))
    return peak


def trial(name, config, seed, ticks):
    started = time.monotonic()
    with (
        MujocoTeleopAdapter(config, seed=seed) as sim,
        ArmHeadRetargeter(
            config, seed=seed, expected_model_sha256=sim.describe()["model_xml_sha256"]
        ) as rt,
    ):
        state = sim.read_state()
        goals = rt.measured_poses(state)
        initial = dict(goals)
        if name in ("left_up", "right_up", "both_up"):
            sides = ("left", "right") if name == "both_up" else (name.split("_")[0],)
            for side in sides:
                goals[side] = Pose(
                    tuple(np.asarray(goals[side].position_m) + [0, 0, 0.02]),
                    goals[side].quaternion_wxyz,
                )
        if name.startswith("head_"):
            axis = {"head_root_x": (1, 0, 0), "head_root_y": (0, 1, 0), "head_root_z": (0, 0, 1)}[
                name
            ]
            goals["head"] = rotate(goals["head"], axis, 0.12)
        if name == "outside_workspace":
            goals["left"] = Pose((100, 0, 1), goals["left"].quaternion_wxyz)
        trace, duration, peak_force = [], [], contact_peak(sim)
        status, reason = "executed", None
        for _ in range(ticks):
            state = sim.read_state()
            before = time.monotonic()
            proposal = rt.propose(state, {s: goals[s] for s in ("left", "right")}, goals["head"])
            duration.append(time.monotonic() - before)
            complete = dict(zip(state.actuator_names, state.applied_targets, strict=True))
            complete.update(proposal.targets)
            if name == "default_blocked_finger" and proposal.valid:
                complete["Left Index MCP"] = float(rt.model.actuator("Left Index MCP").ctrlrange[1])
            guard = rt.check_composed(state, complete) if proposal.valid else None
            row = dict(
                sim_time_s=state.sim_time_s,
                proposal_valid=proposal.valid,
                proposal_reason=proposal.reason,
                diagnostics=proposal.diagnostics,
                composed_guard=guard,
            )
            trace.append(row)
            if not proposal.valid or not guard["valid"]:
                status, reason = (
                    "rejected",
                    proposal.reason if not proposal.valid else guard["reason"],
                )
                break
            sim.apply_actuator_targets(complete)
            measured = sim.step()
            peak_force = max(peak_force, contact_peak(sim))
            row["applied_targets"] = {
                n: float(measured.applied_targets[measured.actuator_names.index(n)])
                for n in ALL_NAMES
            }
            row["measured_actuator_length"] = {
                n: float(measured.actuator_lengths[measured.actuator_names.index(n)])
                for n in ALL_NAMES
            }
            row["measured_tracking_error"] = errors(rt.measured_poses(measured), goals)
        final = sim.read_state()
        measured = rt.measured_poses(final)
        result = dict(
            name=name,
            status=status,
            reason=reason,
            simulation=sim.describe(),
            retargeter=rt.describe(),
            steps=final.step_index,
            sim_duration_s=final.sim_time_s,
            wall_duration_s=time.monotonic() - started,
            proposal_time_s=dict(
                p50=float(np.percentile(duration, 50)),
                p95=float(np.percentile(duration, 95)),
                max=max(duration),
            ),
            final_ik=trace[-1]["diagnostics"].get("ik"),
            final_measured_errors=errors(measured, goals),
            palm_displacement_m={
                s: float(np.linalg.norm(np.asarray(measured[s].position_m) - initial[s].position_m))
                for s in ("left", "right")
            },
            peak_native_contact_normal_n=peak_force,
            trace=trace,
        )
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/quest-retarget-arms/probe.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ticks", type=int, default=50)
    args = parser.parse_args()
    if not 1 <= args.ticks <= 500:
        parser.error("ticks must be1..500")
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty_at_start = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
    fingerprints = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (
            root / "scripts/probe_quest_retarget_arms.py",
            root / "src/abstract_stack/teleop/retarget_arms.py",
        )
    }
    cases = (
        "left_up",
        "right_up",
        "both_up",
        "head_root_x",
        "head_root_y",
        "head_root_z",
        "outside_workspace",
        "default_blocked_finger",
    )
    results = []
    for name in cases:
        config = ShelfConfig(
            product_count=0,
            front_x=-0.62 if name == "default_blocked_finger" else -1.4,
            terminate_on_success=False,
        )
        try:
            result = trial(name, config, args.seed, args.ticks)
        except Exception as error:
            result = dict(
                name=name, status="process_failure", error=f"{type(error).__name__}: {error}"
            )
        results.append(result)
        print(
            json.dumps(
                {
                    k: result.get(k)
                    for k in ("name", "status", "reason", "steps", "final_measured_errors")
                }
            ),
            flush=True,
        )
    report = dict(
        source_revision_at_start=revision,
        worktree_dirty_at_start=dirty_at_start,
        source_sha256=fingerprints,
        scheduled=len(cases),
        results=results,
        scope=(
            "isolated native fixture execution; not live tracking, "
            "engagement or manipulation acceptance"
        ),
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(r["status"] == "process_failure" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
