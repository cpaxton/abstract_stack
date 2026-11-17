#!/usr/bin/env python3
"""Native unloaded hold characterization; no human-input qualification claim."""

import argparse
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np

from abstract_stack.scene import ShelfConfig
from abstract_stack.teleop.session import EngagementGate, GateSettings, SessionState
from abstract_stack.teleop.sim import MujocoTeleopAdapter


def run(directory, *, ticks=100):
    directory.mkdir(parents=True, exist_ok=True)
    results = []
    for case, front_x, prepare_counterexample in (
        ("clearance_unloaded", -1.4, False),
        ("default_measured_hold_counterexample", -0.62, True),
    ):
        config = ShelfConfig(product_count=0, front_x=front_x, terminate_on_success=False)
        with MujocoTeleopAdapter(config, seed=0) as sim:
            preparation = []
            if prepare_counterexample:
                # #14's counterexample is reached through native actuator targets
                # and physics, never by writing robot/object poses.
                for index in range(10):
                    try:
                        targets = sim.stationary_command()
                    except ValueError as error:
                        preparation.append({"step": index, "hold_rejection": str(error)})
                        break
                    sim.apply_actuator_targets(targets)
                    preparation.append({"step": index, "targets": targets.tolist()})
                    sim.step()
            gate = EngagementGate(sim, settings=GateSettings(enabled_groups=("head",)))
            initial = sim.read_state()
            # No input/provider/guard is installed: engagement is impossible.
            gate.connect("00000000-0000-4000-8000-000000000001", now=100)
            gate.host_pause(now=100)
            traces = []
            for index in range(ticks):
                now = 100 + index * sim.control_dt  # explicit synthetic host clock
                if index == ticks // 2:
                    gate.host_pause(now=now)
                applied = gate.tick(now=now)
                telemetry = asdict(gate.telemetry(now=now))
                telemetry.update(tick=index, applied=applied is not None)
                traces.append(telemetry)
                if gate.state == SessionState.FAULT:
                    break
                sim.step()
            final = sim.read_state()
            drift = float(np.linalg.norm(final.root_position_m - initial.root_position_m))
            rows = {
                "case": case,
                "provenance": sim.describe(),
                "settings": asdict(gate.settings),
                "preparation": preparation,
                "scheduled_ticks": ticks,
                "executed_ticks": len(traces),
                "last_state": gate.state.value,
                "reason": gate.reason,
                "base_displacement_m": drift,
                "max_actuator_force_abs": max(
                    max(abs(x) for x in t["measured_forces"].values()) for t in traces
                ),
                "trace": traces,
            }
            results.append(rows)
    sources = [
        Path(__file__),
        Path("src/abstract_stack/teleop/session.py"),
        Path("src/abstract_stack/teleop/session_bridge.py"),
    ]
    output = {
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "worktree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True)
        ),
        "versions": {
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
        },
        "scope": "unloaded native physics, synthetic host time; no objects held, no liveQuest",
        "cases": results,
    }
    (directory / "native-hold.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n"
    )
    for r in results:
        print(
            json.dumps(
                {
                    k: r[k]
                    for k in (
                        "case",
                        "scheduled_ticks",
                        "executed_ticks",
                        "last_state",
                        "reason",
                        "base_displacement_m",
                    )
                }
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/quest-engagement/native"))
    parser.add_argument("--ticks", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.ticks <= 1000:
        parser.error("ticks must be1..1000")
    run(args.output, ticks=args.ticks)
