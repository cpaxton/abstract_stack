#!/usr/bin/env python3
"""Read-only failed-episode IK/reference diagnosis using its frozen package.

Replays recorded actuator controls into independent MjData. Alternative IK and
bounded-step comparisons are scratch forecasts, never fresh task outcomes.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from abstract_stack.config import load_scene_config
from abstract_stack.dense_interactions import FingerPusher
from abstract_stack.env import ShelfStockingEnv
from abstract_stack.interactions import ScriptConfig
from abstract_stack.provenance import policy_provenance


def state(model, data):
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    values = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, values, spec)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bounded-step-module", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = json.loads((args.episode / "summary.json").read_text())
    config = ScriptConfig(**summary["controller"])
    env = ShelfStockingEnv(load_scene_config(args.episode / "scene.json"))
    model = env.model
    terminal = np.load(args.episode / "rollout_terminal.npz")
    mujoco.mj_setState(model, env.data, terminal["state"], int(terminal["state_spec"]))
    mujoco.mj_forward(model, env.data)
    before = state(model, env.data)
    event = next(e for e in reversed(summary["events"]) if e["phase"] == "finger_push")
    approach = next(e for e in reversed(summary["events"]) if e["phase"] == "pusher_approach")
    settings = next(
        e["selected"] for e in reversed(summary["events"]) if e["phase"] == "pusher_plan"
    )
    offset = np.asarray(
        next(
            e["nominal_palm_offset_m"]
            for e in reversed(summary["events"])
            if e["phase"] == "tool_change"
        )
    )
    product = event["product"]
    target = np.asarray(summary["last_ik_prediction"]["requested_position"])
    bounded = None
    if args.bounded_step_module:
        module_name = "abstract_stack._contact_reference_diagnostic_feedback"
        spec = importlib.util.spec_from_file_location(module_name, args.bounded_step_module)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        bounded = module.bounded_cartesian_step

    def hand_for(data, cfg=config, **choices):
        world = SimpleNamespace(model=model, data=data, dt=env.dt)
        hand = FingerPusher(world, cfg, **(choices or settings))
        calibrated = hand.OFFSET.copy()
        hand.OFFSET, hand.ik.offset = offset.copy(), offset.copy()
        return hand, calibrated

    hand, recalibrated = hand_for(env.data)
    nominal = np.array(
        [summary["last_nominal_targets"][name] for name in hand.names]
        + [summary["last_nominal_targets"]["Lift"]]
    )
    hand.command = nominal.copy()
    point_nominal, _ = hand.ik.forward(nominal)
    point_measured = hand.position()
    result = dict(
        qualification=(
            "Recorded-failure replay and scratch counterfactual forecasts; zero task credit"
        ),
        source=policy_provenance(),
        failure=summary["failure"],
        selected=settings,
        original_offset_m=offset.tolist(),
        terminal_recalibrated_offset_m=recalibrated.tolist(),
        posture_replans=event.get("posture_replans", []),
        requested_position_m=target.tolist(),
        sent_nominal_fk_m=point_nominal.tolist(),
        measured_reference_point_m=point_measured.tolist(),
        nominal_fk_minus_requested_m=(point_nominal - target).tolist(),
        measured_minus_nominal_fk_m=(point_measured - point_nominal).tolist(),
        measured_nominal_joint_error=(hand.ik.measured_q() - nominal).tolist(),
        joint_velocity=dict(
            (name, float(env.data.qvel[int(model.joint(name).dofadr[0])]))
            for name in hand.names + ["Lift Lower", "Lift Upper"]
        ),
        original_last_ik=summary["last_ik_prediction"],
    )
    candidates = []
    for tilt in (10, 20, 30, 40, 50, 60):
        for wrist in (0.0, 0.2, 0.4, 0.6):
            candidate, _ = hand_for(
                env.data,
                replace(config, ik_position_tolerance=0.001),
                finger=settings["finger"],
                tilt_degrees=tilt,
                wrist_posture=wrist,
            )
            candidate.command = nominal.copy()
            endpoint = candidate.predict(
                target,
                allowed_object=product,
                alternatives=False,
                carrying=False,
                sliding=True,
                check_path=False,
            )
            path = candidate.predict(
                target,
                allowed_object=product,
                alternatives=False,
                carrying=False,
                sliding=True,
                path_mode="cartesian",
            )
            entry = dict(
                tilt_degrees=tilt,
                wrist_posture=wrist,
                endpoint=endpoint.report(),
                cartesian_path=path.report(),
            )
            if path.joint_path is not None and len(path.joint_path):
                initial_gap = path.joint_path[0] - candidate.ik.measured_q()
                rates = np.r_[np.ones(8), 0.08]
                entry.update(
                    retained_path_initial_q_gap=initial_gap.tolist(),
                    omitted_initial_gap_minimum_seconds=float(
                        1.875 * np.max(np.abs(initial_gap) / rates)
                    ),
                )
            candidates.append(entry)
    result["strict_1mm_candidates"] = candidates
    result["terminal_state_unchanged_by_search"] = bool(
        np.array_equal(before, state(model, env.data))
    )

    # Reproduce the original physical history from recorded native targets,
    # while observing the original IK decisions at the final fourteen ticks.
    replay = mujoco.MjData(model)
    original = np.load(args.episode / "rollout_start.npz")
    mujoco.mj_setState(model, replay, original["state"], int(original["state_spec"]))
    mujoco.mj_forward(model, replay)
    archive = np.load(args.episode / "trace.npz")
    controls, samples = archive["controls"], archive["samples"]
    targets = [np.asarray(row[-3:]) for row in event.get("push_control_samples", [])] + [target]
    virtual, approach_prediction, tick = None, None, 0
    decisions, contact_bodies = [], []
    for control, expected in zip(controls, samples, strict=True):
        if virtual is None and replay.time >= approach["sim_time"] - 1e-7:
            virtual, _ = hand_for(replay)
            approach_prediction = virtual.predict(
                approach["ik_prediction"]["requested_position"],
                alternatives=False,
                carrying=False,
                path_mode="cartesian",
            )
            virtual.command = approach_prediction.q.copy()
            result["recomputed_approach"] = approach_prediction.report()
            if approach_prediction.joint_path is not None:
                qgap = approach_prediction.joint_path[0] - virtual.ik.measured_q()
                result["approach_initial_retained_q_gap"] = qgap.tolist()
                result["approach_omitted_gap_seconds"] = float(
                    1.875 * np.max(np.abs(qgap) / np.r_[np.ones(8), 0.08])
                )
        if (
            virtual is not None
            and tick < len(targets)
            and replay.time >= event["sim_time"] + tick * env.dt - 1e-7
        ):
            requested = targets[tick]
            previous_q = virtual.command.copy()
            prediction = virtual.predict(
                requested, allowed_object=product, alternatives=False, carrying=False, sliding=True
            )
            action = virtual.joint_targets(prediction.q, 0)
            nominal_fk, _ = virtual.ik.forward(virtual.command, prediction.reference_qpos)
            item = dict(
                sim_time=float(replay.time),
                requested_position_m=requested.tolist(),
                ik_q=prediction.q.tolist(),
                previous_nominal_q=previous_q.tolist(),
                sent_nominal_q=virtual.command.tolist(),
                measured_q=virtual.ik.measured_q().tolist(),
                ik_position_error_m=prediction.position_error_m,
                sent_nominal_position_m=nominal_fk.tolist(),
                sent_nominal_error_m=float(np.linalg.norm(nominal_fk - requested)),
                sent_nominal_error_xyz_m=(nominal_fk - requested).tolist(),
                ik_minimum_seconds=prediction.minimum_seconds,
                original_action_targets=action,
            )
            if bounded:
                step = bounded(
                    virtual.ik,
                    previous_q,
                    requested,
                    virtual.ROTATION,
                    dt=env.dt,
                    reference=prediction.reference_qpos,
                )
                item["bounded_step_counterfactual"] = step.report()
                item["bounded_step_q"] = step.q.tolist()
            decisions.append(item)
            tick += 1
        replay.ctrl[:] = control
        mujoco.mj_step(model, replay)
        mujoco.mj_forward(model, replay)
        if replay.time >= event["sim_time"] and expected[7] > 0.2:
            for index, contact in enumerate(replay.contact):
                b1, b2 = (
                    int(model.geom_bodyid[contact.geom1]),
                    int(model.geom_bodyid[contact.geom2]),
                )
                if model.body(product).id not in (b1, b2):
                    continue
                other = b2 if b1 == model.body(product).id else b1
                if other not in virtual.ik.hand:
                    continue
                force = np.zeros(6)
                mujoco.mj_contactForce(model, replay, index, force)
                if force[0] > 0.05:
                    contact_bodies.append(
                        [float(replay.time), model.body(other).name, float(force[0])]
                    )
    result["original_control_replay"] = dict(
        terminal_state_max_absolute_difference=float(
            np.max(np.abs(state(model, replay) - terminal["state"]))
        ),
        reconstructed_final_nominal_q_difference=(virtual.command - nominal).tolist(),
        online_ik_decisions=decisions,
        hand_contacts=contact_bodies,
    )
    np.savez_compressed(args.output / "replayed_terminal.npz", state=state(model, replay))
    (args.output / "diagnostic.json").write_text(json.dumps(result, indent=2) + "\n")
    sources = args.output / "diagnostic_sources"
    sources.mkdir()
    (sources / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    if args.bounded_step_module:
        (sources / args.bounded_step_module.name).write_bytes(args.bounded_step_module.read_bytes())
    (sources / "hashes.json").write_text(
        json.dumps(
            {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources.glob("*.py")},
            indent=2,
        )
        + "\n"
    )
    env.close()
    print(
        json.dumps(
            {
                "terminal_unchanged": result["terminal_state_unchanged_by_search"],
                "nominal_fk_minus_requested_m": result["nominal_fk_minus_requested_m"],
                "replay_difference": result["original_control_replay"][
                    "terminal_state_max_absolute_difference"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
