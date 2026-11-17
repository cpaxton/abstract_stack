"""Composed native gate + #17/#18; runs once independent modules are available."""

import json
from pathlib import Path
from uuid import uuid4

import mujoco
import numpy as np
import pytest

from abstract_stack.scene import ShelfConfig
from abstract_stack.teleop.calibration import Pose, TargetSpec, pose_from_tracking
from abstract_stack.teleop.session import (
    GROUPS,
    CommandProposal,
    EngagementGate,
    GateSettings,
    LifecycleEvidence,
    SessionState,
    VideoEvidence,
)
from abstract_stack.teleop.sim import MujocoTeleopAdapter
from abstract_stack.teleop.transport import ClockEstimate

arms = pytest.importorskip("abstract_stack.teleop.retarget_arms", reason="requires issue17")
fingers = pytest.importorskip("abstract_stack.teleop.retarget_fingers", reason="requires issue18")
fixtures = pytest.importorskip("abstract_stack.teleop.retarget_fingers_fixtures")
pytestmark = pytest.mark.mabel
FIXTURE = Path(__file__).parents[1] / "docs/quest_teleop/examples/tracking.json"


def input_packet(gate, now, sequence, curl=0):
    packet = json.loads(FIXTURE.read_text())
    packet.update(
        session_id=gate.session_id,
        calibration_generation=gate.generation,
        sequence=sequence,
        sent_monotonic_s=now,
    )
    for key in ("head", "left_hand", "right_hand", "body"):
        packet[key]["observed_monotonic_s"] = now
        if key.endswith("hand"):
            source = fixtures.synthetic_hand("Left" if key == "left_hand" else "Right", curl)
            packet[key]["pose"].update(source["pose"])
            packet[key]["joints"] = source["joints"]
            packet[key]["source_time_s"] = now
    gate.observe(
        packet,
        ClockEstimate(0, 0.001, now),
        LifecycleEvidence(gate.session_id, now, True, True, True, True),
        VideoEvidence(gate.session_id, gate.generation, sequence, now - 0.01, now, 0.001, True),
        now=now,
    )  # synthetic trusted providers, not physical qualification
    return packet


@pytest.mark.parametrize("groups", [("head",), ("head", "arms", "fingers")])
def test_native_composition_uses_one_gate_and_real_proposed_finger_geometry(groups):
    config = ShelfConfig(product_count=0, front_x=-1.4, terminate_on_success=False)
    with MujocoTeleopAdapter(config, seed=0) as sim, arms.ArmHeadRetargeter(config, seed=0) as rt:
        gate = EngagementGate(sim, settings=GateSettings(enabled_groups=groups))
        gate.composed_guard = lambda state, command: rt.check_composed(
            state, command, body_hold=gate.body_hold_references()
        )
        mapper = fingers.FingerRetargeter(rt.model)
        initial = sim.read_state()
        measured = rt.measured_poses(initial)
        gate.connect(str(uuid4()), now=100)
        packet = input_packet(gate, 100, 0)
        neutral = {
            name: pose_from_tracking(packet[name]["pose"])
            for name in ("head", "left_hand", "right_hand")
        }
        specs = {
            name: TargetSpec(measured[{"left_hand": "left", "right_hand": "right"}.get(name, name)])
            for name in neutral
        }
        assert gate.control(
            gate.session_id, "calibrate", 0, 0, now=100, neutral=neutral, specs=specs
        )["accepted"]
        input_packet(gate, 100.02, 1)
        assert gate.control(gate.session_id, "engage", 1, gate.generation, now=100.02)["accepted"]
        mapper.reset(dict(zip(initial.actuator_names, initial.applied_targets, strict=True)))
        target = measured["head"]
        q, matrix = np.empty(4), np.empty(9)
        mujoco.mju_axisAngle2Quat(q, np.array([0.0, 0.0, 1.0]), 0.08)
        mujoco.mju_quat2Mat(matrix, q)
        rotation = matrix.reshape(3, 3) @ target.rotation
        orientation = np.empty(4)
        mujoco.mju_mat2Quat(orientation, rotation.ravel())
        target = Pose(target.position_m, tuple(orientation))
        for index in range(30):
            now = 100.04 + index * 0.02
            packet = input_packet(gate, now, index + 2, curl=0.05 if "fingers" in groups else 0)
            state = sim.read_state()
            arm = rt.propose(
                state,
                {"left": measured["left"], "right": measured["right"]},
                target,
                channels=("arms", "head") if "arms" in groups else ("head",),
            )
            assert arm.valid, arm.reason
            proposals = {}
            for group in groups:
                if group == "fingers":
                    finger = mapper.propose(
                        packet["left_hand"],
                        packet["right_hand"],
                        sim.control_dt,
                        applied_references=dict(
                            zip(state.actuator_names, state.applied_targets, strict=True)
                        ),
                    )
                    assert finger.valid, finger.reason
                    values = finger.targets
                else:
                    values = {n: arm.targets[n] for n in GROUPS[group]}
                proposals[group] = CommandProposal(
                    gate.session_id, gate.generation, index + 2, now, values
                )
            applied = gate.tick(proposals, now=now)
            assert applied is not None, gate.reason
            assert gate.state == SessionState.ENGAGED, gate.reason
            sim.step()
        assert gate.command_sequence == 30
        assert np.max(np.abs(sim.read_state().actuator_lengths - initial.actuator_lengths)) > 0.01
        gate.host_pause(now=100.66)
        assert gate.state != SessionState.ENGAGED
        assert gate.calibration.artifact is None
        # A new valid pose cannot automatically resume after pause/fault.
        input_packet(gate, 100.68, 32)
        assert gate.state != SessionState.ENGAGED
