"""Pick, release on the shelf front, then insert with a native selected finger."""

import copy
import time
from dataclasses import replace

import mujoco
import numpy as np

from abstract_stack.interactions import CartesianHand, InteractionFailure, StockingScript
from abstract_stack.mabel_controller import MabelController


class FingerPusher(CartesianHand):
    """Geometry-derived extended fingertip, with the other fingers folded.

    Offset comes from native collision geometry in the straight-finger posture;
    it is a kinematic reference, not an attachment or a measured contact point.
    """

    OFFSET = np.array([0.002, 0.006, 0.139])
    BASE_ROTATION = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    def finger_targets(self, grip):
        targets = super().finger_targets(0)
        for finger in ("Index", "Middle", "Ring", "Pinky"):
            for joint in ("MCP", "PIP"):
                targets[f"Left {finger} {joint}"] = (
                    (-0.2 if joint == "MCP" else 0.0) if finger == self.finger else 1.0
                )
        return targets

    def __init__(self, env, config, *, finger="Middle", tilt_degrees=20, wrist_posture=0.4):
        if finger not in ("Index", "Middle", "Ring"):
            raise ValueError("Unsupported native pushing finger")
        if not np.isfinite(tilt_degrees) or not 0 <= tilt_degrees <= 60:
            raise ValueError("Pusher tilt must be within 0–60 degrees")
        self.finger = finger
        tilt = np.radians(tilt_degrees)
        self.ROTATION = (
            np.array(
                [
                    [np.cos(tilt), 0, -np.sin(tilt)],
                    [0, 1, 0],
                    [np.sin(tilt), 0, np.cos(tilt)],
                ]
            )
            @ self.BASE_ROTATION
        )
        super().__init__(env, config)
        # Keep the native wrist mount above the board; unrestricted redundant
        # IK can realize the palm angle with a drooping mount. This is an
        # explicit planning constraint, not a change to joint/actuator limits.
        self.ik.constrain_posture("Left Wrist", wrist_posture)
        # Derive the nominal tip from unmodified collision meshes in the
        # proposed finger posture. Actual compliant contact remains measured.
        scratch = mujoco.MjData(self.model)
        mujoco.mj_copyData(scratch, self.model, self.data)
        for name, value in self.finger_targets(0).items():
            scratch.qpos[int(self.model.joint(name).qposadr[0])] = value
        mujoco.mj_forward(self.model, scratch)
        body = self.model.body(f"left_{finger.lower()}_ip").id
        points = []
        palm_rotation = scratch.xmat[self.palm].reshape(3, 3)
        for geom in range(self.model.ngeom):
            if self.model.geom_bodyid[geom] != body or not (
                self.model.geom_contype[geom] or self.model.geom_conaffinity[geom]
            ):
                continue
            if int(self.model.geom_type[geom]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                raise ValueError("Native tip calibration requires collision meshes")
            mesh = self.model.geom_dataid[geom]
            start, count = self.model.mesh_vertadr[mesh], self.model.mesh_vertnum[mesh]
            vertices = self.model.mesh_vert[start : start + count]
            world = vertices @ scratch.geom_xmat[geom].reshape(3, 3).T + scratch.geom_xpos[geom]
            points.append((world - scratch.xpos[self.palm]) @ palm_rotation)
        if not points:
            raise ValueError("Selected native finger has no collision meshes")
        points = np.vstack(points)
        front = points[points[:, 2] >= points[:, 2].max() - 0.002]
        self.OFFSET = front.mean(axis=0)
        self.OFFSET[2] = points[:, 2].max()
        self.ik.offset = self.OFFSET.copy()
        self.last_target = self.position().copy()


class DenseStockingScript(StockingScript):
    """Distinct contact protocol; the coplanar transfer table is an explicit aid."""

    FRONT_SUPPORT = "shelf"
    CONTACT_PROTOCOL = "shelf_front_release_then_ungrasped_selected_finger_push"
    ALLOW_OFFSET_START = True

    def __init__(self, env, *args, **kwargs):
        config = env.config
        if config.robot_yaw != 0:
            raise ValueError("Dense insertion currently requires a shelf-aligned initial base yaw")
        if config.row_free_space is None:
            raise ValueError("DenseStockingScript requires a dense row scene")
        if config.initially_stocked == config.product_count:
            raise ValueError("All-prefilled rows are geometry fixtures, not insertion episodes")
        if not np.isclose(config.table_top, config.levels[0]) or not np.isclose(
            config.table_xy[0] - config.table_size[0] / 2, config.front_x
        ):
            raise ValueError("Dense insertion requires the explicit coplanar, flush transfer table")
        super().__init__(env, *args, **kwargs)
        self.object_pair_peak = 0.0
        self.object_pair_bins = {}
        self.wrist_load = 0.0
        self.wrist_load_peak = 0.0
        self.wrist_load_clipped = 0

    def _load_feedforward(self):
        if not isinstance(self.controller, FingerPusher):
            return None
        # last_forces is the wrench ON the carton from the distal hand.
        # Counter the opposite reaction at the wrist through its native servo.
        data, model = self.env.data, self.env.model
        joint = model.joint("Left Wrist").id
        force = np.asarray(self.last_forces.get("hand_force_world_N", [0, 0, 0]))
        moment = np.asarray(
            self.last_forces.get("hand_moment_about_body_origin_Nm", [0, 0, 0])
        ) + np.cross(data.body(self.product.name).xpos - data.xanchor[joint], force)
        requested = float(np.dot(moment, data.xaxis[joint]))
        # The upstream wrist has kp=1 and no native effort cap. Make the
        # additional load effort explicit and bounded, rather than changing kp.
        bounded = float(np.clip(requested, -0.3, 0.3))
        self.wrist_load_clipped += abs(requested - bounded) > 1e-9
        self.wrist_load += float(
            np.clip(bounded - self.wrist_load, -0.5 * self.env.dt, 0.5 * self.env.dt)
        )
        self.wrist_load_peak = max(self.wrist_load_peak, abs(self.wrist_load))
        return {"Left Wrist": self.wrist_load}

    def _sample(self, env):
        super()._sample(env)
        pairs = {}
        for index, contact in enumerate(env.data.contact):
            b1, b2 = (
                int(env.model.geom_bodyid[contact.geom1]),
                int(env.model.geom_bodyid[contact.geom2]),
            )
            if b1 not in self.product_bodies or b2 not in self.product_bodies:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(env.model, env.data, index, force)
            pair = tuple(sorted((b1, b2)))
            pairs[pair] = pairs.get(pair, 0.0) + max(0.0, float(force[0]))
        for (b1, b2), normal in pairs.items():
            self.object_pair_peak = max(self.object_pair_peak, normal)
            if normal > 0.01:
                key = (int(np.floor(env.data.time / env.dt + 1e-7)), b1, b2)
                entry = self.object_pair_bins.get(key)
                if entry is None or entry[-1] < normal:
                    self.object_pair_bins[key] = [
                        float(env.data.time),
                        env.model.body(b1).name,
                        env.model.body(b2).name,
                        normal,
                    ]
            if normal > self.config.max_force:
                raise InteractionFailure("dense insertion: carton/carton force limit exceeded")

    def episode_evidence(self):
        return {
            "peak_carton_carton_normal_N": self.object_pair_peak,
            "carton_contact_20ms_peaks": list(self.object_pair_bins.values()),
            "fixture_aid": "coplanar transfer table; explicitly part of the scene",
            "wrist_load_feedforward": {
                "source": "measured distal-hand/carton reaction about native wrist axis",
                "maximum_effort_Nm": 0.3,
                "maximum_effort_rate_Nm_s": 0.5,
                "peak_requested_effort_Nm": self.wrist_load_peak,
                "clipped_updates": int(self.wrist_load_clipped),
            },
        }

    def _front_center_x(self, extent):
        # Stage in the empty front portion of the deep shelf, before the
        # occupied row. Reduces unsupported lateral/yaw drift during the push.
        return self.env.config.front_x - 0.07

    def _change_hand(self, kind, **kwargs):
        previous = self.controller
        self.controller = kind(self.env, self.config, **kwargs)
        # ctrl contains compensation; carry NOMINAL references across tool
        # changes so the tracker never receives compensation twice.
        self.controller.command = previous.command.copy()
        self.controller.last_target = self.controller.position().copy()
        # Preserve both tracker states across a configuration change. Native
        # ctrl already contains compensation and must not become a new nominal.
        tracker = MabelController(
            self.env.model,
            self.env.data,
            replace(self.tracker.config, wrist_offset_limit=0.5 if kind is FingerPusher else None),
        )
        tracker.nominal = self.tracker.nominal.copy()
        tracker.applied = self.tracker.applied.copy()
        self.tracker = tracker
        self.wrist_load = 0.0
        self._stage("tool_change")
        self.events[-1].update(
            tool=kind.__name__,
            nominal_palm_offset_m=self.controller.OFFSET.tolist(),
            live_state_changed=False,
            pushing_finger=getattr(self.controller, "finger", None),
        )

    def _pushing_finger(self):
        row = sorted(self.env.products, key=lambda p: p.target[1])
        if self.product == row[0]:
            return "Index"
        if self.product == row[-1]:
            return "Ring"
        return "Middle"

    def _select_grasp(self, pending):
        if isinstance(self.controller, FingerPusher):
            self._change_hand(CartesianHand)
        return super()._select_grasp(pending)

    def _one_box(self, selected=None):
        self._move("clear_high", [-0.30, -0.20, 1.35])
        grip = self._pick_and_place_front(selected)
        self.carry_reference = None
        self._precision_release(grip)
        self.payload_rotation_goal = None
        retreat = self.controller.last_target.copy()
        for fraction in np.linspace(1 / 6, 1, 6):
            self._move(
                "staging_clear",
                retreat + [0.12 * fraction, 0, 0],
                seconds=0.4,
                orientation_alternatives=False,
            )
        self._move("staging_clear", retreat + [0.12, 0, 0.02], orientation_alternatives=False)
        self._insert_from_front()

    def _precision_release(self, grip):
        """Square the supported carton, then open in place before withdrawing.

        References integrate MEASURED object error; they are not recaptured
        from the sagging measured palm each tick. No object poses are assigned.
        """
        env, hand, product = self.env, self.controller, self.product
        desired = np.array([self._front_center_x(product.half_size[0]), product.target[1]])
        target = hand.last_target.copy()
        origin = target.copy()
        yaw_correction = 0.0
        self._stage("stage_align")
        self.events[-1]["desired_object_xy_m"] = desired.tolist()
        velocity_address = int(env.model.joint(f"{product.name}_joint").dofadr[0])

        def tick(*, opening=None):
            nonlocal yaw_correction
            body = env.data.body(product.name)
            rotation = body.xmat.reshape(3, 3)
            yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
            error = desired - body.xpos[:2]
            velocity = env.data.qvel[velocity_address : velocity_address + 6]
            speed = 2 * error - 0.5 * velocity[:2]
            yaw_speed = -2 * yaw - 0.5 * velocity[5]
            if opening is not None:
                speed -= 0.002 * np.asarray(self.last_forces["hand_force_world_N"][:2])
                yaw_speed -= 0.3 * self.last_forces["hand_moment_about_body_origin_Nm"][2]
                if self.last_forces["hand"] < 0.2:
                    speed[:] = 0
                    yaw_speed = 0.0
            target[:2] += np.clip(speed * env.dt, -0.0002, 0.0002)
            step = float(np.clip(yaw_speed * env.dt, -0.3 * env.dt, 0.3 * env.dt))
            yaw_correction += step
            if np.linalg.norm(target[:2] - origin[:2]) > 0.03 or abs(yaw_correction) > 0.26:
                raise InteractionFailure("stage_align: bounded pose-reference correction exhausted")
            c, s = np.cos(step), np.sin(step)
            hand.active_rotation = (
                np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ hand.active_rotation
            )
            prediction = hand.predict(
                target, allowed_object=product.name, alternatives=False, grip=grip, carrying=False
            )
            if not prediction.feasible:
                self.events[-1]["feedback_rejection"] = prediction.report()
                raise InteractionFailure(
                    "stage_align: measured-pose feedback rejected before actuation"
                )
            action = hand.joint_targets(prediction.q, grip)
            if opening is not None:
                # Interpolation can round an exact native endpoint a few ULPs
                # outside its range. Project commands, not acceptance limits.
                opening = {
                    name: float(np.clip(value, *env.model.actuator(name).ctrlrange))
                    for name, value in opening.items()
                }
                if hand.check_hand_step(prediction, 0, product.name, targets=opening):
                    raise InteractionFailure("release_front: coupled stationary opening rejected")
                action.update(opening)
            hand.last_target = target.copy()
            self._apply_tick(action)
            current = env.data.body(product.name)
            error_now = desired - current.xpos[:2]
            matrix_now = current.xmat.reshape(3, 3)
            yaw_now = float(np.arctan2(matrix_now[1, 0], matrix_now[0, 0]))
            self.events[-1].update(
                final_xy_error_m=error_now.tolist(),
                final_yaw_deg=float(np.degrees(yaw_now)),
                reference_xy_correction_m=(target[:2] - origin[:2]).tolist(),
                reference_yaw_correction_deg=float(np.degrees(yaw_correction)),
            )
            self.events[-1].setdefault("pose_force_samples", []).append(
                [
                    float(env.data.time),
                    *error_now,
                    float(yaw_now),
                    *self.last_forces["hand_force_world_N"],
                    self.last_forces["thumb_normal"],
                    self.last_forces["nonthumb_normal"],
                ]
            )
            return np.linalg.norm(error_now) < 0.0002 and abs(yaw_now) < np.radians(0.15)

        stable = 0.0
        for _ in range(int(6.0 / env.dt)):
            good = tick()
            stable = stable + env.dt if good else 0.0
            if stable >= 0.1:
                break
        else:
            raise InteractionFailure("stage_align: supported position/yaw alignment timeout")
        self._stage("release_unload")
        fraction = self.active_grip if self.active_grip is not None else grip
        self.active_grip = None
        fractions = dict(thumb=fraction, nonthumb=fraction)
        initial_force = min(self.last_forces["thumb_normal"], self.last_forces["nonthumb_normal"])
        self.events[-1].update(normal_target_per_group_N=0.15, separate_thumb_finger_feedback=True)
        for index in range(int(4.0 / env.dt)):
            reference = max(0.15, initial_force * (1 - (index + 1) * env.dt / 2))
            for group in fractions:
                error = reference - self.last_forces[f"{group}_normal"]
                fractions[group] = float(
                    np.clip(
                        fractions[group] + np.clip(0.04 * error, -0.25, 0.25) * env.dt,
                        0,
                        self.config.grip,
                    )
                )
            goals = hand.finger_targets(fractions["nonthumb"])
            for name, value in hand.finger_targets(fractions["thumb"]).items():
                if "Thumb" in name:
                    goals[name] = value
            tick(opening=goals)
            if (
                index * env.dt >= 2
                and max(self.last_forces["thumb_normal"], self.last_forces["nonthumb_normal"]) < 0.6
            ):
                break
        else:
            raise InteractionFailure("release_unload: hand pressure did not unload on the support")
        self._stage("release_front")
        self.release_origin = env.data.body(product.name).xpos.copy()
        opened = hand.finger_targets(0)
        start = {
            name: float(
                np.clip(
                    env.data.qpos[int(env.model.joint(name).qposadr[0])],
                    *env.model.actuator(name).ctrlrange,
                )
            )
            for name in opened
        }
        for alpha in np.linspace(0, 1, int(0.6 / env.dt)):
            tick(
                opening={
                    name: start[name] + alpha * (value - start[name])
                    for name, value in opened.items()
                }
            )
        for _ in range(int(0.2 / env.dt)):
            tick(opening=opened)
        if self.last_forces[self.FRONT_SUPPORT] < 0.2 or self.last_forces["hand"] > 0.2:
            raise InteractionFailure("release_front: no unloaded support after opening")
        self.events[-1].update(
            verified_release=True,
            support_surface=self.FRONT_SUPPORT,
            placement_conditions=False,
            stationary_opening=True,
        )
        self.release_origin = None

    @staticmethod
    def _planning_pose(hand, prediction):
        """Set a cloned data fixture, never live simulation data."""
        hand.data.qpos[:] = prediction.reference_qpos
        hand.data.qpos[hand.qadr] = prediction.q[:-1]
        for joint in hand.lift_joints:
            hand.data.qpos[int(joint.qposadr[0])] = prediction.q[-1] / 2
        mujoco.mj_forward(hand.model, hand.data)
        hand.command = prediction.q.copy()

    @staticmethod
    def _clearance_margin(hand, prediction, origin, end):
        geoms = [
            g
            for g in range(hand.model.ngeom)
            if (hand.model.geom_bodyid[g] in hand.ik.hand or g in hand.tower_geoms)
            and (hand.model.geom_contype[g] or hand.model.geom_conaffinity[g])
        ]
        groups = {
            "support": [hand.model.geom("supply_table").id, hand.model.geom("shelf_board_0").id],
            "sides": [hand.model.geom(f"shelf_side_{side}").id for side in (-1, 1)],
        }
        margins = dict.fromkeys(groups, 0.15)
        q = hand.ik.measured_q()
        for alpha in np.linspace(0, 1, 9):
            target = origin + alpha * (end - origin)
            q, point, _ = hand.ik.solve(
                target, hand.ROTATION, q, reference=prediction.reference_qpos
            )
            if np.linalg.norm(point - target) > hand.config.ik_position_tolerance:
                return {"support": -1.0, "sides": -1.0}
            hand.ik.forward(q, prediction.reference_qpos)
            mujoco.mj_fwdPosition(hand.model, hand.ik.scratch)
            for label, fixtures in groups.items():
                for geom in geoms:
                    for fixture in fixtures:
                        distance = mujoco.mj_geomDistance(
                            hand.model,
                            hand.ik.scratch,
                            geom,
                            fixture,
                            0.15,
                            None,
                        )
                        margins[label] = min(margins[label], float(distance))
        return margins

    def _plan_pusher(self, precontact, end):
        self._stage("pusher_plan")
        event = self.events[-1]
        attempts = []
        started = time.monotonic()
        for tilt in (20, 30, 10, 40):
            for wrist in (0.2, 0.4, 0.0, 0.6):
                if time.monotonic() - started > 45:
                    event.update(
                        candidates=attempts, planning_wall_seconds=time.monotonic() - started
                    )
                    raise InteractionFailure("pusher_plan: bounded planning budget exhausted")
                settings = dict(
                    finger=self._pushing_finger(), tilt_degrees=tilt, wrist_posture=wrist
                )
                for raise_first in (False, True):
                    clone = copy.copy(self.env)
                    clone.data = mujoco.MjData(self.env.model)
                    clone._renderer = None
                    mujoco.mj_copyData(clone.data, clone.model, self.env.data)
                    hand = FingerPusher(clone, self.config, **settings)
                    hand.command = self.controller.command.copy()
                    waypoints = ([precontact + [0, 0, 0.08]] if raise_first else []) + [precontact]
                    attempt = {**settings, "raise_first": raise_first}
                    attempts.append(attempt)
                    for point in waypoints:
                        prediction = hand.predict(point, alternatives=False, carrying=False)
                        if not prediction.feasible:
                            attempt.update(rejected="approach", prediction=prediction.report())
                            break
                        self._planning_pose(hand, prediction)
                    else:
                        prediction = hand.predict(
                            end,
                            allowed_object=self.product.name,
                            alternatives=False,
                            carrying=False,
                            sliding=True,
                            path_mode="cartesian",
                        )
                        if not prediction.feasible:
                            attempt.update(rejected="insertion", prediction=prediction.report())
                            continue
                        margins = self._clearance_margin(hand, prediction, precontact, end)
                        attempt["sampled_clearance_m"] = margins
                        if margins["support"] < 0.012 or margins["sides"] < 0.004:
                            attempt["rejected"] = "tracking_clearance_margin"
                            continue
                        event.update(
                            candidates=attempts,
                            selected=settings,
                            planning_wall_seconds=time.monotonic() - started,
                            decision="execute_checked_empty_hand_approach",
                        )
                        return settings, waypoints
        event.update(candidates=attempts, planning_wall_seconds=time.monotonic() - started)
        raise InteractionFailure("pusher_plan: no acceptable posture/path in bounded search")

    def _insert_from_front(self):
        env, config, product = self.env, self.config, self.product
        body = env.data.body(product.name)
        extent = np.abs(body.xmat.reshape(3, 3)) @ np.asarray(product.half_size)
        height = env.config.levels[product.level] + 0.028
        precontact = np.array([body.xpos[0] + extent[0] + 0.02, body.xpos[1], height])
        end = np.array(
            [
                env.config.front_x - env.config.depth + 2 * extent[0] - 0.0005,
                product.target[1],
                height,
            ]
        )
        settings, waypoints = self._plan_pusher(precontact, end)
        self._change_hand(FingerPusher, **settings)
        hand = self.controller
        for point in waypoints:
            self._move("pusher_approach", point, carrying=False, orientation_alternatives=False)
        self._stage("finger_push")
        # Preserve the nominal reference after approach. Recapturing the sagged
        # measured tip would ratchet the reference down before force feedback.
        target = hand.last_target.copy()
        origin = target.copy()
        prediction = hand.predict(
            end,
            allowed_object=product.name,
            alternatives=False,
            carrying=False,
            sliding=True,
            path_mode="cartesian",
        )
        event = self.events[-1]
        event.update(
            ik_prediction=prediction.report(),
            push_origin_m=origin.tolist(),
            maximum_advance_m=env.config.depth + 0.15,
            timeout_seconds=45.0,
            nominal_push_speed_m_s=0.04,
            contact_mode="ungrasped, supported by shelf front then occupied row",
            assumption="object slides along the row; no rigid hand/object attachment",
            feedback_checks=0,
            vertical_control="native contact-force admittance toward zero hand Fz",
            vertical_reference_bounds_m=[height - 0.005, height + 0.025],
        )
        if not prediction.feasible:
            raise InteractionFailure("finger_push: no acceptable empty-hand insertion path")
        held = 0.0
        for _ in range(int(45.0 / env.dt)):
            body = env.data.body(product.name)
            rotation = body.xmat.reshape(3, 3)
            yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
            if rotation[2, 2] < np.cos(np.radians(15)):
                raise InteractionFailure("finger_push: carton tilting; replan required")
            if self.last_forces["back"] >= config.contact_threshold:
                scale = np.clip((0.6 - self.last_forces["back"]) / 0.6, -1, 1)
                speed = 0.01
            else:
                scale = np.clip(
                    (config.push_force_target - self.last_forces["push_force"])
                    / config.push_force_target,
                    -1,
                    1,
                )
                speed = 0.04
            target[0] -= speed * env.dt * scale
            # Use measured yaw/slot error while the carton is still moving.
            # A centered single-point push alone cannot regulate yaw.
            contact = self.last_forces.get("hand_contact_position_world_m")
            # Torque depends on the contact lever relative to the MEASURED
            # carton center. A lever based only on the canonical slot changes
            # sign after lateral drift and can amplify yaw during a jam.
            desired_y = body.xpos[1] - np.clip(0.12 * yaw, -0.014, 0.014)
            measured_y = contact[1] if contact is not None else hand.position()[1]
            force_y = self.last_forces.get("hand_force_world_N", [0, 0, 0])[1]
            lateral_speed = (
                2 * (desired_y - measured_y)
                + 0.2 * (product.target[1] - body.xpos[1])
                - 0.004 * force_y
            )
            target[1] += np.clip(lateral_speed * env.dt, -0.01 * env.dt, 0.01 * env.dt)
            measured_height = contact[2] if contact is not None else hand.position()[2]
            force_z = self.last_forces.get("hand_force_world_N", [0, 0, 0])[2]
            vertical_speed = -0.012 * force_z + 0.2 * (height - measured_height)
            target[2] += np.clip(vertical_speed * env.dt, -0.015 * env.dt, 0.015 * env.dt)
            if (
                not origin[0] - env.config.depth - 0.15 <= target[0] <= origin[0] + 0.03
                or abs(target[1] - product.target[1]) > 0.02
                or not height - 0.005 <= target[2] <= height + 0.025
            ):
                raise InteractionFailure("finger_push: corrected target outside bounded motion")
            prediction = hand.predict(
                target,
                allowed_object=product.name,
                alternatives=False,
                carrying=False,
                sliding=True,
            )
            event["feedback_checks"] += 1
            if not prediction.feasible:
                original = hand.ik.posture_constraints[hand.ik.names.index("Left Wrist")]
                attempt = {
                    "sim_time": float(env.data.time),
                    "original_rejection": prediction.report(),
                    "candidates": [],
                }
                event.setdefault("posture_replans", []).append(attempt)
                for wrist in (0.2, 0.4, 0.0, 0.6):
                    if wrist == original:
                        continue
                    hand.ik.constrain_posture("Left Wrist", wrist)
                    candidate = hand.predict(
                        target,
                        allowed_object=product.name,
                        alternatives=False,
                        carrying=False,
                        sliding=True,
                    )
                    attempt["candidates"].append(candidate.report())
                    if candidate.feasible:
                        prediction = candidate
                        attempt.update(decision="checked_posture_replan", wrist_posture=wrist)
                        break
                else:
                    hand.ik.constrain_posture("Left Wrist", original)
                    event["feedback_rejection"] = prediction.report()
                    raise InteractionFailure("finger_push: feedback path rejected before actuation")
            hand.last_target = target.copy()
            self._apply_tick(hand.joint_targets(prediction.q, 0))
            event.setdefault("push_control_samples", []).append(
                [
                    float(env.data.time),
                    *env.data.body(product.name).xpos,
                    *self.last_forces["hand_force_world_N"],
                    *target,
                ]
            )
            if self.last_forces["hand"] > 0.2:
                self.supported_slide = True
            if self.last_forces["shelf"] > 0.2:
                event["verified_shelf_support"] = True
            contacted = (
                self.last_forces["back"] >= config.contact_threshold
                and self.last_forces["hand"] > 0.2
                and self.last_forces["shelf"] > 0.2
                and rotation[2, 2] >= np.cos(np.radians(config.payload_upright_degrees))
            )
            held = held + env.dt if contacted else 0.0
            if held >= config.contact_dwell:
                event["back_contact_verified"] = True
                break
        else:
            raise InteractionFailure("finger_push: contact dwell timeout")
        self.supported_slide = False
        self.observer.begin_retreat(product.name)
        self._release_grasp()
        self._move("clear", hand.position() + [0.04, 0, 0.02], orientation_alternatives=False)
        self.observer.end_retreat(product.name)
