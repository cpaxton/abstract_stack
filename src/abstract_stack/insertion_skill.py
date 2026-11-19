"""Reusable physical shelf-front release and selected-finger insertion skills.

Robot tool classes are imported at invocation time to keep the shared skill
independent of StockingScript's module initialization. All physics stays native.
"""

import time
from dataclasses import replace

import mujoco
import numpy as np

from abstract_stack.contact_progress import ContactProgress
from abstract_stack.mabel_controller import MabelController
from abstract_stack.skill_entries import record_skill_entry


class SelectedFingerInsertion:
    FORWARD_UNLOAD_DOWNFORCE_N = 0.6

    def _pusher_class(self):
        from abstract_stack.finger_pusher import (
            FingerPusher,
            PairedFingerPusher,
            WallFingerPusher,
        )

        if (
            self.config.paired_finger_insertion
            and self._wall_front_staging_y() != self.product.target[1]
        ):
            return WallFingerPusher
        return (
            getattr(self, "PAIRED_PUSHER_CLASS", PairedFingerPusher)
            if self.config.paired_finger_insertion
            else getattr(self, "PUSHER_CLASS", FingerPusher)
        )

    def _initialize_insertion_state(self):
        self.object_pair_peak = 0.0
        self.object_pair_bins = {}
        self.wrist_load = self.wrist_load_peak = self.wrist_load_requested_peak = 0.0
        self.wrist_load_last_requested = 0.0
        self.wrist_load_clipped = 0
        self.carton_wall_peak = 0.0
        self.carton_wall_forces = {}
        self._carton_wall_geoms = {self.env.model.geom(f"shelf_side_{side}").id for side in (-1, 1)}

    def _restore_grasp_hand(self):
        from abstract_stack.finger_pusher import FingerPusher
        from abstract_stack.interactions import CartesianHand

        if isinstance(self.controller, FingerPusher):
            self._change_hand(CartesianHand)

    def _ungrasped_forward_fraction(self):
        """Unload downward wrench ON the carton before advancing a free push."""
        force_z = self.last_forces.get("hand_force_world_N", [0, 0, 0])[2]
        return float(np.clip(1 + force_z / self.FORWARD_UNLOAD_DOWNFORCE_N, 0, 1))

    def _pusher_load_feedforward(self):
        from abstract_stack.finger_pusher import FingerPusher

        if not isinstance(self.controller, FingerPusher):
            return None
        return self._wrist_contact_load()

    def _wrist_contact_load(self):
        # last_forces is the wrench ON the carton from the distal hand.
        # Counter the opposite reaction at the wrist through its native servo.
        data, model = self.env.data, self.env.model
        joint = model.joint("Left Wrist").id
        force = np.asarray(self.last_forces.get("hand_force_world_N", [0, 0, 0]))
        moment = np.asarray(
            self.last_forces.get("hand_moment_about_body_origin_Nm", [0, 0, 0])
        ) + np.cross(data.body(self.product.name).xpos - data.xanchor[joint], force)
        requested = float(np.dot(moment, data.xaxis[joint]))
        self.wrist_load_last_requested = requested
        self.wrist_load_requested_peak = max(self.wrist_load_requested_peak, abs(requested))
        # The upstream wrist has kp=1 and no native effort cap. Make the
        # additional load effort explicit and bounded, rather than changing kp.
        bounded = float(np.clip(requested, -0.3, 0.3))
        self.wrist_load_clipped += abs(requested - bounded) > 1e-9
        self.wrist_load += float(
            np.clip(bounded - self.wrist_load, -0.5 * self.env.dt, 0.5 * self.env.dt)
        )
        self.wrist_load_peak = max(self.wrist_load_peak, abs(self.wrist_load))
        return {"Left Wrist": self.wrist_load}

    def _sample_carton_pairs(self, env):
        from abstract_stack.interactions import InteractionFailure

        model, data = env.model, env.data
        pairs = {}
        walls = {}
        force = np.zeros(6)
        # Read the geom column in native contact order without constructing a
        # separate Python contact wrapper for every contact at every step.
        for index, (geom1, geom2) in enumerate(data.contact.geom):
            b1, b2 = int(model.geom_bodyid[geom1]), int(model.geom_bodyid[geom2])
            carton_pair = b1 in self.product_bodies and b2 in self.product_bodies
            wall_pair = (
                b1 in self.product_bodies
                and geom2 in self._carton_wall_geoms
                or b2 in self.product_bodies
                and geom1 in self._carton_wall_geoms
            )
            if not carton_pair and not wall_pair:
                continue
            mujoco.mj_contactForce(model, data, index, force)
            if carton_pair:
                pair = tuple(sorted((b1, b2)))
                pairs[pair] = pairs.get(pair, 0.0) + max(0.0, float(force[0]))
            else:
                key = b1 if b1 in self.product_bodies else b2
                walls[key] = walls.get(key, 0.0) + max(0.0, float(force[0]))
        self.carton_wall_forces = dict(walls)
        if walls:
            self.carton_wall_peak = max(self.carton_wall_peak, max(walls.values()))
            if max(walls.values()) > self.config.max_force:
                raise InteractionFailure("insertion: summed carton/sidewall force limit exceeded")
        for (b1, b2), normal in pairs.items():
            self.object_pair_peak = max(self.object_pair_peak, normal)
            if normal > 0.01:
                key = (int(np.floor(env.data.time / env.dt + 1e-7)), b1, b2)
                entry = self.object_pair_bins.get(key)
                if entry is None or entry[-1] < normal:
                    self.object_pair_bins[key] = [
                        float(env.data.time),
                        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1),
                        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2),
                        normal,
                    ]
            if normal > self.config.max_force:
                raise InteractionFailure("dense insertion: carton/carton force limit exceeded")

    def _wrist_offset_limit(self, kind):
        from abstract_stack.finger_pusher import FingerPusher

        if issubclass(kind, FingerPusher):
            return 0.5
        return 0.4 if self.config.payload_wrist_compensation else None

    def _change_hand(self, kind, **kwargs):
        from abstract_stack.finger_pusher import FingerPusher

        previous = self.controller
        self.controller = kind(self.env, self.config, **kwargs)
        self.controller.tool_parameters = dict(kwargs)
        if not issubclass(kind, FingerPusher):
            self._configure_payload_hand(self.controller)
        # ctrl contains compensation; carry NOMINAL references across tool
        # changes so the tracker never receives compensation twice.
        self.controller.command = previous.command.copy()
        self.controller.last_target = self.controller.position().copy()
        # Preserve both tracker states across a configuration change. Native
        # ctrl already contains compensation and must not become a new nominal.
        tracker = MabelController(
            self.env.model,
            self.env.data,
            replace(
                self.tracker.config,
                wrist_offset_limit=self._wrist_offset_limit(kind),
            ),
            nominal_references=(
                dict(zip(self.tracker.NAMES, self.tracker.nominal, strict=True))
                if self.tracker.held_unknown_names
                else None
            ),
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
            pushing_fingers=list(
                getattr(self.controller, "fingers", (getattr(self.controller, "finger", None),))
            ),
            tool_parameters=dict(kwargs),
        )

    def _pushing_finger(self):
        if (
            self.config.paired_finger_insertion
            and self._wall_front_staging_y() != self.product.target[1]
        ):
            return "Pinky" if self.product.target[1] > self._wall_front_staging_y() else "Index"
        row = sorted(self.env.products, key=lambda p: p.target[1])
        if self.product == row[0]:
            return "Index"
        if self.product == row[-1]:
            return "Ring"
        return "Middle"

    def _pushing_finger_candidates(self):
        return (self._pushing_finger(),)

    def _pusher_settings(self):
        for finger in self._pushing_finger_candidates():
            for tilt in getattr(self, "PUSHER_TILTS", (20, 30, 10, 40)):
                for wrist in getattr(self._pusher_class(), "WRISTS", (0.2, 0.4, 0.0, 0.6)):
                    yield dict(finger=finger, tilt_degrees=tilt, wrist_posture=wrist)

    def _release_and_insert_front(self, grip, *, push=True):
        """Finish a measured supported front placement, including checked retreat.

        ``push=False`` stops after the front registration stage so the contact
        finish (finger push) and retreat can be composed as separate skills. The
        default keeps the fused release/insert/retreat behavior unchanged.
        """
        self.carry_reference = None
        self._precision_release(grip)
        self.payload_rotation_goal = None
        retreat = self.controller.last_target.copy()
        clear_fractions = np.linspace(1 / 6, 1, 6)
        if getattr(self, "front_lead_y_m", 0.0):
            self._cartesian_move("staging_clear_prefix", retreat + [0.02, 0, 0])
            clear_fractions = clear_fractions[1:]
        for fraction in clear_fractions:
            self._move(
                "staging_clear",
                retreat + [0.12 * fraction, 0, 0],
                seconds=0.4,
                orientation_alternatives=False,
            )
        self._move("staging_clear", retreat + [0.12, 0, 0.02], orientation_alternatives=False)
        if self.config.ungrasped_insertion:
            from abstract_stack.interactions import InteractionFailure

            # Reobserve after the grasp hand has cleared. A successful seat
            # does not establish that release left an executable push entry.
            body = self.env.data.body(self.product.name)
            forces = self.last_forces
            supported = forces[self.FRONT_SUPPORT] > 0.2
            clear = forces["robot_product"] <= 0.2
            upright = body.xmat[8] >= np.cos(np.radians(self.config.payload_upright_degrees))
            self._stage(
                "free_landing_verified"
                if supported and clear and upright
                else "free_landing_rejected"
            )
            self.events[-1].update(
                measured_landing_world_m=body.xpos.tolist(),
                measured_landing_quaternion_wxyz=body.xquat.tolist(),
                canonical_final_target_world_m=list(self.product.target),
                support_normal_N=forces[self.FRONT_SUPPORT],
                robot_product_normal_N=forces["robot_product"],
                upright=bool(upright),
            )
            if not (supported and clear and upright):
                raise InteractionFailure("free_landing: carton not supported, upright and clear")
        if self._wall_front_staging_y() != self.product.target[1]:
            self._register_front_laterally()
        self._stage("front_registered")
        if not push:
            return
        self._insert_from_front()

    def _wall_front_staging_y(self):
        """Leave room to unload a wrapping grasp before moving an edge carton."""
        if not self.config.paired_finger_insertion:
            return self.product.target[1]
        env, product = self.env, self.product
        margin = product.half_size[1] + 0.05
        lower = env.config.center_y - env.config.width / 2 + margin
        upper = env.config.center_y + env.config.width / 2 - margin
        if lower > upper:
            raise ValueError("Shelf has no wrapping-hand front staging corridor")
        return float(np.clip(product.target[1], lower, upper))

    @property
    def lateral_goal_y(self):
        staging = self._wall_front_staging_y()
        canonical = self.product.target[1]
        if self.config.center_last:
            from abstract_stack.interactions import RowCapacityFailure
            from abstract_stack.packing import is_two_sided_slot, open_slot_interval

            lower, upper = open_slot_interval(self.env, self.product, self.observer.completed)
            if lower > upper:
                raise RowCapacityFailure(
                    dict(
                        admissible_center_lower_y_m=lower,
                        admissible_center_upper_y_m=upper,
                        open_span_shortfall_m=lower - upper,
                    )
                )
            if is_two_sided_slot(self.env, self.product, self.observer.completed):
                return float((lower + upper) / 2)
            # Pack each completed half outward, preserving the measured
            # reserve for the pending slots between them.
            return float(upper if canonical >= self.env.config.center_y else lower)
        if staging == canonical:
            if self.config.ungrasped_insertion and self.env.config.row_free_space is None:
                env, product = self.env, self.product
                goal = float(canonical)
                for item in self._adjacent_placed_neighbors():
                    name = item["product"]
                    neighbor = next(p for p in env.products if p.name == name)
                    occupied = env.data.body(name)
                    extent_y = float((np.abs(occupied.xmat.reshape(3, 3)) @ neighbor.half_size)[1])
                    separation = extent_y + product.half_size[1] + 0.001
                    if occupied.xpos[1] > canonical:
                        goal = min(goal, float(occupied.xpos[1]) - separation)
                    else:
                        goal = max(goal, float(occupied.xpos[1]) + separation)
                return goal
            if self.config.wall_packing:
                from abstract_stack.interactions import RowCapacityFailure

                completed = self.observer.completed
                higher = [
                    p
                    for p in self.env.products
                    if p.level == self.product.level
                    and p.target[1] > canonical
                    and p.name in completed
                ]
                if higher:
                    neighbor = min(higher, key=lambda p: p.target[1])
                    body = self.env.data.body(neighbor.name)
                    extent = float((np.abs(body.xmat.reshape(3, 3)) @ neighbor.half_size)[1])
                    goal = float(body.xpos[1] - extent - self.product.half_size[1])
                    lower_width = sum(
                        2 * p.half_size[1]
                        for p in self.env.products
                        if p.level == self.product.level and p.target[1] < canonical
                    )
                    lower_wall = self.env.config.center_y - self.env.config.width / 2
                    reserve = lower_wall + lower_width + self.product.half_size[1]
                    if goal < reserve or abs(goal - canonical) > 0.02:
                        available = float(body.xpos[1]) - extent - lower_wall
                        required = lower_width + 2 * self.product.half_size[1]
                        raise RowCapacityFailure(
                            dict(
                                measured_neighbor=neighbor.name,
                                available_open_span_m=available,
                                required_remaining_width_m=required,
                                open_span_shortfall_m=max(0.0, required - available),
                                lateral_goal_world_y_m=goal,
                                canonical_lateral_goal_world_y_m=float(canonical),
                            )
                        )
                    return goal
            return float(canonical)
        direction = 1 if canonical > staging else -1
        return float(
            self.env.config.center_y
            + direction * (self.env.config.width / 2 - self.product.half_size[1])
        )

    def _cartesian_move(self, *args, **kwargs):
        # Reuse the existing native exposed-side skill without duplicating its
        # coupled path admission or actuator execution. Import after class setup.
        from abstract_stack.lateral_contact import LateralStockingScript

        return LateralStockingScript._cartesian_move(self, *args, **kwargs)

    def _register_front_laterally(self):
        from abstract_stack.interactions import InteractionFailure
        from abstract_stack.lateral_contact import LateralPushConfig, LateralStockingScript

        registration_y = self.lateral_goal_y
        for _ in range(2):
            error = float(registration_y - self.env.data.body(self.product.name).xpos[1])
            if abs(error) <= 0.0005:
                break
            LateralStockingScript.push_laterally(
                self,
                direction=1 if error > 0 else -1,
                distance_m=min(abs(error), 0.05),
                settings=LateralPushConfig(
                    speed_m_s=0.015,
                    force_target_N=2.5,
                    timeout_s=30,
                    distance_limit_m=0.05,
                    position_tolerance_m=0.0005,
                ),
                yaw_candidates=(90, 60, 45, 30),
                brace_wall=True,
            )
        error = float(registration_y - self.env.data.body(self.product.name).xpos[1])
        if abs(error) > 0.0005:
            raise InteractionFailure("front_lateral: bounded registration needs replanning")
        goal = self.controller.last_target.copy()
        goal[0] = max(goal[0], self.env.config.front_x + 0.08)
        self._cartesian_move("lateral_front_exit", goal)

    def _front_reference_y(self):
        """Choose the handoff corridor before seating, preserving task targets."""
        product, env = self.product, self.env
        if not self.config.ungrasped_insertion:
            return product.target[1]
        from abstract_stack.interactions import InteractionFailure

        staging = self._wall_front_staging_y()
        lead = getattr(self, "front_lead_y_m", 0.0)
        if staging != product.target[1]:
            # Previous stocks occupy the back of the board. This explicit
            # front-only intermediate goal is followed by a measured side push;
            # the task's canonical final target is never changed.
            toward_center = np.sign(env.config.center_y - staging)
            return float(staging + toward_center * lead)

        lower_y, upper_y = product.target[1] - 0.02, product.target[1] + 0.02
        clearance_m = self.config.adjacent_release_clearance_m
        if env.config.row_free_space is not None:
            row = [p for p in env.products if p.level == product.level]
            gap = env.config.row_free_space / (len(row) + 1)
            clearance_m = min(clearance_m, 0.4 * gap)
            # The row budget includes both walls. Choose a reachable handoff
            # inside them instead of borrowing space from outside the shelf.
            half_width = product.half_size[1]
            lower_y = max(
                lower_y, env.config.center_y - env.config.width / 2 + half_width + clearance_m
            )
            upper_y = min(
                upper_y, env.config.center_y + env.config.width / 2 - half_width - clearance_m
            )
        shelf_y_bounds = (lower_y, upper_y)
        neighbors = self._adjacent_placed_neighbors()
        for item in neighbors:
            neighbor = next(p for p in env.products if p.name == item["product"])
            other = env.data.body(neighbor.name)
            extent_y = float((np.abs(other.xmat.reshape(3, 3)) @ neighbor.half_size)[1])
            clearance = extent_y + product.half_size[1] + clearance_m
            if other.xpos[1] > product.target[1]:
                upper_y = min(upper_y, float(other.xpos[1]) - clearance)
            else:
                lower_y = max(lower_y, float(other.xpos[1]) + clearance)
        if lower_y > upper_y:
            raise InteractionFailure("release_front: no bounded neighbor-clear staging interval")
        reference_y = float(
            np.clip(
                product.target[1] + getattr(self, "front_landing_y_offset_m", 0.0), lower_y, upper_y
            )
        )
        if lead:
            # The lead landing occupies only the free front depth band.
            leading_edge = self._front_center_x(product.half_size[0]) - product.half_size[0]
            for item in neighbors:
                other = next(p for p in env.products if p.name == item["product"])
                body = env.data.body(other.name)
                depth = (np.abs(body.xmat.reshape(3, 3)) @ other.half_size)[0]
                if leading_edge < body.xpos[0] + depth + 0.02:
                    raise InteractionFailure("release_front: no free depth band for lead landing")
            toward_center = np.sign(env.config.center_y - reference_y)
            reference_y = float(np.clip(reference_y + toward_center * lead, *shelf_y_bounds))
        return reference_y

    def _precision_release(self, grip):
        """Square the supported carton, then open in place before withdrawing.

        References integrate MEASURED object error; they are not recaptured
        from the sagging measured palm each tick. No object poses are assigned.
        """
        from abstract_stack.interactions import InteractionFailure

        env, hand, product = self.env, self.controller, self.product
        front_x = self._front_center_x(product.half_size[0])
        front_x += getattr(self, "front_landing_x_offset_m", 0.0)
        desired = np.array([front_x, product.target[1]])
        desired[1] = self._front_reference_y()
        target = hand.last_target.copy()
        origin = target.copy()
        yaw_correction = 0.0
        self._stage("stage_align")
        self.events[-1]["desired_object_xy_m"] = desired.tolist()
        self.events[-1]["canonical_target_y_m"] = float(product.target[1])
        position_gain = (
            self.config.front_alignment_position_gain_s if self.config.ungrasped_insertion else 2
        )
        yaw_gain = self.config.front_alignment_yaw_gain_s if self.config.ungrasped_insertion else 2
        self.events[-1].update(position_gain_s=position_gain, yaw_gain_s=yaw_gain)
        velocity_address = int(env.model.joint(f"{product.name}_joint").dofadr[0])

        def wall_digit_holds(opening, collisions):
            digits = {
                digit
                for digit in ("Index", "Middle", "Ring", "Pinky")
                if any(
                    any(g.startswith("shelf_side_") for g in error["geoms"])
                    and any(g.startswith(f"left_{digit.lower()}_") for g in error["geoms"])
                    for error in collisions
                )
            }
            held = {
                name: float(env.data.ctrl[env.model.actuator(name).id])
                for name in opening
                if any(name.startswith(f"Left {digit} ") for digit in digits)
            }
            return held

        def tick(*, opening=None):
            nonlocal yaw_correction
            body = env.data.body(product.name)
            rotation = body.xmat.reshape(3, 3)
            yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
            error = desired - body.xpos[:2]
            velocity = env.data.qvel[velocity_address : velocity_address + 6]
            speed = position_gain * error - 0.5 * velocity[:2]
            yaw_speed = -yaw_gain * yaw - 0.5 * velocity[5]
            if opening is not None:
                speed -= 0.002 * np.asarray(self.last_forces["hand_force_world_N"][:2])
                yaw_speed -= 0.3 * self.last_forces["hand_moment_about_body_origin_Nm"][2]
                if self.last_forces["hand"] < 0.2:
                    speed[:] = 0
                    yaw_speed = 0.0
                if self.config.ungrasped_insertion and self.config.continuous_finger_opening:
                    # A loosening grasp no longer transmits a reliable rigid
                    # pose correction. Hold the arm reference while pressure
                    # unloads instead of winding it up against a slipping box.
                    speed[:] = 0
                    yaw_speed = 0.0
            step = float(np.clip(yaw_speed * env.dt, -0.3 * env.dt, 0.3 * env.dt))
            c, s = np.cos(step), np.sin(step)
            turn = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            translation = np.r_[speed * env.dt, 0.0]
            if self.config.ungrasped_insertion:
                # Rotate the nominal grasp reference about the measured carton
                # center. Rotating about the offset tool point alone introduces
                # translation and makes the centering and yaw loops fight.
                translation += (turn - np.eye(3)) @ (target - body.xpos)
                self.events[-1]["yaw_reference_pivot"] = "measured_carton_center"
            target[:2] += np.clip(translation[:2], -0.0002, 0.0002)
            yaw_correction += step
            if np.linalg.norm(target[:2] - origin[:2]) > 0.03 or abs(yaw_correction) > 0.26:
                raise InteractionFailure("stage_align: bounded pose-reference correction exhausted")
            hand.active_rotation = turn @ hand.active_rotation
            predict = self._contact_prediction if self.config.ungrasped_insertion else hand.predict
            prediction = predict(
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
                collisions = hand.check_hand_step(prediction, 0, product.name, targets=opening)
                if (
                    collisions
                    and self.config.live_guarded_withdrawal
                    and self.phase == "release_front"
                ):
                    held = wall_digit_holds(opening, collisions)
                    if held:
                        candidate = opening | held
                        if not hand.check_hand_step(prediction, 0, product.name, targets=candidate):
                            hand.open_posture_overrides.update(held)
                            opening = candidate
                            self.events[-1].setdefault("wall_side_digit_holds_rad", {}).update(held)
                            collisions = []
                if collisions:
                    self.events[-1]["finger_sweep_rejection"] = collisions
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
        for _ in range(int(self.config.alignment_timeout_seconds / env.dt)):
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
        precise_handoff = self.config.ungrasped_insertion and self.config.continuous_finger_opening
        force_goal = 0.15
        unload_seconds = 4.0
        group_gate = 0.6
        self.events[-1].update(
            normal_target_per_group_N=force_goal,
            separate_thumb_finger_feedback=True,
            maximum_seconds=unload_seconds,
            handoff_group_force_gate_N=group_gate,
        )
        for index in range(int(unload_seconds / env.dt)):
            reference = max(force_goal, initial_force * (1 - (index + 1) * env.dt / 2))
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
                and max(self.last_forces["thumb_normal"], self.last_forces["nonthumb_normal"])
                < group_gate
            ):
                break
        else:
            raise InteractionFailure("release_unload: hand pressure did not unload on the support")
        self._stage("release_front")
        self.release_origin = env.data.body(product.name).xpos.copy()
        opened = hand.finger_targets(0)
        if self.config.paired_finger_insertion:
            # A splayed thumb can extend into a side panel during opening even
            # though the retained grasp fits. Search native folded-thumb goals;
            # task/contact/collision acceptance is unchanged.
            endpoint = hand.predict(
                hand.last_target,
                allowed_object=product.name,
                alternatives=False,
                grip=grip,
                carrying=False,
                check_path=False,
            )
            self.events[-1]["opening_candidates"] = []
            held_thumb = tuple(
                float(env.data.qpos[int(env.model.joint(f"Left Thumb {joint}").qposadr[0])])
                for joint in ("PIP", "DIP")
            )
            # Search partial extension before curling through the carton. A
            # posture must clear both the wall and the subsequent departure;
            # holding a hooked thumb alone can block withdrawal.
            for pip, dip in (
                (None, None),
                (-0.4, -0.4),
                (-0.2, -0.2),
                (-0.6, -0.6),
                held_thumb,
                (0.0, 0.0),
                (0.5, 0.5),
                (1.0, 1.0),
                (1.5, 1.5),
            ):
                candidate = dict(opened)
                if pip is not None:
                    candidate.update({"Left Thumb PIP": pip, "Left Thumb DIP": dip})
                candidate = {
                    name: float(
                        np.clip(
                            value,
                            max(
                                env.model.actuator(name).ctrlrange[0],
                                env.model.joint(name).range[0],
                            ),
                            min(
                                env.model.actuator(name).ctrlrange[1],
                                env.model.joint(name).range[1],
                            ),
                        )
                    )
                    for name, value in candidate.items()
                }
                errors = hand.check_hand_step(endpoint, 0, product.name, targets=candidate)
                retained = {}
                if errors and getattr(self, "front_lead_y_m", 0.0):
                    retained = wall_digit_holds(candidate, errors)
                    candidate.update(retained)
                    errors = hand.check_hand_step(endpoint, 0, product.name, targets=candidate)
                departure = None
                if not errors:
                    previous_overrides = hand.open_posture_overrides
                    try:
                        hand.open_posture_overrides = {
                            name: value
                            for name, value in candidate.items()
                            if "Thumb" in name or name in retained
                        }
                        departure = hand.predict(
                            hand.last_target + [0.02, 0, 0],
                            alternatives=False,
                            grip=0,
                            carrying=False,
                            path_mode="cartesian"
                            if getattr(self, "front_lead_y_m", 0.0)
                            else "joint",
                        )
                    finally:
                        hand.open_posture_overrides = previous_overrides
                self.events[-1]["opening_candidates"].append(
                    dict(
                        thumb_pip_rad=candidate["Left Thumb PIP"],
                        thumb_dip_rad=candidate["Left Thumb DIP"],
                        rejections=errors,
                        departure_prediction=departure.report() if departure is not None else None,
                    )
                )
                if not errors and departure is not None and departure.feasible:
                    opened = candidate
                    hand.open_posture_overrides = {
                        name: value
                        for name, value in opened.items()
                        if "Thumb" in name or name in retained
                    }
                    self.events[-1]["selected_native_opening"] = opened
                    break
            else:
                raise InteractionFailure("release_front: no coupled native thumb opening")
        start = {
            name: float(
                np.clip(
                    env.data.ctrl[env.model.actuator(name).id]
                    if precise_handoff
                    else env.data.qpos[int(env.model.joint(name).qposadr[0])],
                    *env.model.actuator(name).ctrlrange,
                )
            )
            for name in opened
        }
        self.events[-1]["opening_reference_source"] = (
            "previous_applied_servo_targets" if precise_handoff else "measured_finger_angles"
        )
        for alpha in np.linspace(0, 1, int((1.2 if precise_handoff else 0.6) / env.dt)):
            tick(
                opening={
                    name: start[name] + alpha * (value - start[name])
                    for name, value in opened.items()
                }
            )
        for _ in range(int((0.4 if precise_handoff else 0.2) / env.dt)):
            tick(opening=opened)
        separated = False
        if precise_handoff and self.last_forces["hand"] > 0.2:
            # An open native finger can retain a small gravity/contact load.
            # Use the already screened departure direction, then verify the
            # same force/support gates; waiting is not evidence of release.
            self._cartesian_move(
                "release_front_separate",
                hand.last_target + [0.005, 0, 0],
                speed_m_s=0.002,
                allowed_object=product.name,
            )
            self.events[-1].update(
                maximum_separation_m=0.005,
                unchanged_hand_release_gate_N=0.2,
            )
            separated = True
        if self.last_forces[self.FRONT_SUPPORT] < 0.2 or self.last_forces["hand"] > 0.2:
            raise InteractionFailure("release_front: no unloaded support after opening")
        if env.data.body(product.name).xmat.reshape(3, 3)[2, 2] < np.cos(
            np.radians(self.config.payload_upright_degrees)
        ):
            raise InteractionFailure("release_front: carton tipped during opening")
        self.events[-1].update(
            verified_release=True,
            support_surface=self.FRONT_SUPPORT,
            placement_conditions=False,
            stationary_opening=not separated,
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
        from abstract_stack.interactions import InteractionFailure

        for relocation in range(3):
            try:
                return self._plan_pusher_at_base(precontact, end)
            except InteractionFailure:
                if not self.config.base_reposition or relocation == 2:
                    raise
            self._stage("pusher_base_plan")
            event = self.events[-1]
            attempts = []
            started = time.monotonic()
            selected_displacement = None
            for dx, dy in (
                (0, 0.03),
                (0, 0.06),
                (0, 0.10),
                (0, -0.03),
                (0.04, 0.06),
                (-0.03, 0.06),
            ):
                for settings in self._pusher_settings():
                    if time.monotonic() - started > 30:
                        break
                    clone = self.env.clone_for_planning()
                    root = clone.model.body("mabel_chassis").id
                    joint = int(clone.model.body_jntadr[root])
                    address = int(clone.model.jnt_qposadr[joint])
                    clone.data.qpos[address : address + 2] += [dx, dy]
                    mujoco.mj_forward(clone.model, clone.data)
                    hand = self._pusher_class()(clone, self.config, **settings)
                    hand.command = self.controller.command.copy()
                    prediction = hand.predict(
                        end,
                        allowed_object=self.product.name,
                        alternatives=False,
                        carrying=False,
                        sliding=True,
                        check_path=False,
                        path_mode="cartesian",
                    )
                    attempts.append(
                        dict(
                            displacement_xy_m=[dx, dy],
                            settings=settings,
                            prediction=prediction.report(),
                        )
                    )
                    if prediction.feasible and prediction.position_error_m <= 0.002:
                        approach = hand.predict(
                            precontact,
                            alternatives=False,
                            carrying=False,
                            path_mode=getattr(self, "PUSHER_APPROACH_PATH_MODE", "joint"),
                        )
                        attempts[-1]["approach"] = approach.report()
                        if not approach.feasible:
                            continue
                        self._planning_pose(hand, approach)
                        continuation = hand.predict(
                            end,
                            allowed_object=self.product.name,
                            alternatives=False,
                            carrying=False,
                            sliding=True,
                            path_mode="cartesian",
                        )
                        attempts[-1]["continuation"] = continuation.report()
                        if not continuation.feasible or continuation.position_error_m > 0.002:
                            continue
                        clearance = self._clearance_margin(hand, continuation, precontact, end)
                        attempts[-1]["sampled_clearance_m"] = clearance
                        if clearance["support"] < 0.012 or clearance["sides"] < 0.004:
                            continue
                        reserve = self._pusher_feedback_reserve(hand, end)
                        attempts[-1]["feedback_reserve"] = reserve
                        if not reserve["admitted"]:
                            continue
                        selected_displacement = [dx, dy]
                        break
                if selected_displacement is not None:
                    break
                if time.monotonic() - started > 30:
                    break
            if selected_displacement is None:
                event.update(candidates=attempts, decision="bounded_base_search_exhausted")
                raise InteractionFailure("pusher_plan: bounded base/posture search exhausted")
            event.update(
                candidates=attempts,
                decision="guarded_empty_base_move_then_replan",
                selected_displacement_xy_m=selected_displacement,
                scope="nominal approach, insertion and clearance forecast; "
                "recheck after native move",
            )
            self._move_base(selected_displacement, reason="selected_finger_insertion_reachability")

    def _pusher_feedback_reserve(self, hand, end):
        """Necessary endpoint screen for the paired controller's pose corridor."""
        report = dict(admitted=True, endpoints=[])
        if not getattr(hand, "fingers", ()):
            return report
        original = hand.active_rotation.copy()
        try:
            for yaw in (-3, 3):
                c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
                hand.active_rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ hand.ROTATION
                for dz in (0.0, 0.025):
                    candidate = hand.predict(
                        end + [0, 0, dz],
                        allowed_object=self.product.name,
                        alternatives=False,
                        check_path=False,
                        carrying=False,
                        sliding=True,
                    )
                    report["endpoints"].append(
                        dict(yaw_bias_deg=yaw, height_bias_m=dz, prediction=candidate.report())
                    )
                    if (
                        not candidate.feasible
                        or candidate.position_error_m > 0.002
                        or candidate.orientation_error_rad > np.radians(1)
                    ):
                        report["admitted"] = False
                        return report
        finally:
            hand.active_rotation = original
        report["scope"] = (
            "endpoint necessary condition; every live correction still needs path admission"
        )
        return report

    def _plan_pusher_at_base(self, precontact, end):
        from abstract_stack.interactions import InteractionFailure

        self._stage("pusher_plan")
        event = self.events[-1]
        attempts = []
        started = time.monotonic()
        for settings in self._pusher_settings():
            if time.monotonic() - started > 45:
                event.update(candidates=attempts, planning_wall_seconds=time.monotonic() - started)
                raise InteractionFailure("pusher_plan: bounded planning budget exhausted")
            for raise_first in (False, True):
                clone = self.env.clone_for_planning()
                hand = self._pusher_class()(clone, self.config, **settings)
                hand.command = self.controller.command.copy()
                waypoints = ([precontact + [0, 0, 0.08]] if raise_first else []) + [precontact]
                attempt = {**settings, "raise_first": raise_first}
                attempts.append(attempt)
                for point in waypoints:
                    prediction = hand.predict(
                        point,
                        alternatives=False,
                        carrying=False,
                        path_mode=getattr(self, "PUSHER_APPROACH_PATH_MODE", "joint"),
                    )
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
                    if not prediction.feasible or (
                        self.config.ungrasped_insertion and prediction.position_error_m > 0.002
                    ):
                        attempt.update(rejected="insertion", prediction=prediction.report())
                        continue
                    margins = self._clearance_margin(hand, prediction, precontact, end)
                    attempt["sampled_clearance_m"] = margins
                    if margins["support"] < 0.012 or margins["sides"] < 0.004:
                        attempt["rejected"] = "tracking_clearance_margin"
                        continue
                    reserve = self._pusher_feedback_reserve(hand, end)
                    attempt["feedback_reserve"] = reserve
                    if not reserve["admitted"]:
                        attempt["rejected"] = "feedback_endpoint_reserve"
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

    def _partial_push_state(self, stop_x, *, unloaded=False):
        """Measured intermediate gate; this never grants final task credit."""
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.env.model,
            self.env.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.env.model.body(self.product.name).id,
            velocity,
            0,
        )
        body = self.env.data.body(self.product.name)
        report = dict(
            position_error_x_m=float(body.xpos[0] - stop_x),
            linear_speed_m_s=float(np.linalg.norm(velocity[3:])),
            upright=bool(body.xmat.reshape(3, 3)[2, 2] >= np.cos(np.radians(15))),
            shelf_support_N=float(self.last_forces["shelf"]),
            hand_contact_N=float(self.last_forces["hand"]),
        )
        report["accepted"] = (
            abs(report["position_error_x_m"]) <= 0.005
            and report["linear_speed_m_s"] <= 0.005
            and report["upright"]
            and report["shelf_support_N"] > 0.2
            and (not unloaded or report["hand_contact_N"] <= 0.2)
        )
        return report

    def _replan_limited_pusher(self, target, end, event):
        """Search wrist postures on independent data before changing a reference."""
        hand = self.controller
        attempt = dict(sim_time=float(self.env.data.time), candidates=[])
        event.setdefault("limited_progress_replans", []).append(attempt)
        for wrist in getattr(type(hand), "WRISTS", (0.0, 0.4, 0.6, 0.2)):
            clone = type(hand)(
                self.env.clone_for_planning(),
                self.config,
                finger=hand.finger,
                tilt_degrees=hand.tilt_degrees,
                wrist_posture=wrist,
            )
            clone.command = hand.command.copy()
            clone.OFFSET = hand.OFFSET.copy()
            clone.ik.offset = hand.ik.offset.copy()
            clone.active_rotation = hand.active_rotation.copy()
            point = clone.predict(
                target,
                allowed_object=self.product.name,
                alternatives=False,
                carrying=False,
                sliding=True,
                path_mode="cartesian",
            )
            row = dict(wrist_posture_rad=wrist, next_prediction=point.report())
            attempt["candidates"].append(row)
            if (
                not point.feasible
                or point.position_error_m > 0.002
                or point.orientation_error_rad > np.radians(1)
            ):
                continue
            self._planning_pose(clone, point)
            goal = end.copy()
            goal[2] = target[2]
            continuation = clone.predict(
                goal,
                allowed_object=self.product.name,
                alternatives=False,
                carrying=False,
                sliding=True,
                path_mode="cartesian",
            )
            row["continuation"] = continuation.report()
            if (
                not continuation.feasible
                or continuation.position_error_m > 0.002
                or continuation.orientation_error_rad > np.radians(1)
            ):
                continue
            margin = self._clearance_margin(clone, continuation, target, goal)
            row["sampled_clearance_m"] = margin
            if margin["support"] < 0.012 or margin["sides"] < 0.004:
                continue
            hand.ik.constrain_posture("Left Wrist", wrist)
            attempt.update(decision="checked_wrist_reference_replan", selected_wrist_rad=wrist)
            return True
        attempt["decision"] = "no_checked_wrist_continuation"
        return False

    def _settle_pusher_preshape(self):
        """Unfold selected fingers in free space before admitting approach."""
        from abstract_stack.interactions import InteractionFailure

        hand = self.controller
        self._stage("pusher_preshape")
        event = self.events[-1]
        event.update(maximum_seconds=4.5, joint_error_goal_rad=0.03, velocity_goal_rad_s=0.1)
        nominal = hand.command.copy()
        held = 0.0
        for _ in range(int(4.5 / self.env.dt)):
            self._apply_tick(hand.joint_targets(nominal, 0))
            goals = hand.finger_targets(0)
            errors, velocities = [], []
            for name, value in goals.items():
                joint = self.env.model.joint(name)
                errors.append(abs(float(self.env.data.qpos[int(joint.qposadr[0])]) - value))
                velocities.append(abs(float(self.env.data.qvel[int(joint.dofadr[0])])))
            if self.last_forces["robot_product"] > 0.2:
                raise InteractionFailure("pusher_preshape: unexpected product contact")
            ready = max(errors) <= 0.03 and max(velocities) <= 0.1
            held = held + self.env.dt if ready else 0.0
            event.update(
                maximum_joint_error_rad=max(errors), maximum_joint_velocity_rad_s=max(velocities)
            )
            if held >= 0.1:
                event["measured_preshape_verified"] = True
                return
        raise InteractionFailure("pusher_preshape: measured native hand did not settle")

    def _insert_from_front(
        self, *, partial_stop_x=None, front_squaring=False, retreat=True, square_in_place=False
    ):
        from abstract_stack.interactions import InteractionFailure

        env, config, product = self.env, self.config, self.product
        body = env.data.body(product.name)
        extent = np.abs(body.xmat.reshape(3, 3)) @ np.asarray(product.half_size)
        square_row_edge = None
        if square_in_place:
            if partial_stop_x is None or not front_squaring:
                raise ValueError("A short square requires an intermediate stop")
            occupied_edges = []
            for other in env.products:
                if other.name in self.observer.completed and other.level == product.level:
                    occupied = env.data.body(other.name)
                    occupied_extent = np.abs(occupied.xmat.reshape(3, 3)) @ other.half_size
                    occupied_edges.append(float(occupied.xpos[0] + occupied_extent[0]))
            if not occupied_edges or partial_stop_x - extent[0] - max(occupied_edges) < 0.01:
                raise InteractionFailure("front_square: too little depth before the occupied row")
            square_row_edge = max(occupied_edges)
        if (
            (config.wall_packing or config.center_last)
            and partial_stop_x is None
            and self.observer.completed
        ):
            yaw = float(np.arctan2(body.xmat.reshape(3, 3)[1, 0], body.xmat.reshape(3, 3)[0, 0]))
            if abs(yaw) > np.radians(0.15):
                stops = []
                square_clearance = 0.016 if config.live_guarded_withdrawal else 0.008
                for other in env.products:
                    if other.name in self.observer.completed and other.level == product.level:
                        occupied = env.data.body(other.name)
                        occupied_extent = np.abs(occupied.xmat.reshape(3, 3)) @ other.half_size
                        stops.append(
                            float(
                                occupied.xpos[0] + occupied_extent[0] + extent[0] + square_clearance
                            )
                        )
                if stops and body.xpos[0] - max(stops) > 0.015:
                    self._stage("front_square_plan")
                    self.events[-1].update(
                        stop_object_x_m=max(stops),
                        measured_yaw_deg=float(np.degrees(yaw)),
                        decision="square_before_overlap_with_occupied_row",
                        fresh_task_credit=0,
                    )
                    self._insert_from_front(partial_stop_x=max(stops), front_squaring=True)
                    body = env.data.body(product.name)
                    extent = np.abs(body.xmat.reshape(3, 3)) @ np.asarray(product.half_size)
        height = env.config.levels[product.level] + 0.028
        precontact = np.array([body.xpos[0] + extent[0] + 0.02, body.xpos[1], height])
        end = np.array(
            [
                env.config.front_x - env.config.depth + 2 * extent[0] - 0.0005,
                self.lateral_goal_y if config.center_last else product.target[1],
                height,
            ]
        )
        if partial_stop_x is not None:
            full_center_x = end[0] - extent[0]
            if not np.isfinite(partial_stop_x) or not (
                full_center_x + 0.02
                < partial_stop_x
                < body.xpos[0] - (0.002 if square_in_place else 0.01)
            ):
                raise ValueError(
                    "Partial push must stop strictly between entry and final placement"
                )
            end[0] = partial_stop_x + extent[0]
        guided_wall_entry = config.guided_wall_entry and bool(self.observer.completed)
        clearance_entry = guided_wall_entry or getattr(self, "clearance_before_square", False)
        clearance_before_square = front_squaring and clearance_entry
        guide_deg = (
            getattr(self, "guided_entry_deg", 3 if guided_wall_entry else 0)
            if partial_stop_x is None or clearance_before_square
            else 0
        )
        screen_end = end.copy()
        if guide_deg and not clearance_before_square:
            # The fixed-Y slide model screens the first short advance only.
            # The live loop below still checks every corrected reference.
            screen_end[0] = max(end[0], float(body.xpos[0] + extent[0] - 0.012))
            screen_end[1] = float(body.xpos[1])
        settings, waypoints = self._plan_pusher(precontact, screen_end)
        self._change_hand(self._pusher_class(), **settings)
        hand = self.controller
        if self.config.ungrasped_insertion:
            if square_in_place and getattr(hand, "fingers", ()):
                hand.extend_in_wide_posture = True
                self._settle_pusher_preshape()
                hand.extend_in_wide_posture = False
            self._settle_pusher_preshape()
        for point in waypoints:
            self._move("pusher_approach", point, carrying=False, orientation_alternatives=False)
        self._stage("finger_push")
        self._push_neighbor_origins = (
            {
                item["product"]: env.data.body(item["product"]).xpos.copy()
                for item in self._adjacent_placed_neighbors()
            }
            if config.ungrasped_insertion and partial_stop_x is None
            else {}
        )
        self.events[-1]["admitted_carton_contact_neighbors"] = sorted(self._push_neighbor_origins)
        push_goal_y = self.lateral_goal_y
        center_slot = False
        if config.center_last and partial_stop_x is None:
            from abstract_stack.packing import is_two_sided_slot, open_slot_interval

            center_slot = is_two_sided_slot(env, product, self.observer.completed)

        diagonal_goal_y = (
            float(body.xpos[1]) - 0.006
            if guide_deg and getattr(self, "diagonal_entry", False) and not clearance_entry
            else None
        )
        if clearance_before_square:
            # Use the same measured-neighbor registration goal as the final
            # insertion. Reserve an extra millimetre while the corner squares.
            diagonal_goal_y = push_goal_y - 0.001
            if diagonal_goal_y < product.target[1] - 0.018:
                raise InteractionFailure(
                    "front_square: lateral clearance exceeds bounded reference"
                )
        required_square_y = diagonal_goal_y if clearance_before_square else None
        if diagonal_goal_y is not None:
            self.events[-1]["diagonal_entry_goal_y_m"] = diagonal_goal_y
        self.events[-1]["push_goal_y_m"] = push_goal_y
        # Preserve the nominal reference after approach. Recapturing the sagged
        # measured tip would ratchet the reference down before force feedback.
        target = hand.last_target.copy()
        origin = target.copy()
        prediction = hand.predict(
            screen_end,
            allowed_object=product.name,
            alternatives=False,
            carrying=False,
            sliding=True,
            path_mode="cartesian",
        )
        event = self.events[-1]
        event["partial_stop_object_x_m"] = partial_stop_x
        if guide_deg:
            event.update(
                guided_entry_deg=guide_deg,
                screened_prefix_end_m=screen_end.tolist(),
                nominal_final_tool_end_m=end.tolist(),
            )
        if partial_stop_x is not None:
            event.update(
                partial_position_tolerance_m=0.005,
                partial_speed_limit_m_s=0.005,
                partial_dwell_seconds=config.contact_dwell,
            )
        event.update(
            ik_prediction=prediction.report(),
            push_origin_m=origin.tolist(),
            maximum_advance_m=env.config.depth + 0.15,
            timeout_seconds=45.0,
            active_push_budget_seconds=45.0,
            maximum_recovery_seconds=4.5 if getattr(self, "contact_recovery", False) else 0.0,
            maximum_combined_control_seconds=(
                49.5 if getattr(self, "contact_recovery", False) else 45.0
            ),
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
        paired = bool(getattr(hand, "fingers", ()))
        plane = hand.active_rotation @ hand.ROTATION.T
        plane_bias = float(
            np.clip(np.arctan2(plane[1, 0], plane[0, 0]), -np.radians(3), np.radians(3))
        )
        event["lateral_feedback_reference"] = (
            "paired tool midpoint, bounded inside measured carton face"
            if paired
            else "measured single-finger contact center"
        )
        yaw_integral_lever = 0.0
        lateral_integral = 0.0
        object_velocity = np.zeros(6)
        limited_seconds = 0.0
        last_limit_replan = -np.inf
        progress = ContactProgress((-1, 0, 0))
        watch_progress = config.ungrasped_insertion and partial_stop_x is None
        push_force_target = config.push_force_target
        if getattr(self, "contact_recovery", False):
            self._start_free_push()
        for _ in range(int(45.0 / env.dt)):
            if getattr(self, "contact_recovery", False) and self._free_push_loss_seconds >= 0.04:
                target = self._reacquire_push_contact(target, origin, height, event)
                yaw_integral_lever = 0.0
                continue
            body = env.data.body(product.name)
            rotation = body.xmat.reshape(3, 3)
            yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
            lateral_extent = float((np.abs(rotation) @ product.half_size)[1])
            square_clearance_y = (
                required_square_y - max(0.0, lateral_extent - product.half_size[1])
                if required_square_y is not None
                else None
            )
            if paired:
                measured_plane = hand.contact_plane_yaw()
                remaining_depth = float(
                    body.xpos[0] - extent[0] - (env.config.front_x - env.config.depth)
                )
                guide_weight = float(np.clip((remaining_depth - 0.01) / 0.04, 0, 1))
                plane_goal = float(
                    np.clip(
                        np.radians(guide_deg) * guide_weight - yaw, -np.radians(3), np.radians(3)
                    )
                )
                if square_clearance_y is not None:
                    # When clearance is lost, steer toward the open side again.
                    # Squaring torque resumes only with measured room for the corner.
                    plane_goal = (
                        abs(np.radians(guide_deg)) if body.xpos[1] > square_clearance_y else -yaw
                    )
                    plane_goal = float(np.clip(plane_goal, -np.radians(3), np.radians(3)))
                if square_in_place:
                    # Depth is bounded ahead of the row. Use the paired
                    # contact plane for squaring here, even if lateral
                    # clearance is not yet admitted.
                    plane_goal = float(np.clip(-2 * yaw, -np.radians(3), np.radians(3)))
                plane_bias = float(
                    np.clip(
                        plane_bias - np.clip(2 * (measured_plane - plane_goal), -0.3, 0.3) * env.dt,
                        -np.radians(3),
                        np.radians(3),
                    )
                )
                c, s = np.cos(plane_bias), np.sin(plane_bias)
                hand.active_rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ hand.ROTATION
                event["paired_plane_yaw_deg"] = float(np.degrees(measured_plane))
                event["paired_plane_reference_bias_deg"] = float(np.degrees(plane_bias))
            if rotation[2, 2] < np.cos(np.radians(15)):
                raise InteractionFailure("finger_push: carton tilting; replan required")
            extent_x = float((np.abs(rotation) @ product.half_size)[0])
            if square_row_edge is not None and body.xpos[0] - extent_x < square_row_edge + 0.005:
                raise InteractionFailure("front_square: short square approached the occupied row")
            backstop_gap = float(body.xpos[0] - extent_x - (env.config.front_x - env.config.depth))
            finishing_push = 0 < backstop_gap <= 0.035
            advance_goal_y = push_goal_y
            if (
                config.live_guarded_withdrawal
                and config.wall_packing
                and paired
                and self.observer.completed
            ):
                # Use the unoccupied side during front squaring and forward advance. Blend back to
                # the measured registration target over the last 12 mm.
                clearance = 0.004 * float(np.clip((backstop_gap - 0.003) / 0.012, 0, 1))
                advance_goal_y -= clearance
                event["advance_lateral_clearance_m"] = clearance
            separating = diagonal_goal_y is not None and body.xpos[1] > diagonal_goal_y
            if separating:
                advance_goal_y = diagonal_goal_y - 0.002
            elif diagonal_goal_y is not None:
                event["diagonal_entry_measured_y_m"] = float(body.xpos[1])
                diagonal_goal_y = None
            if square_clearance_y is not None:
                separating = body.xpos[1] > square_clearance_y
                advance_goal_y = square_clearance_y - 0.002
                event["measured_corner_center_limit_y_m"] = square_clearance_y
            if square_in_place and abs(yaw) > np.radians(0.15):
                advance_goal_y = float(body.xpos[1])
                lateral_integral = 0.0
            if watch_progress and (finishing_push or config.wall_packing and paired):
                now = float(env.data.time)
                push_force = self.last_forces["push_force"]
                if self.last_forces["back"] >= config.contact_threshold:
                    progress.reset()
                else:
                    measurement = progress.observe(body.xpos, now, contact=push_force >= 1.0)
                    if measurement is None:
                        measurement = (0.0, 0.0)
                    stalled_s, advance_m = measurement
                    next_target = next(
                        (
                            target
                            for target in (3.5, 4.5)
                            if push_force_target < target < config.max_force
                        ),
                        None,
                    )
                    if stalled_s >= 1 and push_force >= 1.0 and next_target is not None:
                        recovery = dict(
                            sim_time=now,
                            object_x_m=float(body.xpos[0]),
                            previous_target_N=push_force_target,
                            new_target_N=next_target,
                            progress_threshold_m=0.001,
                            progress_window_s=1.0,
                            backstop_gap_m=backstop_gap,
                            mode="finishing_push" if finishing_push else "insertion_breakaway",
                        )
                        event.setdefault("stall_breakaway_steps", []).append(recovery)
                        if "stall_breakaway" not in event:
                            event["stall_breakaway"] = recovery
                            if config.record_skill_starts:
                                record_skill_entry(self, "loaded_push_stall", case=recovery)
                        push_force_target = next_target
                        progress.reset()
                        progress.observe(body.xpos, now, contact=True)
                    elif stalled_s >= 3:
                        gap = backstop_gap
                        event["stall_abort"] = dict(
                            sim_time=now,
                            stalled_seconds=stalled_s,
                            advance_m=advance_m,
                            backstop_gap_m=gap,
                            measured_push_force_N=push_force,
                            push_force_target_N=push_force_target,
                            backstop_normal_N=self.last_forces["back"],
                            breakaway_attempted="stall_breakaway" in event,
                        )
                        if config.record_skill_starts:
                            record_skill_entry(
                                self, "push_recovery_exhausted", case=event["stall_abort"]
                            )
                        raise InteractionFailure(
                            f"finger_push: {product.name} stalled {gap * 1000:.0f} mm "
                            "short of backstop; retreat not started"
                        )
            if self.last_forces["back"] >= config.contact_threshold:
                scale = np.clip(
                    (config.insertion_back_force_target - self.last_forces["back"])
                    / config.insertion_back_force_target,
                    -1,
                    1,
                )
                speed = 0.01
            else:
                scale = np.clip(
                    (push_force_target - self.last_forces["push_force"]) / push_force_target,
                    -1,
                    1,
                )
                speed = 0.04
            if getattr(self, "contact_recovery", False) and scale > 0:
                # The measured wrench is ON the carton. Downward finger load
                # raises board friction and can pivot a tall carton before it
                # slides. Unload with the existing vertical admittance before
                # advancing again; reverse/unloading motion remains available.
                advance_fraction = self._ungrasped_forward_fraction()
                scale *= advance_fraction
                event["forward_unload_downforce_N"] = self.FORWARD_UNLOAD_DOWNFORCE_N
                if advance_fraction < 1:
                    event["forward_unload_updates"] = event.get("forward_unload_updates", 0) + 1
            if partial_stop_x is not None and scale > 0:
                # Brake from measured object progress before the intermediate
                # stop; do not keep loading the carton after crossing it.
                scale *= float(np.clip((body.xpos[0] - partial_stop_x) / 0.015, 0, 1))
            if separating and scale > 0:
                speed = min(speed, 0.004)
            if center_slot:
                speed = min(speed, 0.012)
                lower_y, upper_y = open_slot_interval(env, product, self.observer.completed)
                advance_goal_y = (lower_y + upper_y) / 2
                overhang = max(0.0, lateral_extent - product.half_size[1])
                if not lower_y + overhang <= body.xpos[1] <= upper_y - overhang:
                    scale = min(0.0, scale)
                    event["center_slot_forward_holds"] = (
                        event.get("center_slot_forward_holds", 0) + 1
                    )
            target[0] -= speed * env.dt * scale
            # Use measured yaw/slot error while the carton is still moving.
            # A centered single-point push alone cannot regulate yaw.
            contact = self.last_forces.get("hand_contact_position_world_m")
            # Torque depends on the contact lever relative to the MEASURED
            # carton center. A lever based only on the canonical slot changes
            # sign after lateral drift and can amplify yaw during a jam.
            desired_lever = 0.0 if paired else -config.insertion_yaw_lever_gain * yaw
            if (
                not paired
                and getattr(self, "square_backstop", False)
                and self.last_forces["back"] >= config.contact_threshold
            ):
                mujoco.mj_objectVelocity(
                    env.model,
                    env.data,
                    mujoco.mjtObj.mjOBJ_BODY,
                    env.model.body(product.name).id,
                    object_velocity,
                    0,
                )
                yaw_integral_lever -= 0.03 * yaw * env.dt
                desired_lever += yaw_integral_lever - 0.01 * object_velocity[2]
                desired_lever = float(np.clip(desired_lever, -0.014, 0.014))
                yaw_integral_lever = (
                    desired_lever
                    + config.insertion_yaw_lever_gain * yaw
                    + 0.01 * object_velocity[2]
                )
                event["yaw_integral_lever_m"] = yaw_integral_lever
            desired_y = body.xpos[1] + np.clip(desired_lever, -0.014, 0.014)
            measured_y = contact[1] if contact is not None and not paired else hand.position()[1]
            force_y = self.last_forces.get("hand_force_world_N", [0, 0, 0])[1]
            position_gain = config.insertion_lateral_position_gain
            if (
                not paired
                and getattr(self, "square_backstop", False)
                and self.last_forces["back"] >= config.contact_threshold
            ):
                # Use the backstop to square the box before resuming lateral
                # positioning; competing translation feedback can reverse the
                # yaw-correcting contact lever while the box is braced.
                position_gain = 0.0
                event["backstop_alignment_uses_yaw_priority"] = True
            lateral_speed = (
                2 * (desired_y - measured_y)
                + position_gain * (advance_goal_y - body.xpos[1])
                - config.insertion_tangent_admittance * force_y
            )
            if paired:
                lateral_integral = float(
                    np.clip(
                        lateral_integral + 2 * (advance_goal_y - body.xpos[1]) * env.dt,
                        -0.006,
                        0.006,
                    )
                )
                lateral_speed += 2 * lateral_integral
                event["lateral_integral_reference_bias_m"] = lateral_integral
            target[1] += np.clip(lateral_speed * env.dt, -0.01 * env.dt, 0.01 * env.dt)
            if paired:
                # A two-pad face squares by its plane. Chasing its instantaneous
                # center of pressure overhangs the carton when one pad unloads.
                # Keep the physical tool centered; canonical registration still
                # acts through the bounded lateral velocity above.
                target[1] = np.clip(target[1], body.xpos[1] - 0.006, body.xpos[1] + 0.006)
                if config.wall_packing and partial_stop_x is None and self.observer.completed:
                    # Preserve pad clearance on the occupied side during advance.
                    # Squaring and final registration are checked separately at the backstop.
                    target[1] = min(target[1], body.xpos[1] - 0.001)
            measured_height = contact[2] if contact is not None else hand.position()[2]
            force_z = self.last_forces.get("hand_force_world_N", [0, 0, 0])[2]
            vertical_speed = -0.012 * force_z + 0.2 * (height - measured_height)
            target[2] += np.clip(vertical_speed * env.dt, -0.015 * env.dt, 0.015 * env.dt)
            if getattr(self, "bounded_contact_ik", False):
                lower = np.array(
                    [
                        origin[0] - env.config.depth - 0.15,
                        product.target[1] - 0.02,
                        height - 0.005,
                    ]
                )
                upper = np.array([origin[0] + 0.03, product.target[1] + 0.02, height + 0.025])
                bounded = np.clip(target, lower + 1e-6, upper - 1e-6)
                if not np.array_equal(bounded, target):
                    event["reference_saturation_updates"] = (
                        event.get("reference_saturation_updates", 0) + 1
                    )
                target = bounded
            if (
                not origin[0] - env.config.depth - 0.15 <= target[0] <= origin[0] + 0.03
                or abs(target[1] - product.target[1]) > 0.02
                or not height - 0.005 <= target[2] <= height + 0.025
            ):
                raise InteractionFailure("finger_push: corrected target outside bounded motion")
            prediction = self._contact_prediction(
                target,
                allowed_object=product.name,
                alternatives=False,
                carrying=False,
                sliding=True,
            )
            feedback = getattr(prediction, "bounded_feedback", None) or {}
            limited = (
                paired
                and scale > 0
                and feedback.get("active_joint_limits")
                and feedback.get("orientation_error_deg", 0) > 0.5
                and abs(prediction.position[0] - hand.ik.forward(hand.command)[0][0]) < 0.00005
            )
            limited_seconds = limited_seconds + env.dt if limited else 0.0
            if limited_seconds >= 0.2 and env.data.time - last_limit_replan >= 1.0:
                if len(event.get("limited_progress_replans", [])) >= 4:
                    raise InteractionFailure("finger_push: bounded wrist-replan budget exhausted")
                if not self._replan_limited_pusher(target, end, event):
                    raise InteractionFailure(
                        "finger_push: joint-limited push needs a new base pose"
                    )
                last_limit_replan = float(env.data.time)
                limited_seconds = 0.0
                prediction = self._contact_prediction(
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
                for wrist in getattr(type(hand), "WRISTS", (0.2, 0.4, 0.0, 0.6)):
                    if wrist == original:
                        continue
                    hand.ik.constrain_posture("Left Wrist", wrist)
                    candidate = self._contact_prediction(
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
            if getattr(self, "bounded_contact_ik", False):
                requested = target.copy()
                # Anti-windup projects onto the commanded nominal FK, never
                # the sagged measured tool pose. Recheck all motion bounds.
                target = prediction.position.copy()
                if (
                    not origin[0] - env.config.depth - 0.15 <= target[0] <= origin[0] + 0.03
                    or abs(target[1] - product.target[1]) > 0.02
                    or not height - 0.005 <= target[2] <= height + 0.025
                ):
                    raise InteractionFailure(
                        "finger_push: projected nominal reference outside bounds"
                    )
                event.setdefault("nominal_projection_samples", []).append(
                    [float(env.data.time), *requested, *target, prediction.position_error_m]
                )
            hand.last_target = target.copy()
            action = hand.joint_targets(prediction.q, 0)
            if getattr(self, "bounded_contact_ik", False) and not np.allclose(
                hand.command, prediction.q, atol=1e-9, rtol=0
            ):
                raise InteractionFailure("finger_push: unexpected second nominal rate projection")
            self._apply_tick(action)
            if getattr(self, "bounded_contact_ik", False):
                tracked_nominal = np.array(
                    [self.tracker.last_report["nominal"][name] for name in [*hand.names, "Lift"]]
                )
                if not np.allclose(tracked_nominal, hand.command, atol=1e-9, rtol=0):
                    raise InteractionFailure(
                        "finger_push: hand and joint-tracker nominal references diverged"
                    )
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
            if partial_stop_x is not None:
                partial_state = self._partial_push_state(partial_stop_x)
                if front_squaring:
                    partial_state["yaw_deg"] = float(np.degrees(yaw))
                    partial_state["accepted"] = bool(
                        partial_state["accepted"]
                        and (required_square_y is None or body.xpos[1] <= required_square_y)
                        and abs(yaw) <= np.radians(0.15)
                        and rotation[2, 2] >= np.cos(np.radians(config.payload_upright_degrees))
                    )
                event["partial_state"] = partial_state
                held = held + env.dt if partial_state["accepted"] else 0.0
                if held >= config.contact_dwell:
                    event["verified_partial_push"] = True
                    break
                continue
            contacted = (
                self.last_forces["back"] >= config.contact_threshold
                and self.last_forces["hand"] > 0.2
                and self.last_forces["shelf"] > 0.2
                and rotation[2, 2] >= np.cos(np.radians(config.payload_upright_degrees))
            )
            extent_x = float((np.abs(rotation) @ product.half_size)[0])
            back_gap = float(body.xpos[0] - extent_x - (env.config.front_x - env.config.depth))
            staging_for_refinement = (
                paired
                and self.config.ungrasped_insertion
                and back_gap < 0.003
                and self.last_forces["hand"] > 0.2
                and self.last_forces["shelf"] > 0.2
            )
            if getattr(self, "square_backstop", False):
                event["requested_backstop_yaw_tolerance_deg"] = 0.15
                if self.config.ungrasped_insertion:
                    event["contact_refinement_owns_squared_dwell"] = True
                else:
                    contacted = contacted and abs(yaw) <= np.radians(0.15)
            held = held + env.dt if contacted or staging_for_refinement else 0.0
            if held >= config.contact_dwell:
                if self.config.ungrasped_insertion:
                    event["first_back_contact_verified"] = bool(contacted)
                    event["bounded_backstop_staging_for_native_refinement"] = bool(
                        staging_for_refinement
                    )
                else:
                    event["back_contact_verified"] = True
                break
        else:
            raise InteractionFailure("finger_push: contact dwell timeout")
        self.supported_slide = False
        self._active_push_event = event
        if not retreat and partial_stop_x is None and not front_squaring:
            # Record the native handoff without conflating staging with contact.
            event["contact_established_pending_retreat"] = bool(
                event.get("first_back_contact_verified") or event.get("back_contact_verified")
            )
            return
        if front_squaring:
            self._cartesian_move(
                "front_square_clear",
                hand.last_target + [0.04, 0, 0],
                speed_m_s=0.01,
                allowed_object=product.name,
            )
            final_partial = self._partial_push_state(partial_stop_x, unloaded=True)
            body = env.data.body(product.name)
            if square_in_place and abs(np.arctan2(body.xmat[3], body.xmat[0])) > np.radians(0.15):
                raise InteractionFailure("front_square: yaw did not survive unloading")
            if not final_partial["accepted"] or (
                required_square_y is not None and body.xpos[1] > required_square_y
            ):
                raise InteractionFailure("front_square: support/clearance verification failed")
            event.update(
                verified_front_squaring=True,
                partial_state_after_retreat=final_partial,
                fresh_task_credit=0,
            )
            self._stage("front_squaring_complete")
            return
        if self.config.ungrasped_insertion and partial_stop_x is None:
            from abstract_stack.contact_execution import squared_contact_retreat

            squared_contact_retreat(self, event)
            return
        self.observer.begin_retreat(product.name)
        self._release_grasp()
        self._clear_pusher(hand)
        self.observer.end_retreat(product.name)
        if partial_stop_x is not None:
            final_partial = self._partial_push_state(partial_stop_x, unloaded=True)
            event["partial_state_after_retreat"] = final_partial
            if not final_partial["accepted"]:
                raise InteractionFailure(
                    "partial_push: intermediate state not settled after retreat"
                )
            event["verified_partial_support_after_retreat"] = True

    def _clear_pusher(self, hand):
        self._move("clear", hand.position() + [0.04, 0, 0.02], orientation_alternatives=False)

    def _contact_prediction(self, target, **kwargs):
        """Hook for an opt-in rate-constrained online Cartesian controller."""
        proposal = self.controller.predict(target, **kwargs)
        if not self.config.ungrasped_insertion:
            return proposal
        from abstract_stack.cartesian_feedback import bounded_contact_prediction

        hand = self.controller
        rates = np.r_[np.full(len(hand.names), self.config.max_joint_speed), 0.08]
        result = bounded_contact_prediction(hand, proposal, target, dt=self.env.dt, rates=rates)
        hand.last_prediction = result
        hand.last_ik_error = float(
            np.linalg.norm([result.position_error_m, 0.15 * result.orientation_error_rad])
        )
        self.events[-1].setdefault("bounded_feedback_samples", []).append(
            dict(
                sim_time=float(self.env.data.time),
                feasible=bool(result.feasible),
                **result.bounded_feedback,
            )
        )
        return result
