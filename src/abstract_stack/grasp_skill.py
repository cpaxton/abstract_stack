"""An isolated actuator-driven grasp/lift/hold task using the same stocking skill."""

import copy
import hashlib
import json

import numpy as np

from abstract_stack.grasping import object_aligned_rotation, open_hand_clearance
from abstract_stack.interactions import InteractionFailure, StockingScript


def select_grasp(script, pending):
    """Search bounded object-frame grasp offsets with native open-hand clearance."""
    env, hand = script.env, script.controller
    base = env.data.body("mabel_chassis").xpos
    offsets = ((0, 0), (0.004, -0.004), (0.008, -0.004), (0.004, -0.008), (0.008, -0.008))
    for product in sorted(
        pending, key=lambda p: np.linalg.norm(env.data.body(p.name).xpos[:2] - base[:2])
    ):
        script.product = product
        body = env.data.body(product.name)
        rotation = body.xmat.reshape(3, 3).copy()
        try:
            grasp_rotation, yaw = object_aligned_rotation(rotation, hand.ROTATION)
        except ValueError as error:
            script._stage("grasp_candidate")
            script.events[-1].update(decision="unsupported_grasp_entry", reason=str(error))
            continue
        for height in (product.half_size[2] / 3, 0.0, product.half_size[2] * 2 / 3):
            for x, y in offsets:
                # Offset directions follow the selected symmetric grasp frame.
                local = np.array([x, y, 0])
                turn = grasp_rotation @ hand.ROTATION.T
                target = body.xpos + [0, 0, height] + turn @ local
                hand.active_rotation = grasp_rotation.copy()
                prediction = hand.predict(
                    target, allowed_object=product.name, alternatives=False, check_path=False
                )
                clearance = open_hand_clearance(hand, prediction, product.name)
                script._stage("grasp_candidate")
                script.events[-1].update(
                    ik_prediction=prediction.report(),
                    object_aligned_yaw_deg=yaw,
                    object_frame_offset_m=[x, y, height],
                    open_hand_clearance=clearance,
                    minimum_open_clearance_m=0.002,
                    decision="try_next_candidate",
                )
                if not prediction.feasible:
                    continue
                if clearance["minimum_distance_m"] < 0.002:
                    script.events[-1]["decision"] = "reject_open_hand_object_overlap"
                    continue
                script.events[-1]["decision"] = "select_candidate"
                return target.copy(), prediction
    raise InteractionFailure("No checked object-aligned grasp candidate; no grasp motion commanded")


class GraspSkillProbe(StockingScript):
    """Pick one declared carton in the original inventory, without stocking credit."""

    TASK_SCOPE = "grasp_lift_hold"

    def __init__(self, *args, product, entry_checkpoint=None, **kwargs):
        self.entry_checkpoint = entry_checkpoint
        if entry_checkpoint is not None and (
            entry_checkpoint.controller.get("kind") != "stocking_skill_entry"
            or entry_checkpoint.context["skill_entry"]["skill"] != "grasp_lift"
            or entry_checkpoint.context["skill_entry"]["active_product"] != product
            or entry_checkpoint.controller["active_grip"] is not None
            or entry_checkpoint.controller["carry_reference"] is not None
        ):
            raise ValueError("Grasp probe requires a matching empty-hand grasp entry checkpoint")
        super().__init__(*args, **kwargs)
        self.target_product = next(p for p in self.env.products if p.name == product)
        self.grasp_skill_verified = False

    def _prepare_episode(self):
        if self.entry_checkpoint is None:
            return super()._prepare_episode()
        checkpoint = self.entry_checkpoint
        directory = self.directory / "initial-scene"
        directory.mkdir()
        scene = copy.deepcopy(checkpoint.scene)
        scene["continuation"] = dict(
            checkpoint=str(checkpoint.directory),
            manifest_sha256=hashlib.sha256(
                (checkpoint.directory / "manifest.json").read_bytes()
            ).hexdigest(),
            fresh_stocking_credit=0,
        )
        (directory / "manifest.json").write_text(json.dumps(scene, indent=2) + "\n")

    def _new_cartesian_hand(self):
        hand = super()._new_cartesian_hand()
        if self.entry_checkpoint is not None:
            state = self.entry_checkpoint.controller
            hand.command = np.array(
                [state["hand_nominal_subset"][n] for n in [*hand.names, "Lift"]]
            )
            hand.active_rotation = np.array(state["active_rotation_world"])
        return hand

    def _new_tracker(self):
        if self.entry_checkpoint is None:
            return super()._new_tracker()
        from abstract_stack.mabel_controller import MabelController, TrackingConfig

        state = self.entry_checkpoint.controller
        return MabelController(
            self.env.model,
            self.env.data,
            TrackingConfig(**state["tracking_config"]),
            nominal_references=state["nominal_subset"],
        )

    def _new_observer(self):
        observer = super()._new_observer()
        if self.entry_checkpoint is not None:
            history = self.entry_checkpoint.context["observer_history"]
            for name, value in copy.deepcopy(history).items():
                if name == "seen":
                    value = {tuple(item) for item in value}
                if name == "completed":
                    value = set(value)
                setattr(observer, name, value)
        return observer

    def _execute_task(self, result):
        selected = self._select_grasp([self.target_product])
        grip = self._grasp_and_lift(selected)
        self._stage("grasp_hold")
        target = self.controller.position().copy()
        for _ in range(int(np.ceil(0.5 / self.env.dt))):
            self._tick(target, grip)
            if (
                not self.last_forces["opposing_contacts"]
                or self.last_forces["hand"] < 0.2
                or self.last_forces["table"] > 0.15
            ):
                raise InteractionFailure(
                    "grasp_hold: loaded opposing contact or table clearance lost"
                )
        if self.observer.incidents:
            raise InteractionFailure("grasp_hold: an inventory product tipped or dropped")
        self.grasp_skill_verified = True
        self.events[-1].update(
            verified_grasp_skill=True,
            required_hold_seconds=0.5,
            measured_hand_normal_N=self.last_forces["hand"],
            measured_table_normal_N=self.last_forces["table"],
        )

    def episode_evidence(self):
        return dict(
            kind="isolated_grasp_lift_hold",
            scheduled_grasp_products=[self.target_product.name],
            grasp_skill_verified=self.grasp_skill_verified,
            full_stocking_task_success=False,
            entry_scope="fresh_reset" if self.entry_checkpoint is None else "retained_skill_entry",
            scope="One physical pickup, not transfer/placement/retreat or a whole shelf",
        )
