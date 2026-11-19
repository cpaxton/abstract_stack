"""Development sequence that inserts a measured row press between stocks."""

from dataclasses import asdict, replace

import numpy as np

from abstract_stack.contact_execution import hand_clears_shelf, shelf_exit_distance
from abstract_stack.interactions import InteractionFailure, StockingScript
from abstract_stack.lateral_contact import (
    FRONT_SIDE_BASE_OFFSET_M,
    PRESS_BASE_FOOTPRINT_RADIUS_M,
    LateralContact,
    LateralPushConfig,
)
from abstract_stack.navigation import plan_base_path


class PressInterleavedStockingScript(LateralContact, StockingScript):
    """Stock two cartons, press their exposed seam, then stock the third."""

    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        if (
            len(env.config.levels) != 1
            or len(env.products) != 3
            or env.config.product_kind != "carton"
            or env.config.row_free_space is not None
            or env.config.initially_stocked
            or not self.config.base_reposition
        ):
            raise ValueError(
                "Interleaved pressing needs a spacious single shelf, three fresh cartons, "
                "and base motion"
            )

    def _execute_task(self, result, *, next_cartons=None):
        if next_cartons is not None:
            raise ValueError("The interleaved sequence starts from a fresh episode")
        super()._execute_task(result, next_cartons=2)
        self.press_and_finish(result)

    def episode_evidence(self):
        return {**super().episode_evidence(), "interleave_row_press": True}

    def press_and_finish(self, result):
        """Run the remaining sequence from the measured two-carton boundary."""
        stocking_config = self.config
        try:
            self.config = replace(stocking_config, ungrasped_insertion=True)
            self._press_stocked_pair(result)
        finally:
            self.config = stocking_config
        self._restore_grasp_hand()
        self._settle_pusher_preshape()
        super()._execute_task(result)

    def _press_stocked_pair(self, result):
        env = self.env
        products = sorted(
            (p for p in env.products if p.name in self.observer.completed),
            key=lambda p: env.data.body(p.name).xpos[1],
        )
        if (
            len(products) != 2
            or self.observer.incidents
            or self.active_grip is not None
            or self.carry_reference is not None
        ):
            raise InteractionFailure(
                "row_press_entry: requires two verified retreats and an empty hand"
            )
        ids = tuple(sorted(env.model.body(p.name).id for p in products))

        def gap():
            bodies = [env.data.body(p.name) for p in products]
            extents = [
                (np.abs(b.xmat.reshape(3, 3)) @ p.half_size)[1]
                for b, p in zip(bodies, products, strict=True)
            ]
            return float(bodies[1].xpos[1] - bodies[0].xpos[1] - sum(extents))

        initial_gap = gap()
        report = dict(
            phase="interleaved_row_press",
            seam_before_m=initial_gap,
            pair_contact_peak_N=0.0,
            verified_staging_press=False,
            controller=asdict(self.config),
        )
        result["interleaved_row_press"] = report
        settings = LateralPushConfig(
            distance_limit_m=0.05,
            speed_m_s=0.006,
            force_target_N=12.0,
            timeout_s=26.0,
            position_tolerance_m=0.0008,
            height_m=0.018,
        )
        if initial_gap <= 0.001 or initial_gap >= settings.distance_limit_m:
            raise InteractionFailure("row_press_entry: seam outside the bounded press range")

        direction, active, neighbor, base_goal = self._side_workspace(products, report)
        self.product = active
        report.update(product=active.name, neighbor=neighbor.name, direction=direction)
        self._stage("interleaved_row_press")
        if not hand_clears_shelf(env, self.hand_bodies):
            distance = shelf_exit_distance(env, self.controller)
            self._cartesian_move(
                "row_press_clear_hand", self.controller.position() + [distance, 0, 0.04]
            )

        def pair_in_contact():
            normal = self.carton_pair_forces.get(ids, 0.0)
            report["pair_contact_peak_N"] = max(normal, report["pair_contact_peak_N"])
            return normal >= self.config.contact_threshold

        event = self.push_laterally(
            direction=direction,
            distance_m=min(initial_gap + 0.004, settings.distance_limit_m),
            near_front_corner=True,
            row_ready=pair_in_contact,
            front_contact_fraction=0.0,
            settings=settings,
            base_goal_xy=base_goal,
        )
        if not hand_clears_shelf(env, self.hand_bodies):
            distance = shelf_exit_distance(env, self.controller)
            self._cartesian_move(
                "row_press_retreat", self.controller.position() + [distance, 0, 0.04]
            )
        final_gap = gap()
        snapshot = self.observer.snapshot()
        report.update(
            seam_after_m=final_gap,
            seam_reduction_m=initial_gap - final_gap,
            pair_contact_verified=event["displacement_before_retreat_verified"],
            both_arms_unloaded=max(
                self.last_forces["robot_product"], self.last_forces["other_product"]
            )
            <= 0.2,
            placement_preserved=all(snapshot[p.name]["placement_conditions"] for p in products),
        )
        report["verified_staging_press"] = bool(
            not self.observer.incidents
            and report["both_arms_unloaded"]
            and report["placement_preserved"]
            and report["pair_contact_verified"]
            and final_gap < initial_gap - 0.001
            and hand_clears_shelf(env, self.hand_bodies)
        )
        if not report["verified_staging_press"]:
            raise InteractionFailure(
                "row_press_exit: supported compaction did not survive withdrawal"
            )
        self._stage("interleaved_row_press_complete")
        self.events[-1].update(report, phase="interleaved_row_press_complete")

    def _side_workspace(self, products, report):
        """Select an exposed end with a navigable approach before moving the arm."""
        env = self.env
        start = env.data.body("mabel_chassis").xpos[:2].copy()
        pending = next(p for p in env.products if p.name not in self.observer.completed)
        midpoint = sum(env.data.body(p.name).xpos[1] for p in products) / 2
        preferred = 1 if pending.target[1] < midpoint else -1
        routes = report["workspace_candidates"] = []
        for direction in (preferred, -preferred):
            active, neighbor = products if direction > 0 else products[::-1]
            body = env.data.body(active.name)
            radius = np.abs(body.xmat.reshape(3, 3)[1]) @ active.half_size
            goal_y = body.xpos[1] - direction * (radius + 0.015) + FRONT_SIDE_BASE_OFFSET_M
            for dx in (-0.04, 0.0, 0.04, 0.08):
                # Allow extra room for wheel steering near the supply table.
                goal = np.array([start[0] + dx, goal_y + direction * 0.025])
                navigable = (
                    plan_base_path(env, start, goal, footprint_radius=PRESS_BASE_FOOTPRINT_RADIUS_M)
                    is not None
                )
                routes.append(
                    dict(direction=direction, goal_xy_m=goal.tolist(), navigable=navigable)
                )
                if navigable:
                    return direction, active, neighbor, goal
        raise InteractionFailure("row_press_entry: no navigable exposed-side workspace")
