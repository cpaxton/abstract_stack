"""Dense-scene admission using the shared native shelf-insertion skill."""

import numpy as np

from abstract_stack.finger_pusher import FingerPusher  # noqa: F401 -- compatibility import
from abstract_stack.interactions import StockingScript


class DenseStockingScript(StockingScript):
    """Coplanar transfer scene with release before every selected-finger push."""

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
        self.wrist_load_requested_peak = 0.0
        self.wrist_load_last_requested = 0.0
        self.wrist_load_clipped = 0

    def episode_evidence(self):
        evidence = super().episode_evidence()
        evidence["fixture_aid"] = "coplanar transfer table; explicitly part of the scene"
        return evidence

    def _front_center_x(self, extent):
        # Stage in the empty front portion of the deep shelf, before the
        # occupied row. Reduces unsupported lateral/yaw drift during the push.
        return self.env.config.front_x - self.config.front_staging_depth

    def _one_box(self, selected=None):
        self._record_agent_event(
            "carton_plan",
            protocol="dense_released_finger",
            steps=["pick", "place", "insert", "push_to_contact", "retreat"],
            plan_driven=self.config.plan_driven,
        )
        self._move("clear_high", [-0.30, -0.20, 1.35])
        if self.config.plan_driven:
            self._execute_planned_box(selected)
            return
        grip = self._pick_and_place_front(selected)
        self._release_and_insert_front(grip)
