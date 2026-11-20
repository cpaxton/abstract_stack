"""Native selected-finger posture with explicit nominal abduction geometry."""

import numpy as np

from abstract_stack.dense_interactions import FingerPusher


class LoadAwareFingerPusher(FingerPusher):
    """Put selected Abd in commands and IK before native tip calibration.

    This changes position references only. External holding effort still comes
    from a separate bounded FingerLoadCompensator with applied-geometry checks.
    Baseline FingerPusher behavior remains unchanged.
    """

    def __init__(self, env, config, *, abduction_posture=0.0, **kwargs):
        self.abduction_posture = float(abduction_posture)
        finger = kwargs.get("finger", "Middle")
        if finger not in ("Index", "Middle", "Ring"):
            raise ValueError("Unsupported native pushing finger")
        name = f"Left {finger} Abd"
        joint, actuator = env.model.joint(name), env.model.actuator(name)
        if not np.isfinite(self.abduction_posture) or not max(
            joint.range[0], actuator.ctrlrange[0]
        ) <= self.abduction_posture <= min(joint.range[1], actuator.ctrlrange[1]):
            raise ValueError("Pusher abduction posture exceeds native bounds")
        super().__init__(env, config, **kwargs)

    def finger_targets(self, grip):
        targets = super().finger_targets(grip)
        targets[f"Left {self.finger} Abd"] = self.abduction_posture
        return targets
