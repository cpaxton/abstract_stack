"""Named force setpoints, independent of contact-control and hand mode flags."""

from __future__ import annotations

FORCE_PROFILE_NAMES = ("selected", "legacy")


def force_profile_settings(name="selected"):
    if name not in FORCE_PROFILE_NAMES:
        raise ValueError(f"Unknown force profile: {name}")
    if name == "legacy":
        return {"grasp_force_target": 6.0, "push_force_target": 2.0}
    # Keep CLI parsing independent of MuJoCo/backend initialization.
    from abstract_stack.interactions import ScriptConfig

    defaults = ScriptConfig()
    return {
        "grasp_force_target": defaults.grasp_force_target,
        "push_force_target": defaults.push_force_target,
    }


def resolve_control_settings(profile="selected", path=None):
    """Apply a named profile first, then validated explicit JSON overrides."""
    from abstract_stack.tuning import load_control_config

    return force_profile_settings(profile) | load_control_config(path)
