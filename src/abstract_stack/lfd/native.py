"""Read native robot actuator contracts and pre-action proprioception only."""

import numpy as np

from abstract_stack.lfd.episode import ActionContract, SensorSample, Snapshot


def action_contract(model):
    import mujoco

    names, units, modes = [], [], []
    for aid in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        if not name or not np.isclose(model.actuator_gear[aid, 0], 1):
            raise ValueError("Collection requires named native unit-gear servo actuators")
        bias = model.actuator_biasprm[aid]
        mode = "position" if bias[1] < 0 else "velocity" if bias[2] < 0 else None
        if mode is None:
            raise ValueError("Unsupported actuator: declare its action semantics before collection")
        transmission = model.actuator_trntype[aid]
        if transmission == mujoco.mjtTrn.mjTRN_JOINT:
            joint = int(model.actuator_trnid[aid, 0])
            kind = model.jnt_type[joint]
            unit = (
                "rad"
                if kind == mujoco.mjtJoint.mjJNT_HINGE
                else "m"
                if kind == mujoco.mjtJoint.mjJNT_SLIDE
                else None
            )
        elif transmission == mujoco.mjtTrn.mjTRN_TENDON and name == "Lift":
            tendon = int(model.actuator_trnid[aid, 0])
            start = int(model.tendon_adr[tendon])
            count = int(model.tendon_num[tendon])
            if count != 2 or any(
                model.wrap_type[i] != mujoco.mjtWrap.mjWRAP_JOINT
                or model.jnt_type[model.wrap_objid[i]] != mujoco.mjtJoint.mjJNT_SLIDE
                or not np.isclose(model.wrap_prm[i], 1)
                for i in range(start, start + count)
            ):
                raise ValueError("Lift contract requires the native 1:1 sliding-joint tendon")
            unit = "m"
        else:
            raise ValueError("Unsupported actuator transmission")
        if unit is None:
            raise ValueError("Unsupported actuator coordinate")
        names.append(name)
        modes.append(mode)
        units.append(unit if mode == "position" else unit + "/s")
    return ActionContract(
        tuple(names),
        tuple(units),
        tuple(modes),
        tuple(
            float(model.actuator_ctrlrange[a, 0]) if model.actuator_ctrllimited[a] else None
            for a in range(model.nu)
        ),
        tuple(
            float(model.actuator_ctrlrange[a, 1]) if model.actuator_ctrllimited[a] else None
            for a in range(model.nu)
        ),
    )


def proprioception(data):
    """Measured transmission positions, speeds and efforts, not target controls."""
    values = np.concatenate((data.actuator_length, data.actuator_velocity, data.actuator_force))
    return Snapshot(
        float(data.time), {"observation.state": SensorSample(values.copy(), float(data.time))}
    )
