"""Planning snapshots carry scratch physics, never mutable task bookkeeping."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from abstract_stack.env import ShelfStockingEnv


def test_planning_snapshot_isolated_from_episode_state_and_operations():
    env = ShelfStockingEnv(robot=False)
    try:
        env.step()
        snapshot = env.clone_for_planning()
        assert snapshot.model is env.model  # Explicit read-only sharing contract.
        assert snapshot.data is not env.data
        assert snapshot.dt == env.dt
        before = env.data.qpos.copy()
        np.testing.assert_array_equal(snapshot.data.qpos, before)
        snapshot.data.qpos[0] += 0.01
        snapshot.data.qvel[0] = 1.0
        snapshot.data.time += 1.0
        np.testing.assert_array_equal(env.data.qpos, before)
        assert env.data.qvel[0] != snapshot.data.qvel[0]
        assert env.data.time != snapshot.data.time
        for name in ("_stocked", "interventions", "reset", "step", "render"):
            assert not hasattr(snapshot, name)
        with pytest.raises(FrozenInstanceError):
            snapshot.dt = 1.0
    finally:
        env.close()
