"""A transient inactive-object tip must survive recovery in evaluation evidence."""

import json
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from abstract_stack.benchmark_report import campaign_report, write_comparison
from abstract_stack.bundle import build_scene_bundle
from abstract_stack.campaign import run_campaign
from abstract_stack.config import load_scene_config
from abstract_stack.env import ShelfStockingEnv
from abstract_stack.evaluation import StageObserver
from abstract_stack.tuning import load_control_config, score_campaign


def test_transient_inactive_tip_is_retained_without_mutating_physics():
    env = ShelfStockingEnv(load_scene_config(scene="pick_place"), robot=False)
    try:
        observer = StageObserver(env)
        observer.begin_retreat("carton_0")
        joint = env.model.joint("carton_1_joint")
        adr = int(joint.qposadr[0])
        original = env.data.qpos.copy()
        # Fixture states, never a claimed successful physical rollout.
        env.data.qpos[adr + 3 : adr + 7] = [np.sqrt(0.5), np.sqrt(0.5), 0, 0]
        mujoco.mj_forward(env.model, env.data)
        before, controls = env.data.qpos.copy(), env.data.ctrl.copy()
        observer.sample("clear", "carton_0")
        np.testing.assert_array_equal(env.data.qpos, before)
        np.testing.assert_array_equal(env.data.ctrl, controls)
        env.data.qpos[:] = original
        mujoco.mj_forward(env.model, env.data)
        observer.sample("clear", "carton_0")
        observer.end_retreat("carton_0")
        report = observer.report([])
        assert not env.info()["tipped"]
        assert not report["no_knockovers"]
        assert report["incidents"][0]["product"] == "carton_1"
        assert report["incidents"][0]["phase"] == "clear"
        assert not report["products"]["carton_0"]["retreat"]
    finally:
        env.close()


@pytest.mark.parametrize(
    "settings",
    [
        {"max_force": 100},
        {"target_tolerance": 1},
        {"grasp_force_target": True},
        {"push_force_target": float("nan")},
        {"grip_force_gain": 1},
        [],
    ],
)
def test_search_cannot_relax_success_or_abort_limits(tmp_path, settings):
    path = tmp_path / "controls.json"
    path.write_text(json.dumps(settings))
    with pytest.raises(ValueError):
        load_control_config(path)


def test_crashed_run_cannot_earn_stage_progress():
    run = {
        "returncode": -11,
        "artifacts_valid": False,
        "physical_success": True,
        "stage_evaluation": {
            "no_knockovers": True,
            "products": {
                "box": dict.fromkeys(("pick", "place", "final_position", "retreat"), True)
            },
        },
    }
    score = score_campaign({"runs": [run]})
    assert score[:2] == (0, -1)
    assert score[3] == 0


def test_unreached_and_crashed_objects_stay_in_report_denominator(tmp_path):
    episode = tmp_path / "crash"
    episode.mkdir()
    campaign = tmp_path / "campaign.json"
    campaign.write_text(
        json.dumps(
            {
                "seeds": [0, 1],
                "objects": 2,
                "video": False,
                "runs": [{"seed": 0, "directory": str(episode), "returncode": -11}],
            }
        )
    )
    report = campaign_report(campaign)
    assert report["scheduled_objects"] == 4
    assert report["success_rate"] == 0
    assert report["completed_scenes"] == 1
    assert not any(report["stages"].values())


def test_wall_timeout_retains_unknown_process_and_does_not_count_completion(tmp_path):
    result = run_campaign(tmp_path, [0], episode_timeout=0.01)
    run = result["runs"][0]
    assert run["returncode"] == 124
    assert run["verification_status"] == "unknown"
    assert run["physical_success"] is None
    assert result["successful_runs"] == 0
    evidence = json.loads((tmp_path / "seed-0/process_failure.json").read_text())
    assert evidence["kind"] == "wall_timeout"
    assert (tmp_path / "seed-0/run.log").exists()


def test_search_never_rewards_unknown_force_or_extra_progress_after_knocking():
    partial = {
        "returncode": 2,
        "artifacts_valid": True,
        "physical_success": False,
        "product_count": 1,
        "peak_forces_N": {"hand": 10},
        "elapsed_sim_time": 80,
        "stage_evaluation": {
            "no_knockovers": True,
            "incidents": [],
            "products": {
                "box": {"pick": True, "place": True, "final_position": False, "retreat": False}
            },
        },
    }
    tipped = json.loads(json.dumps(partial))
    tipped["stage_evaluation"]["incidents"] = [{"kind": "tipped"}]
    tipped["stage_evaluation"]["products"]["box"]["final_position"] = True
    unknown = {"returncode": 124, "artifacts_valid": False, "physical_success": None}
    assert score_campaign({"runs": [partial]}) > score_campaign({"runs": [tipped]})
    assert score_campaign({"runs": [partial]}) > score_campaign({"runs": [unknown]})


def test_same_seed_with_changed_geometry_cannot_be_reported_as_paired(tmp_path):
    manifests = []
    for label, width in (("before", 0.6), ("after", 0.7)):
        root = tmp_path / label
        episode = root / "seed-0"
        config = replace(load_scene_config(scene="pick_place"), width=width)
        env = ShelfStockingEnv(config, robot=False)
        try:
            build_scene_bundle(env, episode / "initial-scene")
            summary = {
                "physical_success": False,
                "metrics": env.info(),
                "controller": {},
                "stage_evaluation": {"products": {}, "incidents": []},
                "trace_columns": ["sim_time", "hand_normal_N"],
            }
            (episode / "summary.json").write_text(json.dumps(summary))
            np.savez(episode / "trace.npz", samples=np.array([[0.0, 0.0], [0.002, 0.0]]))
            np.savez(episode / "rollout_start.npz", state=env.data.qpos)
            path = root / "campaign.json"
            path.write_text(
                json.dumps(
                    {
                        "seeds": [0],
                        "objects": 2,
                        "video": False,
                        "runs": [{"seed": 0, "directory": str(episode), "returncode": 2}],
                    }
                )
            )
            manifests.append(path)
        finally:
            env.close()
    with pytest.raises(ValueError, match="different scene/model/initial conditions"):
        write_comparison(*manifests, tmp_path / "report")
