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
from abstract_stack.evaluation import StageObserver, manipulation_evaluation
from abstract_stack.tuning import load_control_config, score_campaign


def test_prefilled_cartons_do_not_earn_manipulation_stage_credit():
    stages = ("pick", "place", "final_position", "retreat")
    result = {
        "physical_success": True,
        "scheduled_pick_products": ["inserted"],
        "initially_stocked_products": ["prefilled"],
        "stage_evaluation": {
            "no_knockovers": True,
            "products": {
                "inserted": dict.fromkeys(stages, True),
                "prefilled": dict.fromkeys(stages, False),
            },
        },
    }
    report = manipulation_evaluation(result)
    assert report["scheduled_manipulations"] == 1
    assert report["stages"] == dict.fromkeys(stages, 1)
    assert report["clean_success"]
    result["stage_evaluation"]["products"]["inserted"]["final_position"] = False
    report = manipulation_evaluation(result)
    assert report["packed_success"]
    assert not report["clean_success"]
    assert report["remaining_alignment_products"] == ["inserted"]
    result["stage_evaluation"]["products"]["inserted"]["retreat"] = False
    assert not manipulation_evaluation(result)["packed_success"]
    assert not manipulation_evaluation(result)["clean_success"]
    result["scheduled_pick_products"] = []
    assert not manipulation_evaluation(result)["clean_success"]


def test_campaign_and_search_use_scheduled_manipulations_consistently(tmp_path):
    episode = tmp_path / "episode"
    episode.mkdir()
    stages = ("pick", "place", "final_position", "retreat")
    summary = {
        "physical_success": True,
        "task_success": True,
        "scheduled_pick_products": ["inserted"],
        "initially_stocked_products": ["prefilled"],
        "controller": {},
        "trace_columns": ["sim_time"],
        "metrics": {"stocked_count": 2, "product_count": 2, "elapsed_sim_time": 90},
        "stage_evaluation": {
            "no_knockovers": True,
            "incidents": [],
            "products": {
                "inserted": dict.fromkeys(stages, True),
                "prefilled": dict.fromkeys(stages, False),
            },
        },
    }
    (episode / "summary.json").write_text(json.dumps(summary))
    np.savez(episode / "trace.npz", samples=np.zeros((1, 1)))
    np.savez(episode / "rollout_start.npz", state=np.zeros(0))
    path = tmp_path / "campaign.json"
    campaign = {
        "seeds": [0, 1],
        "objects": 2,
        "initially_stocked": 1,
        "video": False,
        "runs": [{"seed": 0, "directory": str(episode), "returncode": 0}],
    }
    path.write_text(json.dumps(campaign))
    report = campaign_report(path)
    assert report["clean_successes"] == 1
    assert report["scheduled_objects"] == 2  # Includes the unrun insertion.
    assert report["total_products"] == 4
    assert report["stages"] == dict.fromkeys(stages, 1)
    from abstract_stack.campaign import summarize_run

    run = summarize_run(episode, 0, video=False)
    assert score_campaign({"runs": [run]})[0] == 1
    assert score_campaign({"runs": [run]})[3] == 4
    summary["task_success"] = False
    (episode / "summary.json").write_text(json.dumps(summary))
    assert campaign_report(path)["clean_successes"] == 0


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


@pytest.mark.parametrize("dense", [False, True])
def test_wall_timeout_retains_unknown_process_and_does_not_count_completion(tmp_path, dense):
    options = dict(objects=10, initially_stocked=9, total_free_space=0.005) if dense else {}
    result = run_campaign(tmp_path, [0], episode_timeout=0.01, **options)
    run = result["runs"][0]
    assert run["returncode"] == 124
    assert run["verification_status"] == "unknown"
    assert run["physical_success"] is None
    assert result["successful_runs"] == 0
    evidence = json.loads((tmp_path / "seed-0/process_failure.json").read_text())
    assert evidence["kind"] == "wall_timeout"
    assert (tmp_path / "seed-0/run.log").exists()
    if dense:
        assert result["initially_stocked"] == 9
        assert result["total_free_space_m"] == 0.005
        assert json.loads((tmp_path / "seed-0/scene.json").read_text())["product_count"] == 10
        assert campaign_report(tmp_path / "campaign.json")["scheduled_objects"] == 1


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


@pytest.mark.parametrize(
    "after_width,change_velocity,message",
    [
        (0.7, False, "different scene/model/initial conditions"),
        (0.6, True, "different settled integration state"),
    ],
)
def test_same_seed_with_changed_inputs_cannot_be_reported_as_paired(
    tmp_path,
    after_width,
    change_velocity,
    message,
):
    manifests = []
    for label, width in (("before", 0.6), ("after", after_width)):
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
            if label == "after" and change_velocity:
                env.data.qvel[0] += 0.01
            spec = mujoco.mjtState.mjSTATE_INTEGRATION
            state = np.empty(mujoco.mj_stateSize(env.model, spec))
            mujoco.mj_getState(env.model, env.data, state, spec)
            np.savez(episode / "rollout_start.npz", state=state, state_spec=int(spec))
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
    with pytest.raises(ValueError, match=message):
        write_comparison(*manifests, tmp_path / "report")
