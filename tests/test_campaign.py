"""A crashed renderer must never be counted as a verified task success."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from abstract_stack.campaign import summarize_run


def test_campaign_collector_retains_partial_terminal_json_as_unknown(tmp_path):
    episode = tmp_path / "crashed"
    episode.mkdir()
    (episode / "summary.json").write_text('{"physical_success":')
    campaign = tmp_path / "campaign.json"
    campaign.write_text(
        json.dumps(
            {
                "seeds": [42],
                "objects": 2,
                "video": False,
                "runs": [{"seed": 42, "directory": str(episode), "returncode": -11}],
            }
        )
    )
    output = tmp_path / "report.json"
    script = Path(__file__).resolve().parents[1] / "scripts/summarize_stocking_campaigns.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--campaign",
            f"crash={campaign}",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text())["campaigns"]["crash"]
    assert report["unknown_scenes"] == report["scheduled_scenes"] == 1
    assert report["clean_successes"] == 0
    assert report["stages"] == dict.fromkeys(("pick", "place", "final_position", "retreat"), 0)
    assert report["runs"][0]["physical_success"] is None
    assert report["runs"][0]["verification_status"] == "unknown"


def test_native_crash_without_summary_remains_unknown(tmp_path):
    result = summarize_run(tmp_path, -11, video=True)
    assert result["physical_success"] is None
    assert result["verification_status"] == "unknown"
    assert not result["artifacts_valid"]
    assert result["returncode"] == -11
    assert result["process_failure"]["kind"] == "native_signal"
    assert result["process_failure"]["signal"] == "SIGSEGV"


def test_unknown_child_status_retains_parent_termination_separately(tmp_path):
    (tmp_path / "process_failure.json").write_text(
        json.dumps(
            {
                "kind": "orchestration_termination",
                "orchestration_exit_status": 143,
                "child_returncode": None,
            }
        )
    )
    result = summarize_run(tmp_path, None, video=False)
    assert result["returncode"] is None
    assert result["physical_success"] is None
    assert result["verification_status"] == "unknown"
    assert result["process_failure"]["orchestration_exit_status"] == 143
    assert result["process_failure"]["kind"] == "orchestration_termination"


def test_video_files_without_task_summary_do_not_establish_success(tmp_path):
    (tmp_path / "videos").mkdir()
    for name in ("first_person", "third_person"):
        (tmp_path / f"videos/{name}.mp4").write_bytes(b"partial stream")
    result = summarize_run(tmp_path, 2, video=True)
    assert result["physical_success"] is None
    assert not result["artifacts_valid"]


def test_missing_video_metadata_is_reported_without_losing_task_result(tmp_path):
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "physical_success": True,
                "failure": None,
                "metrics": {"stocked_count": 2, "product_count": 2, "elapsed_sim_time": 90},
            }
        )
    )
    result = summarize_run(tmp_path, 1, video=True)
    assert result["physical_success"] is True
    assert not result["artifacts_valid"]
    assert result["artifact_error"]


def test_native_signal_after_terminal_summary_retains_physical_evidence(tmp_path):
    # Terminal-evidence fixture; no claim of a real manipulation in this test.
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "physical_success": True,
                "failure": None,
                "metrics": {"stocked_count": 2, "product_count": 2, "elapsed_sim_time": 90},
            }
        )
    )
    np.savez(tmp_path / "trace.npz", samples=np.zeros((0, 0)))
    np.savez(tmp_path / "rollout_start.npz", state=np.zeros(0))
    result = summarize_run(tmp_path, -11, video=False)
    assert result["physical_success"] is True
    assert result["verification_status"] == "verified"
    assert result["artifacts_valid"]
    assert result["process_failure"]["signal"] == "SIGSEGV"
