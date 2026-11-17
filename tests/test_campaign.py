"""A crashed renderer must never be counted as a verified task success."""

import json

from abstract_stack.campaign import summarize_run


def test_native_crash_without_summary_remains_unknown(tmp_path):
    result = summarize_run(tmp_path, -11, video=True)
    assert result["physical_success"] is None
    assert result["verification_status"] == "unknown"
    assert not result["artifacts_valid"]
    assert result["returncode"] == -11


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
