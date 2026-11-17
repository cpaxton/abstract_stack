"""The real CLI must construct the controller with supported recording arguments."""

import runpy
import sys
from pathlib import Path

import pytest

from abstract_stack.assets import mabel_model_path
from abstract_stack.receding_stocking import RecedingStockingScript


@pytest.mark.mabel
@pytest.mark.parametrize("video", [False, True])
def test_runner_constructs_native_controller_with_and_without_recording(
    tmp_path, monkeypatch, video
):
    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    observed = []

    def stop_before_episode(script):
        observed.append(script.video_options)
        assert script.env.data.time == 0
        return dict(physical_success=False, task_success=False, failure="constructor-only probe")

    monkeypatch.setattr(RecedingStockingScript, "run", stop_before_episode)
    path = Path(__file__).resolve().parents[1] / "scripts/run_replanning.py"
    arguments = [str(path), "--output-dir", str(tmp_path / "episode")]
    if video:
        arguments += ["--video", "--video-fps", "10"]
    monkeypatch.setattr(sys, "argv", arguments)
    entry = runpy.run_path(str(path))
    assert entry["main"]() == 2
    assert len(observed) == 1
    if video:
        assert observed[0]["fps"] == 10
        assert observed[0]["first_person"] == "zed_left"
    else:
        assert observed[0] is None
