"""Real dual-view encoding, timestamps and first-person camera ownership."""

import json
import os
import shutil
import subprocess
import sys

import pytest

from abstract_stack.assets import mabel_model_path


@pytest.mark.mabel
def test_native_camera_mp4s_have_identical_frame_counts_and_timestamps(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe are required for the encoding integration test")
    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    # Isolate GL backend selection from other tests that imported MuJoCo first.
    source = """
import sys
from abstract_stack.config import load_scene_config
from abstract_stack.env import ShelfStockingEnv
from abstract_stack.recording import DualVideoRecorder, HeadGaze
from abstract_stack.mabel_controller import MabelController
env = ShelfStockingEnv(load_scene_config(scene='pick_place'))
try:
    recorder = DualVideoRecorder(env, sys.argv[1], fps=10, width=160, height=120)
    gaze = HeadGaze(env.model, env.data)
    tracker = MabelController(env.model, env.data)
    recorder.capture(phase='test', product='carton_1', forces={})
    for _ in range(10):
        goal = gaze.targets(env.data.body('carton_1').xpos.copy(), env.dt)
        env.step(tracker.targets(goal, dt=env.dt))
        recorder.capture(phase='test', product='carton_1', forces={})
    recorder.finish({'physical_success': False, 'failure': 'Encoding test only'})
finally:
    env.close()
"""
    process_env = dict(os.environ, MUJOCO_GL="egl")
    subprocess.run(
        [sys.executable, "-c", source, str(tmp_path)], env=process_env, check=True, timeout=60
    )
    metadata = json.loads((tmp_path / "video.json").read_text())
    assert metadata["cameras"]["first_person"] == "zed_left"
    assert metadata["physics_frames"] == 3
    assert metadata["paused_end_frames"] == 10
    assert not metadata["physical_success"]
    assert metadata["physics_frame_times"] == sorted(set(metadata["physics_frame_times"]))
    for name in ("first_person", "third_person"):
        output = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,r_frame_rate,nb_frames",
                "-of",
                "json",
                str(tmp_path / f"{name}.mp4"),
            ],
            text=True,
        )
        stream = json.loads(output)["streams"][0]
        assert stream["codec_name"] == "h264"
        assert stream["width"] == 160 and stream["height"] == 120
        assert stream["r_frame_rate"] == "10/1"
        assert int(stream["nb_frames"]) == metadata["encoded_frames"]
