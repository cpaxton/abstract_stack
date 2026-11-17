"""Synchronized raw camera collection without requiring a graphics context."""

import io

import mujoco
import numpy as np
import pytest

from abstract_stack.lfd import EpisodeRecorder, ObservationField, Snapshot, load_episode
from abstract_stack.lfd.cameras import RawCameraCollector
from abstract_stack.lfd.lerobot import _verify_image
from abstract_stack.lfd.native import action_contract, proprioception


@pytest.fixture
def cameras():
    model = mujoco.MjModel.from_xml_string("""<mujoco><option timestep=".002"/>
      <worldbody><camera name="stocking_third_person" pos="1 0 1"/>
      <body><joint name="hinge" range="-1 1"/><geom size=".03"/>
      <camera name="zed_left" pos="0 0 .1"/></body></worldbody>
      <actuator><position name="position_ref" joint="hinge" kp="10" ctrlrange="-1 1"/>
      </actuator></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


class Renderer:
    def __init__(self, model, *, height, width):
        self.height, self.width = height, width
        self.images = []
        self.closed = False

    def update_scene(self, data, *, camera, scene_option):
        self.camera = camera
        assert not np.any(scene_option.sitegroup)
        self.stamp = data.time

    def render(self):
        value = 29 if self.camera == "zed_left" else 73
        image = np.full((self.height, self.width, 3), value, dtype=np.uint8)
        self.images.append(image)
        return image

    def close(self):
        self.closed = True


def test_views_have_same_native_time_raw_rgb_and_separate_capture_diagnostics(cameras):
    model, data = cameras
    collector = RawCameraCollector(model, renderer_factory=Renderer, width=32, height=16)
    original = (data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), model.cam_pos.copy())
    samples = collector.sample(data)
    assert len(samples) == 2
    for name, value in samples.items():
        assert value.available and value.sample_time_s == data.time
        assert value.value.dtype == np.uint8 and value.value.shape == (16, 32, 3)
        assert np.all(value.value == (29 if name.endswith("zed_left") else 73))
    assert all(event["capture_time_s"] == 0 for event in collector.events)
    assert all(
        event["capture_started_monotonic_ns"] <= event["capture_finished_monotonic_ns"]
        for event in collector.events
    )
    for before, after in zip(
        original, (data.qpos, data.qvel, data.ctrl, model.cam_pos), strict=True
    ):
        np.testing.assert_array_equal(before, after)
    renderer = collector._renderer
    renderer.images[0][:] = 0
    assert np.all(samples["observation.images.zed_left"].value == 29)
    metadata = collector.metadata()
    assert metadata["cameras"]["zed_left"]["parent_body_id"] != 0
    assert metadata["cameras"]["stocking_third_person"]["parent_body_id"] == 0
    assert not metadata["overlays"] and not metadata["cutaway"]
    collector.close()
    assert renderer.closed


def test_missing_frame_masks_and_native_timestamps_survive_neutral_roundtrip(tmp_path, cameras):
    class DropThird(Renderer):
        def render(self):
            if self.camera == "stocking_third_person":
                raise RuntimeError("injected dropped frame")
            return super().render()

    model, data = cameras
    collector = RawCameraCollector(model, renderer_factory=DropThird, width=32, height=16)
    state_field = ObservationField("observation.state", (3,), "rad,rad/s,Nm", "native", "measured")
    recorder = EpisodeRecorder(
        tmp_path / "episode",
        action_contract(model),
        (state_field, *collector.fields),
        fps=50,
        physics_dt_s=0.002,
        provenance=dict(
            model_revision="fixture",
            controller_source={},
            controller_settings={},
            scene_seed=0,
            camera_contract=collector.metadata(),
        ),
        task="raw image pipeline fixture",
        purpose="pipeline_fixture",
    )

    def observe():
        measured = proprioception(data)
        return Snapshot(measured.time_s, measured.sensors | collector.sample(data))

    recorder.begin(observe(), [0.1])
    data.ctrl[:] = 0.12
    for _ in range(10):
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
    recorder.end(observe(), [0.12])
    recorder.finish(status="failure", task_success=False, reason="no task attempted")
    episode = load_episode(recorder.directory)
    first, third = "observation.images.zed_left", "observation.images.stocking_third_person"
    np.testing.assert_array_equal(episode.arrays[first + ".available"], [True, True])
    np.testing.assert_array_equal(episode.arrays[first + ".sample_time_s"], [0, data.time])
    np.testing.assert_array_equal(episode.arrays[third + ".available"], [False, False])
    np.testing.assert_array_equal(episode.arrays[third + ".sample_time_s"], [-1, -1])
    assert not np.any(episode.arrays[third])
    assert all(
        "injected dropped frame" in e["error"] for e in collector.events if not e["available"]
    )
    assert not episode.metadata["training_eligible"]
    collector.close()


def test_camera_configuration_fails_before_renderer_initialization(cameras):
    model, _ = cameras
    with pytest.raises(KeyError):
        RawCameraCollector(model, cameras=("nonexistent",), renderer_factory=Renderer)
    with pytest.raises(ValueError, match="framebuffer"):
        RawCameraCollector(model, width=100000, renderer_factory=Renderer)


def test_physics_advancement_during_capture_aborts_instead_of_mislabeling(cameras):
    model, data = cameras

    class Advances(Renderer):
        def render(self):
            data.time += 0.002  # Deliberate broken renderer test double.
            return super().render()

    collector = RawCameraCollector(model, renderer_factory=Advances)
    with pytest.raises(RuntimeError, match="Physics advanced"):
        collector.sample(data)
    collector.close()


def test_nonzero_image_storage_and_reader_normalization_are_checked_independently():
    from PIL import Image

    image = np.arange(16 * 32 * 3, dtype=np.uint8).reshape(16, 32, 3)
    encoded = io.BytesIO()
    Image.fromarray(image).save(encoded, format="PNG")
    stored = dict(bytes=encoded.getvalue())
    normalized = (image.astype(np.float32) / np.float32(255)).transpose(2, 0, 1)
    assert _verify_image(stored, normalized, image) == "float32 RGB divided by 255"
    assert _verify_image(stored, image.copy(), image) == "uint8 RGB"
    changed = image.copy()
    changed[0, 0, 0] = 3
    with pytest.raises(AssertionError):
        _verify_image(stored, normalized, changed)
    normalized[0, 0, 0] += 0.001
    with pytest.raises(AssertionError):
        _verify_image(stored, normalized, image)
