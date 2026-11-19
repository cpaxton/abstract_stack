"""Raw camera observations at a paused native physics boundary.

MuJoCo is imported lazily; importing the neutral episode reader does not acquire
a graphics context. Capture wall times are diagnostics, never physics clocks.
"""

import time

import numpy as np

from abstract_stack.lfd.episode import ObservationField, SensorSample


class RawCameraCollector:
    """Capture sequential views of the same unstepped state, without overlays.

    A renderer failure produces a missing frame and an explicit diagnostic;
    invalid camera configuration fails before collection. The caller must keep
    physics paused throughout ``sample``. No past image is reused after failure.
    """

    def __init__(
        self,
        model,
        *,
        cameras=("zed_left", "stocking_third_person"),
        width=160,
        height=120,
        renderer_factory=None,
    ):
        import mujoco

        if (
            type(width) is not int
            or type(height) is not int
            or min(width, height) < 16
            or width > model.vis.global_.offwidth
            or height > model.vis.global_.offheight
            or not cameras
            or len(set(cameras)) != len(cameras)
        ):
            raise ValueError("Raw camera dimensions must fit the native framebuffer")
        self.model = model
        self.cameras = tuple(cameras)
        self.width, self.height = width, height
        self._factory = renderer_factory or mujoco.Renderer
        self._renderer = None
        self._options = mujoco.MjvOption()
        self._options.sitegroup[:] = 0  # Hide target/annotation sites in raw student images.
        self.events = []
        self._camera_metadata = {}
        for name in self.cameras:
            camera = model.camera(name)
            self._camera_metadata[name] = dict(
                camera_id=int(camera.id),
                parent_body_id=int(model.cam_bodyid[camera.id]),
                mode=int(model.cam_mode[camera.id]),
                local_position=model.cam_pos[camera.id].tolist(),
                local_quaternion_wxyz=model.cam_quat[camera.id].tolist(),
                vertical_fov_degrees=float(model.cam_fovy[camera.id]),
            )

    @property
    def fields(self):
        return tuple(
            ObservationField(
                "observation.images." + name,
                (self.height, self.width, 3),
                "uint8 RGB",
                name,
                "raw MuJoCo RGB; instantaneous native camera; no overlays or target sites",
                dtype="uint8",
            )
            for name in self.cameras
        )

    def metadata(self):
        import mujoco

        return dict(
            enabled=True,
            renderer="mujoco.Renderer",
            mujoco_version=mujoco.__version__,
            width=self.width,
            height=self.height,
            cameras=self._camera_metadata,
            sample_clock="native data.time at paused control boundary",
            exposure_model="instantaneous; no shutter, readout, noise or transport latency",
            synchronization="views rendered sequentially without physics stepping",
            capture_clock="time.monotonic_ns; local process diagnostic, not simulation time",
            sitegroup=self._options.sitegroup.tolist(),
            geomgroup=self._options.geomgroup.tolist(),
            cutaway=False,
            overlays=False,
            missing_frame_policy="zero uint8 image, available=false, sample_time_s=-1; no reuse",
        )

    def sample(self, data):
        stamp = float(data.time)
        if not np.isfinite(stamp) or stamp < 0:
            raise ValueError("Camera boundary requires finite nonnegative native time")
        result = {}
        for name in self.cameras:
            started = time.monotonic_ns()
            error = None
            image = np.zeros((self.height, self.width, 3), np.uint8)
            available = False
            try:
                if self._renderer is None:
                    self._renderer = self._factory(self.model, height=self.height, width=self.width)
                self._renderer.update_scene(data, camera=name, scene_option=self._options)
                captured = np.asarray(self._renderer.render())
                if captured.shape != image.shape or captured.dtype != np.uint8:
                    raise ValueError("Renderer did not produce declared HWC uint8 RGB")
                image, available = captured.copy(), True
            except Exception as exception:
                error = f"{type(exception).__name__}: {exception}"
            finished = time.monotonic_ns()
            self.events.append(
                dict(
                    camera=name,
                    capture_time_s=stamp,
                    sample_time_s=stamp if available else -1.0,
                    capture_started_monotonic_ns=started,
                    capture_finished_monotonic_ns=finished,
                    available=available,
                    error=error,
                )
            )
            if float(data.time) != stamp:
                raise RuntimeError("Physics advanced during raw camera capture")
            result["observation.images." + name] = SensorSample(
                image, stamp if available else -1.0, available
            )
        return result

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
