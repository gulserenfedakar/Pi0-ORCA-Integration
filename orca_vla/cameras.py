from __future__ import annotations
from typing import List
import mujoco
import numpy as np

# Joe's ChArUco-calibrated Logitech Brio 100 camera poses & intrinsics (1920x1080)
# from `Brio 100 Camera Intrinsics.md` and `scene_right_cube_orientation_real2sim.xml`.
CALIBRATED_BRIO100_CAMERAS = {
    "top": {
        "pos": np.array([0.19692332, -0.01392253, 0.66355845], dtype=np.float64),
        "look_dir": np.array([-0.05176770, 0.03077897, -0.99818473], dtype=np.float64),
        "xyaxes": np.array([-0.260288, -0.965394, -0.016269, 0.964142, -0.258974, -0.057988], dtype=np.float64),
        "focalpixel": (1960.109, 1956.528),
        "principalpixel": (-21.975, 13.949),
        "resolution": (1920, 1080),
    },
    "side": {
        "pos": np.array([0.18990486, 0.57769233, 0.14034025], dtype=np.float64),
        "look_dir": np.array([0.06816238, -0.99130002, 0.11259732], dtype=np.float64),
        "xyaxes": np.array([-0.997573, -0.069328, -0.006465, -0.014215, 0.111883, 0.993620], dtype=np.float64),
        "focalpixel": (1935.714, 1931.204),
        "principalpixel": (13.568, 37.484),
        "resolution": (1920, 1080),
    },
}


class VLACameraRenderer:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, width: int = 224, height: int = 224):
        self.model = model
        self.data = data
        self.width = width
        self.height = height
        self.renderer = mujoco.Renderer(model, height=height, width=width)

    def render(self, camera_id: int | str) -> np.ndarray:
        """
        Renders the scene from the specified camera and returns a (H, W, 3) RGB numpy array.
        Supports Joe's calibrated real2sim XML cameras ('top', 'side') directly when loaded from
        `scene_right_cube_orientation_real2sim.xml`, with calibrated virtual fallbacks when using
        the plain scene, plus virtual palmar presets ('palm_front', 'palm_top', 'default_free').
        """
        named_cams = set(self.get_available_cameras())
        if isinstance(camera_id, str) and camera_id in named_cams:
            self.renderer.update_scene(self.data, camera=camera_id)
            return self.renderer.render().copy()

        if isinstance(camera_id, str) and camera_id in ("top", "side", "palm_front", "palm_top", "default_free"):
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(self.model, cam)
            if camera_id == "top":
                # Matches Joe's calibrated overhead Brio 100 camera at (0.197, -0.014, 0.664) m
                cam.lookat[:] = [0.172, 0.001, 0.180]
                cam.distance = 0.485
                cam.azimuth = -107.0
                cam.elevation = -86.5
            elif camera_id == "side":
                # Matches Joe's calibrated lateral Brio 100 camera at (0.190, 0.578, 0.140) m looking along -y
                cam.lookat[:] = [0.218, -0.015, 0.187]
                cam.distance = 0.595
                cam.azimuth = -86.1
                cam.elevation = 6.5
            elif camera_id == "palm_front":
                cam.lookat[:] = [0.165, -0.015, 0.18]
                cam.distance = 0.28
                cam.azimuth = 145.0
                cam.elevation = -32.0
            elif camera_id == "palm_top":
                cam.lookat[:] = [0.17, -0.015, 0.185]
                cam.distance = 0.22
                cam.azimuth = 180.0
                cam.elevation = -75.0
            self.renderer.update_scene(self.data, camera=cam)
            return self.renderer.render().copy()

        self.renderer.update_scene(self.data, camera=camera_id)
        return self.renderer.render().copy()

    def get_available_cameras(self) -> List[str]:
        """Returns a list of all available named XML cameras in the current MuJoCo model."""
        return [self.model.cam(i).name for i in range(self.model.ncam)]
