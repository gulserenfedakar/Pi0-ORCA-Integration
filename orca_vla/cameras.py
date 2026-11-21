from __future__ import annotations
import mujoco
import numpy as np
from typing import List

class VLACameraRenderer:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, width: int = 224, height: int = 224):
        self.model = model
        self.data = data
        self.width = width
        self.height = height
        # MuJoCo Renderer for headless or off-screen rendering
        self.renderer = mujoco.Renderer(model, height=height, width=width)

    def render(self, camera_id: int | str) -> np.ndarray:
        """
        Renders the scene from the specified camera and returns a (H, W, 3) RGB numpy array.
        Supports virtual MjvCamera presets ('palm_front', 'palm_top', 'default_free') when
        the scene has no named XML cameras (e.g., OrcaHandRightCubeOrientation).
        """
        if isinstance(camera_id, str) and camera_id in ("palm_front", "palm_top", "default_free"):
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(self.model, cam)
            if camera_id == "palm_front":
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
        # Return a copy of the rendered RGB image to prevent mutability issues
        return self.renderer.render().copy()

    def get_available_cameras(self) -> List[str]:
        """
        Returns a list of all available camera names in the current MuJoCo model.
        """
        cam_names = []
        for i in range(self.model.ncam):
            cam_names.append(self.model.cam(i).name)
        return cam_names
