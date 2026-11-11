import os
import sys
from pathlib import Path
import mujoco
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orca_vla import (
    OrcaVLAObservationBuilder,
    OrcaVLAObservationConfig,
    VLACameraRenderer,
)


def _resolve_sim_scene(scene_name: str, version: str = "v2") -> Path:
    for cand in [
        os.environ.get("ORCA_SIM_ROOT"),
        str(REPO_ROOT.parent / "SEP730-Dexterous-Hand-Simulation"),
        str(Path.home() / "SEP730-Dexterous-Hand-Simulation"),
    ]:
        if cand:
            p = Path(cand) / "src" / "orca_sim" / "scenes" / version / scene_name
            if p.exists():
                return p.resolve()
    pytest.skip(f"Simulation scene {scene_name} not found locally.")


def test_cube_reorientation_camera_and_observation_builder():
    scene_path = _resolve_sim_scene("scene_right_cube_orientation.xml")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = VLACameraRenderer(model, data, width=224, height=224)
    img_front = renderer.render("palm_front")
    img_top = renderer.render("palm_top")

    assert img_front.shape == (224, 224, 3)
    assert img_top.shape == (224, 224, 3)
    assert img_front.dtype == np.uint8
    assert np.any(img_front > 0)
    assert np.any(img_top > 0)

    cfg = OrcaVLAObservationConfig(external_cam="palm_front", wrist_cam="palm_top", num_joints=17)
    builder = OrcaVLAObservationBuilder(model, data, renderer, config=cfg)
    obs = builder.build_observation(
        "flip the cube red face up and hold it still in the palm",
        use_cartesian_eef_state=True,
        gripper_close_ratio=0.5,
    )
    assert obs["observation/image"].shape == (224, 224, 3)
    assert obs["observation/wrist_image"].shape == (224, 224, 3)
    assert obs["observation/state"].shape == (8,)
    assert np.all(np.isfinite(obs["observation/state"]))
