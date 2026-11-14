import os
import sys
from pathlib import Path
import mujoco
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orca_vla import OrcaVLAActionAdapter


def _resolve_sim_scene(scene_name: str, version: str = "v2") -> Path:
    for cand in [
        os.environ.get("ORCA_SIM_ROOT"),
        *[str(p) for p in sorted(REPO_ROOT.parent.glob("*Dexterous-Hand-Simulation*"))],
        *[str(p) for p in sorted(Path.home().glob("*Dexterous-Hand-Simulation*"))],
    ]:
        if cand:
            p = Path(cand) / "src" / "orca_sim" / "scenes" / version / scene_name
            if p.exists():
                return p.resolve()
    pytest.skip(f"Simulation scene {scene_name} not found locally.")


def test_17dof_hand_synergy_and_bounds():
    scene_path = _resolve_sim_scene("scene_right_cube_orientation.xml")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    adapter = OrcaVLAActionAdapter(model)
    assert len(adapter.hand_act_indices) == 17

    hand_q, ratio = adapter.gripper_to_hand_ctrl(1.0, mode="libero")
    assert hand_q.shape == (17,)
    assert np.isclose(ratio, 1.0)
    assert np.all(hand_q >= adapter.action_low)
    assert np.all(hand_q <= adapter.action_high)


def test_in_hand_17dof_synergy_delta():
    scene_path = _resolve_sim_scene("scene_right_cube_orientation.xml")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    adapter = OrcaVLAActionAdapter(model)
    assert len(adapter.hand_act_indices) == 17

    vla_7d = np.array([0.2, -0.1, 0.05, 0.4, -0.3, 0.1, 0.8], dtype=np.float32)
    delta_17d = adapter.in_hand_synergy_delta(vla_7d)
    assert delta_17d.shape == (17,)
    assert np.all(delta_17d >= -1.0)
    assert np.all(delta_17d <= 1.0)
