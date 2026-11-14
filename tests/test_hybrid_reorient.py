import os
import sys
from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

for cand_sim in [
    os.environ.get("ORCA_SIM_ROOT"),
    *[str(p) for p in sorted(REPO_ROOT.parent.glob("*Dexterous-Hand-Simulation*"))],
    *[str(p) for p in sorted(Path.home().glob("*Dexterous-Hand-Simulation*"))],
]:
    if cand_sim and (Path(cand_sim) / "src" / "orca_sim").exists():
        sim_src = str((Path(cand_sim) / "src").resolve())
        if sim_src not in sys.path:
            sys.path.insert(0, sim_src)
        break

from scripts.run_cube_reorient import (
    LowLevelReorientRLController,
    classify_high_level_reorient_skill,
)


def test_hybrid_controller_reorients_and_holds_cube():
    try:
        from orca_sim import OrcaHandRightCubeOrientation
        from orca_sim.wrappers import RelativeJointTargetAction
    except ImportError:
        pytest.skip("orca_sim not available")

    base_env = OrcaHandRightCubeOrientation(version="v2", initial_red_face="down", max_episode_steps=200)
    env = RelativeJointTargetAction(base_env, max_delta_frac=0.05)
    obs, info = env.reset(seed=7)

    ctrl = LowLevelReorientRLController()
    ctrl.reset(info)

    dummy_vla_chunk = np.zeros((10, 7), dtype=np.float32)
    for _ in range(120):
        skill, vla_res = classify_high_level_reorient_skill(info, dummy_vla_chunk)
        act_17d = ctrl.step_action(obs, info, skill, vla_res)
        obs, _, terminated, truncated, info = env.step(act_17d)
        if terminated or truncated:
            break

    base_env.close()
    assert info["dropped"] is False
    assert info["is_success"] is True
    assert info["success_streak"] >= 25
    assert np.rad2deg(info["red_face_up_angle_rad"]) <= 15.0
