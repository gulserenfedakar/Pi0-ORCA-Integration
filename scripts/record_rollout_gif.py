"""Generate animated rollout GIFs of trained RL policies on OrcaHandRightCubeOrientation.
"""
import sys
from pathlib import Path
import numpy as np
import imageio
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str((REPO_ROOT.parent / "SEP730-Dexterous-Hand-Simulation" / "src").resolve()))
sys.path.insert(0, str(REPO_ROOT))

from orca_sim import OrcaHandRightCubeOrientation
from orca_sim.wrappers import RelativeJointTargetAction
from orca_vla.cameras import VLACameraRenderer

def record_gif(algo="sac", model_path=None, vec_path=None, out_gif="assets/sac_reorient_demo.gif", init="down", seed=7, max_steps=250):
    env = OrcaHandRightCubeOrientation(version="v2", initial_red_face=init)
    env = RelativeJointTargetAction(env, max_delta_frac=0.05)
    env.reset(seed=seed)
    
    vec_env = DummyVecEnv([lambda: env])
    if vec_path and Path(vec_path).exists():
        vec_env = VecNormalize.load(vec_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    
    if algo == "sac":
        model = SAC.load(model_path, env=vec_env)
    else:
        model = PPO.load(model_path, env=vec_env)
        
    renderer = VLACameraRenderer(env.unwrapped.model, env.unwrapped.data, width=320, height=240)
    
    frames = []
    obs = vec_env.reset()
    
    success_streak = 0
    for step in range(max_steps):
        # Render palm front view
        frame = renderer.render(camera_id="palm_front")
        frames.append(frame)
        
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        
        info_d = info[0]
        if info_d.get("is_success", False):
            success_streak += 1
            if success_streak >= 25:
                # Add a few still frames at the end to showcase hold
                for _ in range(15):
                    frames.append(renderer.render(camera_id="palm_front"))
                break
        else:
            success_streak = 0
            
        if done[0]:
            break
            
    out_path = REPO_ROOT / out_gif
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames[::2], fps=15, loop=0)
    print(f"Saved demo GIF to {out_path} ({len(frames[::2])} frames)")

if __name__ == "__main__":
    sac_model = REPO_ROOT / "models/drac_sac_10M/best_model.zip"
    sac_vec = REPO_ROOT / "models/drac_sac_10M/vecnormalize.pkl"
    record_gif(algo="sac", model_path=sac_model, vec_path=sac_vec, out_gif="assets/sac_reorient_demo.gif", init="down", seed=7)
