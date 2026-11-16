# Comprehensive Experimental Benchmark & Cluster Report
**Investigation:** 17-DoF ETH ORCA Dexterous Hand In-Hand Cube Reorientation (`OrcaHandRightCubeOrientation`)  
**Author:** Gulseren Fedakar Gonul  
**Computing Platforms:** Digital Research Alliance of Canada (DRAC) `Nibi` Cluster & Local NVIDIA RTX 4070 (8 GB) Platform  

---

## 1. Executive Summary

This report documents the experimental evaluation of high-degree-of-freedom dexterous manipulation using Physical Intelligence's Vision-Language-Action model ($\pi_0$) and deep reinforcement learning (RL) policies on the **17-DoF ETH ORCA Hand**. 

Direct end-to-end VLA deployment on dexterous multi-finger hardware faces two fundamental bottlenecks:
1. **Embodiment and Action Space Discrepancy:** $\pi_0$ outputs low-dimensional 7D action chunks designed for 1-DoF parallel-jaw grippers. Mapping VLA outputs directly to 17 coupled tendon actuators causes severe joint oscillations, wrist drop, and instant object ejection (100% drop rate).
2. **Control Latency vs. Contact Frequency:** 3B-parameter VLA inference runs at ~3.5 Hz on consumer GPU hardware, whereas fine finger gaiting and velocity damping operate at a control frequency of $100\text{ Hz}$ ($\Delta t = 0.01\text{ s}$).

To evaluate these regimes, we benchmarked standalone $\pi_0$ configurations alongside $\pi_0$-integrated low-level controllers and deep RL policies (`PPO`, `SAC`, `TD3`) trained at scale on the Digital Research Alliance of Canada (DRAC) `Nibi` cluster across 10,000,000 steps.

---

## 2. Quantitative Results & Comparison

### Part A: VLA Inference, Residual Ablation & Low-Level Coupling (Local RTX 4070 Laptop GPU, 8 GB VRAM)

*Task criteria: Reorient cube inside the palm until red face is world-up within $15^\circ$ error cone and maintain stationary hold ($\|\omega\| \le 1.0\text{ rad/s}$, $\|v\| \le 0.1\text{ m/s}$) for 25 consecutive control steps ($0.25\text{ s}$ at $100\text{ Hz}$) without dropping ($Z < 0.05\text{ m}$).*

| Benchmark Configuration | Control Mode | Viewpoint | Trials | Inference Latency | E2E Latency | Goal Align ($<15^\circ$) | Hold Streak (/25) | Held Success | Drop Rate | Final Angle Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Standalone $\pi_0$ — Naive Direct Mapping** | `direct_17dof` | `palm_front` | 10 | 289.18 ms | 424.04 ms | 0.0% | 0.0 | **0.0%** | 100.0% | 25.76° *(Dropped)* |
| **2. Standalone $\pi_0$ — Relative + 17-DoF Synergy** | `synergy_reorient`| `palm_front` | 10 | 290.06 ms | 432.26 ms | 0.0% | 0.0 | **0.0%** | **0.0%** | 97.06° *(Min: 89.37°)* |
| **3. $\pi_0$ Integrated + Scripted Skills (5 Starts, Res=0.1)** | `pi0_rl_hybrid` | `palm_front` | 5 | 278.85 ms | 403.36 ms | **100.0%** | 20.6 | **80.0%** | 20.0% | **19.01°** *(Min: 7.25°)* |
| **4. $\pi_0$ Integrated + Scripted Skills (Zero-Res Ablation, Res=0.0)** | `pi0_rl_hybrid` | `palm_front` | 5 | 274.11 ms | 397.11 ms | **100.0%** | 20.6 | **80.0%** | 20.0% | 38.13° *(11.68° kept)* |
| **5. $\pi_0$ Integrated + 10M-Step PPO Policy (No Scripted Trajs, Res=0.0)** | `pi0_rl_hybrid` | `palm_front` | 10 | 300.03 ms | 465.98 ms | **80.0%** | 19.5 | **70.0%** | **0.0%** | **18.30°** *(8.55° aligned)* |

> *Note on Residual & Low-Level Ablation (Configs 3–5):* Ablating $\pi_0$'s action residual to zero (`--residual-scale 0.0`, Config 4) yields identical task success (`80.0%` and `20.6/25` hold streak across the 5 axis-aligned initial orientations) to Config 3 (`--residual-scale 0.1`). Therefore, in `scripted` mode, $\pi_0$ is integrated into the closed-loop visual inference pipeline (`~274–290 ms` per chunk) while the 100 Hz scripted low-level skill primitives execute the physical finger gaiting. Replacing the scripted trajectories completely with our trained 10M-step `PPO` policy (`--low-level ppo_10m` + `HOLD_STILL` latch, Config 5) achieves **`70.0%` held success (`80.0%` goal alignment, `0.0%` drop rate)** across the 10 GPU trials (`p = 0.003` vs. Standalone $\pi_0$ on $N=10$, and `79.0%` over 200 seeds in Part B).

---

### Part B: Matched Multi-Algorithm RL Benchmark (DRAC `Nibi` Cluster — 10,000,000 Steps, 200 Fixed Seeds)

*Trained under identical environments (`RelativeJointTargetAction`, `max_delta_frac=0.05`, `VecNormalize`, 16 parallel `SubprocVecEnv` CPU workers) across randomized initial orientations (`init=random`), evaluated deterministically over **200 fixed seeds (`seeds 20000–20199`)** via `scripts/evaluate_rl_reorient.py --runs-dir models --episodes 200`.*

| Algorithm | Training Steps | Total Wall Time | Throughput | Goal Align ($<15^\circ$) | `EvalCallback` (Best / Peak) | 200-Seed Hold Streak | 200-Seed Held Success (Pure / +Latch) | Drop Rate | Final Angle (All 200 / Aligned) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PPO** | 10,002,432 | **69.2 min (1.15 h)** | **2,409 FPS** | **84.0%** | 80.0% / **95.0%** | **18.6 / 25** | **67.0% / 79.0%** | **0.0%** | **19.20°** *(11.45° aligned)* |
| **SAC** | 10,000,000 | 352.4 min (5.87 h) | 473 FPS | 72.5% | **90.0% / 90.0%** | 16.6 / 25 | **64.5% / 69.5%** | **0.0%** | 24.58° *(8.84° aligned)* |
| **TD3** | 10,000,000 | 316.8 min (5.28 h) | 527 FPS | 38.5% | 70.0% / 70.0% | 8.4 / 25 | 33.5% / 33.5% | 4.5% | 55.78° *(12.29° aligned)* |

> *Statistical Significance & Angle Error Note (`N = 200` Fixed Seeds):* Evaluated over 200 fixed seeds (`seeds 20000–20199`), standalone `PPO` (`67.0%`, `134/200`) and `SAC` (`64.5%`, `129/200`) are statistically tied on 25-step held success ($p = 0.67$, Fisher's exact test), while `PPO` achieves significantly higher goal alignment (**`84.0%` vs. `72.5%`, $p = 0.007$**) and higher latched held success (**`79.0%` vs. `69.5%`, $p = 0.039$**) in **5.1× less training wall-clock time** (`69.2 min` vs. `352.4 min`). Both `PPO` and `SAC` significantly outperform `TD3` (`33.5%`, $p < 10^{-9}$). Across aligned episodes, `SAC` achieves the lowest mean final angle error (`8.84°` final / `6.89°` min vs. `11.45°` final / `8.75°` min for `PPO`).

---

## 3. Key Engineering & Algorithmic Findings

1. **Why Pure Zero-Shot VLA Fails on Dexterous Hands:**
   Direct actuator mapping causes catastrophic wrist deflection within 25 steps (`100%` drop rate). Adding a relative joint boundary (`RelativeJointTargetAction`) and palmar synergy (`in_hand_synergy_delta`) eliminates drops (`0%` drop rate) and initiates rotation (`90° → 61.5°–74.1°` on side starts), but 3.5 Hz flow-matching outputs alone cannot sustain the 100 Hz multi-finger contact gaiting required to complete a $180^\circ$ flip (`0/10` trials).
2. **Resolving the Short-Horizon (200K-Step) SAC Bottleneck & 200-Episode PPO vs. SAC Comparison:**
   Under short-horizon training (`200,000` / `0.2M` steps), `SAC` appears stuck near `16%` success because off-policy replay buffers require `~2M–4M` steps to populate contact transitions across 17 degrees of freedom. Scaling to `10,000,000` steps on DRAC `Nibi` and evaluating over **200 fixed seeds** shows that `SAC` rises to **`64.5%` standalone held success (`69.5%` with hold latch, `90.0%` `EvalCallback`)**, matching `PPO`'s **`67.0%` standalone held success (`79.0%` with hold latch, `80.0%` best / `95.0%` peak `EvalCallback`)**, while `PPO` trains **5.1× faster** (`69.2 min` at `2,409 FPS`) and reaches `<15°` in **`84.0%`** of episodes.
3. **Transparent Ablation of $\pi_0$-Integrated Control (Residual & Low-Level Skills vs. 10M-Step RL):**
   Setting $\pi_0$'s residual to zero (`--residual-scale 0.0`) confirms that a zero-shot pretrained 7D gripper VLA does not contribute low-level finger-gaiting deltas (`80.0%` with or without residual under scripted skills). Instead, physical reorientation is driven by the 100 Hz low-level controller (achieving **`70.0%` over 10 GPU trials** and **`79.0%` over 200 seeds** when using our 10M-step `PPO` policy + `HOLD_STILL` latch without any scripted trajectories), while $\pi_0$ operates in the closed-loop visual inference pipeline (`~274–300 ms`).

---

## 4. Deliverables & Artifacts

- **Source Code Repository:** [https://github.com/gulserenfedakar/Pi0-ORCA-Integration](https://github.com/gulserenfedakar/Pi0-ORCA-Integration)
- **Trained Model Weights:** `models/drac_sac_10M/`, `models/drac_ppo_10M/`, `models/drac_td3_10M/` (including checkpoints and normalization statistics)
- **Animated Demonstration:** `assets/sac_reorient_demo.gif`
- **Cluster Deployment Scripts:** `scripts/slurm_drac_rl_benchmark.sh`
- **Presentation Slides (`1920x1080` JPEGs):** `assets/slides/slide1_architecture_pipeline.jpg` through `slide4_rollout_filmstrip_and_takeaways.jpg`

---

## 5. Methodological Notes & Defense Q&A Guide

1. **How `OrcaVLAObservationBuilder` Maps a Stationary 17-DoF Hand into `pi0_libero`'s 8D State:**
   - Standard `pi0_libero` expects an 8D proprioceptive vector $[x, y, z, r_x, r_y, r_z, q_1, q_2]$ representing a mobile end-effector pose and 2-finger gripper opening. Because the 17-DoF ORCA hand in `OrcaHandRightCubeOrientation` is mounted on a stationary base, passing a static palm pose would carry zero task-progress information. Instead, `OrcaVLAObservationBuilder.build_libero_eef_state()` (`orca_vla/observations.py`) populates the first 6 dimensions with `task_cube`'s 3D Cartesian world position and 3D axis-angle orientation and the final 2 dimensions with normalized palmar closure state.
2. **Division of Labor & Residual Ablation in `pi0_rl_hybrid` (Part A) vs. Standalone 10M-Step DRAC RL (Part B):**
   - In Part A (`scripts/run_cube_reorient.py`, `mode="pi0_rl_hybrid"`), $\pi_0$ is queried every 10 control steps (`~3.4 Hz`) on dual `224×224` palmar cameras (`palm_front`, `palm_top`). Setting $\pi_0$'s action residual to zero (`--residual-scale 0.0`, Config 4) produces identical task success (`80.0%` across the 5 axis-aligned starts) to Config 3 (`--residual-scale 0.1`), confirming that in `scripted` mode, $\pi_0$ is integrated into the closed-loop visual pipeline while the 100 Hz low-level skills (`reorient_expert_trajs.npz`) perform the finger gaiting. Replacing the scripted trajectories entirely with our 10M-step `PPO` policy (`--low-level ppo_10m` + `HOLD_STILL` latch, Config 5) achieves **`70.0%` held success (`80.0%` goal alignment, `0.0%` drops)** across 10 GPU trials (`seeds 0–9`) and **`79.0%` held success (`84.0%` goal alignment, `0.0%` drops)** across 200 fixed seeds (`seeds 20000–20199`).
   - In Part B (`scripts/train_rl_reorient.py` and `scripts/evaluate_rl_reorient.py`), `PPO`, `SAC`, and `TD3` operate as **100 Hz closed-loop policies** ($\Delta t = 0.01\text{ s}$, `25` hold steps = $0.25\text{ s}$) trained for `10,000,000` steps on DRAC `Nibi` (`models/drac_{ppo,sac,td3}_10M/best_model.zip`), evaluated deterministically across **200 fixed seeds (`seeds 20000–20199`)** both as pure RL and with the terminal hold latch.
3. **Observation Space Dimensionality (`51D` Gymnasium Vector vs. `43D` Physical Kinematic State):**
   - In `OrcaHandRightCubeOrientation` (`version="v2"`), the physical kinematic state consists of **43 dimensions** (`17` hand joint positions `qpos` + `17` hand joint velocities `qvel` = `34D`, plus `3D` cube position + `6D` cube linear/angular velocity = `9D`). Including the `4D` cube quaternion and `4D` red-face goal features (`3D` red-face normal vector + `1D` cosine alignment scalar) yields a total **51-dimensional** Gymnasium `Box` observation vector.
4. **200-Episode Statistical Comparison (`PPO` vs. `SAC` vs. `TD3`):**
   - Over `N = 200` fixed seeds, standalone `PPO` (`67.0%` held success, `84.0%` goal alignment, `0.0%` drops) and `SAC` (`64.5%` held success, `72.5%` goal alignment, `0.0%` drops) are statistically tied on 25-step held success ($p = 0.67$, Fisher's exact test), while `PPO` achieves significantly higher goal alignment ($p = 0.007$) and higher latched held success (`79.0%` vs. `69.5%`, $p = 0.039$) in **5.1× less training wall-clock time** (`69.2 min` vs. `352.4 min`). Both `PPO` and `SAC` significantly outperform `TD3` (`33.5%` held success, $p < 10^{-9}$).
