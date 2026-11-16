# Benchmark Protocol and Experimental Analysis

## 1. Evaluation Metrics

To evaluate both high-level VLA inference efficiency and physical manipulation reliability, every benchmark trial records:

* **Inference Latency (ms):** Wall-clock time required for a single `policy.infer(obs)` call on the GPU after step-0 XLA JIT compilation.
* **End-to-End (E2E) Control Latency (ms):** Total wall-clock time per closed-loop VLA replanning cycle, including dual-camera offscreen rendering, observation formatting, GPU inference, kinematic/synergy projection, and physics stepping.
* **Episode Completion Time (s):** Total wall-clock duration of an evaluation episode.
* **Task Metrics (17-DoF In-Hand Cube Reorientation):** Goal Alignment Rate ($\theta_{\text{red,up}} < 15^\circ$), Mean Hold Streak (consecutive stable steps inside the $15^\circ$ cone out of 25 required steps), Held Task Success Rate ($\text{streak} \ge 25$), Cube Drop Rate ($Z_{\text{cube}} < 0.05\text{ m}$), Minimum Angle Error ($^\circ$), and Final Angle Error ($^\circ$).

---

## 2. Benchmark Results (`OrcaHandRightCubeOrientation`)

Source data: [`results/cube_reorient_metrics.csv`](../results/cube_reorient_metrics.csv)

| Experiment | Mode | Camera | Trials | Infer (ms) | E2E (ms) | Time (s) | Goal Align (<15°) | Hold Streak (/25) | Task Success (%) | Drop Rate (%) | Min Angle (°) | Final Angle (°) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Standalone $\pi_0$ — Naive Direct Mapping** | `direct_17dof` | `palm_front` | 10 | 289.18 | 424.04 | 2.12 | 0.0% | 0.0 | **0.0%** | 100.0% | 25.76° | 25.76° *(Drop)* |
| **2. Standalone $\pi_0$ — Relative + 17-DoF Synergy** | `synergy_reorient` | `palm_front` | 10 | 290.06 | 432.26 | 2.59 | 0.0% | 0.0 | **0.0%** | 0.0% | 89.37° | 97.06° |
| **3. $\pi_0$ Integrated + Scripted Skills (5 Starts, Res=0.1)** | `pi0_rl_hybrid` | `palm_front` | 5 | 278.85 | 403.36 | 3.60 | 100.0% | 20.6 | **80.0%** | 20.0% | 7.25° | 19.01° |
| **4. $\pi_0$ Integrated + Scripted Skills (Zero-Res Ablation, Res=0.0)** | `pi0_rl_hybrid` | `palm_front` | 5 | 274.11 | 397.11 | 3.81 | 100.0% | 20.6 | **80.0%** | 20.0% | 7.34° | 38.13° *(11.68° kept)* |
| **5. $\pi_0$ Integrated + 10M-Step PPO Policy (No Scripted Trajs, Res=0.0)** | `pi0_rl_hybrid` | `palm_front` | 10 | 300.03 | 465.98 | 4.04 | 80.0% | 19.5 | **70.0%** | 0.0% | 17.32° | 18.30° *(8.55° aligned)* |

### Key Findings
1. **Inference Efficiency:** `pi0_libero` achieves a mean inference latency of `274–300 ms` (`~3.4 Hz`) on the 8 GB RTX 4070 Laptop GPU, which is **7.4x faster** than the `2,144 ms` single-step latency measured for `OpenVLA-7B` on the same environment.
2. **Effect of Relative Joint Control and In-Hand Synergy:** In Experiment 1 (`direct_17dof`), applying raw VLA predictions directly to the 17 actuators tilts the wrist downward and ejects the cube within 25 simulation steps (`100%` drop rate). Wrapping the environment with `RelativeJointTargetAction` (`max_delta_frac=0.05`) and `in_hand_synergy_delta` (Experiment 2) reduces the drop rate to **`0.0%`** and rotates $90^\circ$ side orientations to `63.8°–74.1°` (overall 10-trial mean `89.37°` min angle across $90^\circ$ and $180^\circ$ starts), but zero-shot VLA outputs alone do not complete multi-step finger gaiting (`0/10` trials).
3. **$\pi_0$ Residual Ablation & Low-Level Control (`scripted` vs. `ppo_10m`):** Comparing Experiment 3 (`residual_scale=0.1`) against Experiment 4 (`residual_scale=0.0`) shows identical task success (`80.0%` and `20.6/25` hold streak across the 5 canonical starts), demonstrating that in `scripted` mode, $\pi_0$ is integrated into the closed-loop visual pipeline while the 100 Hz scripted skill primitives execute the gait. Replacing the scripted trajectories entirely with our trained 10M-step `PPO` policy (`--low-level ppo_10m`, Experiment 5) achieves **`70.0%` held success (`80.0%` goal alignment, `0.0%` drop rate)** across the 10 GPU trials (and **`67.0%` pure / `79.0%` latched held success** over 200 fixed seeds in Section 3).

---

## 3. Matched Multi-Algorithm RL Benchmark on DRAC `Nibi` (`PPO`, `SAC`, `TD3` — 200 Fixed Seeds)

Source data: [`results/rl_reorient_benchmark.csv`](../results/rl_reorient_benchmark.csv)

Trained on the Digital Research Alliance of Canada (DRAC) `Nibi` CPU cluster (`16` parallel `SubprocVecEnv` simulation workers per algorithm, `RelativeJointTargetAction` with `max_delta_frac=0.05`, `VecNormalize` observation/reward normalization, randomized initial orientations `init=random`, evaluated deterministically across **`200` fixed seeds `20000–20199`**):

| Algorithm | Action Mode | Eval Episodes | Policy Latency (ms) | Goal Align (<15°) | `EvalCallback` (Best / Peak) | Mean Hold Streak (/25) | Held Success (Pure RL / +Latch) | Cube Drop Rate (%) | Min Angle (All / Aligned) | Final Angle (All / Aligned) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PPO** (`ppo_reorient_random_10M`) | `relative` | 200 | 0.214 | **84.0%** | 80.0% / **95.0%** | **18.6** | **67.0% / 79.0%** | **0.0%** | **16.45°** *(8.75°)* | **19.20°** *(11.45°)* |
| **SAC** (`sac_reorient_random_10M`) | `relative` | 200 | 0.257 | 72.5% | **90.0% / 90.0%** | 16.6 | **64.5% / 69.5%** | **0.0%** | 20.58° *(6.89°)* | 24.58° *(8.84°)* |
| **TD3** (`td3_reorient_random_10M`) | `relative` | 200 | **0.184** | 38.5% | 70.0% / 70.0% | 8.4 | 33.5% / 33.5% | 4.5% | 47.19° *(5.84°)* | 55.78° *(12.29°)* |

### Key RL Findings (200-Episode Statistical Analysis)
1. **PPO vs. SAC Over 200 Fixed Seeds:** Over a statistically powered sample of **`N = 200` fixed seeds (`20000–20199`)**, on-policy `PPO` achieves **`67.0%` standalone held success (`134/200`)** and **`84.0%` goal alignment (`168/200`)**, while off-policy `SAC` achieves **`64.5%` standalone held success (`129/200`)** and **`72.5%` goal alignment (`145/200`)**, both with **`0.0%` drop rate**. Standalone 25-step held success between `PPO` and `SAC` is statistically indistinguishable ($p = 0.67$, Fisher's exact test), whereas `PPO` achieves significantly higher goal alignment ($p = 0.007$) and trains **5.1× faster** (`69.2 min` at `2,409 FPS` vs. `352.4 min` at `473 FPS`).
2. **Effect of Terminal Hold Latch (`HOLD_STILL`):** Because `PPO` reaches the `<15°` goal cone in `84.0%` of episodes but occasionally experiences brief angular-velocity spikes (`||ω|| > 1.0 rad/s`) that reset the 25-step hold counter, adding the zero-delta `HOLD_STILL` latch once inside the goal cone boosts `PPO`'s 200-episode held success from **`67.0%` to `79.0%` (`158/200`, $p = 0.0095$)**—matching its **`80.0%` best-checkpoint `EvalCallback` (`95.0%` peak)** performance. Meanwhile, across aligned episodes, `SAC` achieves the tightest mean final angle error (**`8.84°` final / `6.89°` min** vs. `11.45°` final / `8.75°` min for `PPO`).
3. **Deterministic Exploration Trade-off:** `TD3` completed `10,000,000` steps in `316.8 minutes` (`527 FPS`) and reached `33.5%` held success (`38.5%` goal alignment) over 200 episodes—significantly below both `PPO` and `SAC` ($p < 10^{-9}$, Fisher's exact test)—confirming that fixed Gaussian action noise (`sigma=0.15`) explores contact-rich multi-axis rotations less reliably than stochastic policies.


