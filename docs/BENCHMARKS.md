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
| **1. Standalone $\pi_0$ — Naive Direct Mapping** | `direct_17dof` | `palm_front` | 2 | 302.50 | 459.18 | 2.30 | 0.0% | 0.0 | **0.0%** | 100.0% | 30.09° | 30.09° |
| **2. Standalone $\pi_0$ — Relative Control + 17-DoF Synergy** | `synergy_reorient` | `palm_front` | 2 | 287.13 | 443.12 | 4.43 | 0.0% | 0.0 | **0.0%** | 0.0% | 72.93° | 90.20° |
| **3. Standalone $\pi_0$ — Relative Control + 17-DoF Synergy** | `synergy_reorient` | `palm_top` | 2 | 289.35 | 446.57 | 4.47 | 0.0% | 0.0 | **0.0%** | 0.0% | 72.96° | 90.41° |
| **4. Hybrid $\pi_0$ + RL (PPO) — Nominal $180^\circ$ Flip** | `pi0_rl_hybrid` | `palm_front` | 2 | 289.16 | 444.09 | 4.44 | 100.0% | 25.0 | **100.0%** | 0.0% | 2.83° | 13.81° |
| **5. Hybrid $\pi_0$ + RL (PPO) — Randomized 5 Orientations** | `pi0_rl_hybrid` | `palm_front` | 5 | 295.62 | 450.47 | 4.03 | 100.0% | 20.6 | **80.0%** | 20.0% | 7.26° | 18.98° |

### Key Findings
1. **Inference Efficiency:** `pi0_libero` achieves a mean inference latency of `287–302 ms` (`~3.4 Hz`) on the 8 GB RTX 4070 Laptop GPU, which is **7.4x faster** than the `2,144 ms` single-step latency measured for `OpenVLA-7B` on the same environment.
2. **Effect of Relative Joint Control and In-Hand Synergy:** In Experiment 1 (`direct_17dof`), applying raw VLA predictions directly to the 17 actuators tilts the wrist downward and ejects the cube within 25 simulation steps (`100%` drop rate). Wrapping the environment with `RelativeJointTargetAction` (`max_delta_frac=0.05`) and `in_hand_synergy_delta` (Experiments 2 and 3) reduces the drop rate to **`0.0%`** and rotates the cube by `~17°` (`90.0° -> 72.9°`), but zero-shot VLA outputs alone do not complete multi-step finger gaiting.
3. **Hybrid $\pi_0$ + RL Performance:** Combining high-level $\pi_0$ phase selection (`FLIP_180`, `ROLL_X`, `ROLL_Y`, `HOLD_STILL`) with low-level 20 Hz relative-joint RL control achieves **`100.0%` held success** on the nominal $180^\circ$ flip (`25/25` hold steps, final angle `13.81°`) and **`100.0%` goal alignment / `80.0%` held success (`4/5` trials)** across all five canonical initial cube orientations.

---

## 3. Matched Multi-Algorithm RL Benchmark on DRAC `Nibi` (`PPO`, `SAC`, `TD3`)

Source data: [`results/rl_reorient_benchmark.csv`](../results/rl_reorient_benchmark.csv)

Trained on the Digital Research Alliance of Canada (DRAC) `Nibi` CPU cluster (`16` parallel `SubprocVecEnv` simulation workers per algorithm, `RelativeJointTargetAction` with `max_delta_frac=0.05`, `VecNormalize` observation/reward normalization, randomized initial orientations `init=random`, evaluated deterministically across `20` unseen seeds):

| Algorithm | Action Mode | Eval Episodes | Policy Latency (ms) | Goal Align (<15°) | `EvalCallback` (Best / Peak) | Mean Hold Streak (/25) | Held Task Success (%) | Cube Drop Rate (%) | Min Angle (All / Aligned) | Final Angle (All / Aligned) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SAC** (`sac_reorient_random_10M`) | `relative` | 20 | 0.330 | **85.0%** | **90.0% / 90.0%** | **21.0** | **80.0%** | **0.0%** | 16.53° *(5.56°)* | **18.37°** *(6.55°)* |
| **PPO** (`ppo_reorient_random_10M`) | `relative` | 20 | 0.292 | **85.0%** | 80.0% / **95.0%** | 16.6 | **50.0%** *(75.0% Final Eval)* | **0.0%** | **14.70°** *(8.49°)* | 19.04° *(12.00°)* |
| **TD3** (`td3_reorient_random_10M`) | `relative` | 20 | **0.238** | 60.0% | 70.0% / 70.0% | 13.8 | **55.0%** | 5.0% | 34.43° *(7.07°)* | 46.36° *(11.70°)* |

### Key RL Findings
1. **Throughput vs. Sample Efficiency:** On-policy `PPO` completed all `10,002,432` environment steps in **69.2 minutes** (`2,409 FPS` across 16 CPU cores), reaching **`85.0%` goal alignment (`< 15°`)**, **`95.0%` peak / `80.0%` best-checkpoint `EvalCallback` success**, and **`0.0%` drop rate**.
2. **Sustained Hold Stability & Velocity Damping:** Off-policy maximum-entropy `SAC` (`10,000,000` steps in `352.4 minutes`, `473 FPS`) achieved the highest 25-step held success rate on the 20 unseen test seeds (**`80.0%`**, `90.0%` `EvalCallback`, mean hold streak `21.0 / 25` steps, `0.0%` drop rate). While `PPO` and `SAC` both align the cube `<15°` in `85.0%` (`17/20`) of unseen seeds, `PPO` experiences brief angular-velocity spikes (`||ω|| > 1.0 rad/s`) after `10–12` hold steps on side-axis rotations, whereas `SAC`'s entropy-regularized critic damps finger micro-oscillation inside the $15^\circ$ target cone. Across the `85%` aligned episodes, `SAC` achieves **`5.56°` mean minimum error and `6.55°` mean final error** (20-episode median: `7.10°` min / `8.01°` final), and `PPO` achieves **`8.49°` mean minimum error and `12.00°` mean final error** (20-episode median: `7.44°` min / `12.33°` final).
3. **Deterministic Exploration Trade-off:** `TD3` completed `10,000,000` steps in `316.8 minutes` (`527 FPS`) and reached `55.0%` held success (`60.0%` goal alignment), indicating that fixed Gaussian action noise (`sigma=0.15`) explores contact-rich multi-axis rotations less reliably than `SAC`'s state-dependent stochastic policy.


