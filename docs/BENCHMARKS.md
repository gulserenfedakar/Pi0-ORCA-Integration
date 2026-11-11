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

