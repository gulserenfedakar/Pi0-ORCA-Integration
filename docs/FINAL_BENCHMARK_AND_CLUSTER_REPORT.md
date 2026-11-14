# Comprehensive Experimental Benchmark & Cluster Report
**Project:** SEP 730 — The Rise of Intelligent Machines (Group 1)  
**Investigation:** 17-DoF ETH ORCA Dexterous Hand In-Hand Cube Reorientation (`OrcaHandRightCubeOrientation`)  
**Lead Contributor:** Gulseren Fedakar Gonul (`fedakarg@mcmaster.ca`)  
**Computing Clusters:** Digital Research Alliance of Canada (DRAC) `Nibi` Cluster & Local NVIDIA RTX 4070 (8 GB) Platform  

---

## 1. Executive Summary

This report documents the experimental evaluation of high-degree-of-freedom dexterous manipulation using Physical Intelligence's Vision-Language-Action model ($\pi_0$) and deep reinforcement learning (RL) policies on the **17-DoF ETH ORCA Hand**. 

Direct end-to-end VLA deployment on dexterous multi-finger hardware faces two fundamental bottlenecks:
1. **Embodiment and Action Space Discrepancy:** $\pi_0$ is pretrained primarily on standard robotic arms with 1-DoF parallel-jaw grippers. Mapping VLA outputs directly to 17 coupled tendon actuators causes severe joint oscillations, wrist drop, and instant object ejection (100% drop rate).
2. **Control Latency vs. Contact Frequency:** 3B-parameter VLA inference runs at ~3.5 Hz on consumer GPU hardware, whereas fine finger gaiting and velocity damping require control frequencies of $\ge 20\text{ Hz}$.

To overcome these limitations, we designed a **hierarchical architecture** where high-level visual semantic guidance ($\pi_0$) is coupled with high-frequency relative joint control and deep RL policies trained at scale on the Digital Research Alliance of Canada (DRAC) `Nibi` cluster across 10,000,000 steps.

### Project Progression & Team Integration (SEP 730 — Group 1)
- **Phase 1 (Midterm — 24-DoF Arm+Hand Tabletop Pick-and-Place, `scene_right_arm_tabletop.xml`):** Validated `pi0_libero` spatial reaching (`269 ms` inference latency, `100%` front-camera reach vs. `0%` side-camera sensitivity) and `BEST_GRASP` synergy lifting (`100%` nominal, `80%` randomized).
- **Phase 2 (Post-Midterm / Final — 17-DoF Hand-Only In-Hand Cube Reorientation, `OrcaHandRightCubeOrientation`):** Aligned simulation with our physical 17-DoF ORCA hand hardware (which operates as a stationary dexterous hand without a Franka arm), building on the shared simulation environment and relative action wrappers by **Yuhui (Joe) Wu**, the `OpenVLA-7B` (`2,144.7 ms`) and initial 200K-step `SAC` (`16%` success) baselines by **Yi Liu**, and the modular `orca_vla` observation/action interfaces co-designed with **Debaditya Ghosh** ($\pi_{0.5}$).

---

## 2. Quantitative Results & Comparison

### Part A: VLA Inference & Hierarchical Coupling (Local RTX 4070 Laptop GPU, 8 GB VRAM)

*Task criteria: Reorient cube inside the palm until red face is world-up within $15^\circ$ error cone and maintain stationary hold ($\|\omega\| \le 1.0\text{ rad/s}$, $\|v\| \le 0.1\text{ m/s}$) for 25 consecutive control steps ($1.25\text{ s}$ at $20\text{ Hz}$) without dropping ($Z < 0.05\text{ m}$).*

| Benchmark Configuration | Control Mode | Viewpoint | Inference Latency | E2E Latency | Goal Align ($<15^\circ$) | Hold Streak (/25) | Held Success | Drop Rate | Final Angle Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Level 1: Standalone $\pi_0$ (Direct)** | `direct_17dof` | `palm_front` | 302.50 ms | 459.18 ms | 0.0% | 0.0 | **0.0%** | 100.0% | 30.09° *(Dropped)* |
| **Level 2: Standalone $\pi_0$ (Synergy)** | `synergy_reorient`| `palm_front` | 287.13 ms | 443.12 ms | 0.0% | 0.0 | **0.0%** | **0.0%** | 90.20° *(Min: 72.93°)* |
| **Level 2: Standalone $\pi_0$ (Top View)** | `synergy_reorient`| `palm_top` | 289.35 ms | 446.57 ms | 0.0% | 0.0 | **0.0%** | **0.0%** | 90.41° *(Min: 72.96°)* |
| **Level 3: Hybrid $\pi_0$ + RL (Nominal)** | `pi0_rl_hybrid` | `palm_front` | 289.16 ms | 444.09 ms | **100.0%** | **25.0** | **100.0%** | **0.0%** | **13.81°** *(Min: 2.83°)* |
| **Level 3: Hybrid $\pi_0$ + RL (Randomized)** | `pi0_rl_hybrid` | `palm_front` | 295.62 ms | 450.47 ms | **100.0%** | 20.6 | **80.0%** | 20.0% | **18.98°** *(Min: 7.26°)* |

> *Note on Level 1 Angle:* In Level 1 (`direct_17dof`), uncoordinated wrist deflection ejects the cube within `2.30 s` (`100%` drop rate); its `30.09°` angle reflects uncontrolled tumbling during free-fall rather than in-palm alignment. In Level 2 (`synergy_reorient`), the cube is safely cradled (`0%` drop rate) and rotated by `~17°` (`90° → 72.9°`).

---

### Part B: Matched Multi-Algorithm RL Benchmark (DRAC `Nibi` Cluster — 10,000,000 Steps)

*Trained under identical environments (`RelativeJointTargetAction`, `max_delta_frac=0.05`, `VecNormalize`, 16 parallel `SubprocVecEnv` CPU workers) across randomized initial orientations (`init=random`), evaluated deterministically over 20 unseen seeds (`seeds 20000–20019`).*

| Algorithm | Training Steps | Total Wall Time | Throughput | Goal Align ($<15^\circ$) | `EvalCallback` (Best / Peak) | 20-Seed Hold Streak | 20-Seed Held Success | Drop Rate | Final Angle (All 20 / Aligned) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SAC** | 10,000,000 | 352.4 min (5.87 h) | 473 FPS | **85.0%** | **90.0% / 90.0%** | **21.0 / 25** | **80.0%** | **0.0%** | **18.37°** *(6.55° aligned)* |
| **PPO** | 10,002,432 | **69.2 min (1.15 h)** | **2,409 FPS** | **85.0%** | 80.0% / **95.0%** | 16.6 / 25 | **50.0%** *(75% Final Eval)* | **0.0%** | 19.04° *(12.00° aligned)* |
| **TD3** | 10,000,000 | 316.8 min (5.28 h) | 527 FPS | 60.0% | 70.0% / 70.0% | 13.8 / 25 | **55.0%** | 5.0% | 46.36° *(11.70° aligned)* |

> *Note on Angle Error Averages:* `18.37°` (SAC) and `19.04°` (PPO) are unfiltered arithmetic means across all 20 test episodes, including the `3/20` (`15%`) unaligned episodes that remain near `~57°–71°`. Across the **85% aligned episodes (`17/20`)**, **SAC achieves `5.56°` mean minimum error and `6.55°` mean final error** (20-episode median: `7.10°` min / `8.01°` final), and **PPO achieves `8.49°` mean minimum error and `12.00°` mean final error** (20-episode median: `7.44°` min / `12.33°` final).

---

## 3. Key Engineering & Algorithmic Findings

1. **Why Pure Zero-Shot VLA Fails on Dexterous Hands:**
   Direct actuator mapping causes catastrophic wrist deflection within 25 steps (`100%` drop rate). Adding a relative joint boundary (`RelativeJointTargetAction`) and palmar synergy (`in_hand_synergy_delta`) eliminates drops (`0%` drop rate) and initiates rotation (`90° → 72.9°`), but 3.5 Hz flow-matching outputs alone cannot sustain the 20 Hz multi-finger contact gaiting required to complete a $180^\circ$ flip.
2. **Resolving the Midterm 200K-Step SAC Bottleneck & Why SAC Outperforms PPO in Sustained Holding:**
   At the Midterm, `SAC` appeared stuck at `16%` after `200,000` (`0.2M`) steps because off-policy replay buffers require `~2M–4M` steps to populate contact transitions across 17 degrees of freedom. Scaling to `10,000,000` steps on DRAC `Nibi` shows that while on-policy `PPO` trains **5.1× faster** (`69.2 min` at `2,409 FPS`) and reaches the `<15°` cone at the same **`85.0%` rate** as `SAC` (`95.0%` peak `EvalCallback`), `PPO` experiences minor finger micro-oscillation (`||ω|| > 1.0 rad/s`) after `10–12` hold steps on side-axis rotations (`50.0%` 25-step hold on unseen seeds). Off-policy maximum-entropy `SAC` learns smoother terminal velocity damping, achieving **`80.0%` 25-step held success** (`90.0%` `EvalCallback`) with `0.0%` drops.
3. **Complementary Roles in the Hierarchical Hybrid ($\pi_0$ + RL) Architecture:**
   While a standalone RL policy relies on 43D proprioceptive state and cannot interpret natural-language task instructions, coupling high-level $\pi_0$ visual-language grounding, bounded residual modulation, and discrete phase gating (`FLIP_180`, `ROLL_X`, `ROLL_Y`, `HOLD_STILL`) with 20 Hz low-level relative-joint skill/RL execution achieves **`100.0%` held success** on the nominal $180^\circ$ flip and **`80.0%` held success** (`100.0%` goal alignment) across all five canonical initial cube orientations.

---

## 4. Deliverables & Artifacts

- **Source Code Repository:** [https://github.com/gulserenfedakar/Pi0-ORCA-Integration](https://github.com/gulserenfedakar/Pi0-ORCA-Integration)
- **Trained Model Weights:** `models/drac_sac_10M/`, `models/drac_ppo_10M/`, `models/drac_td3_10M/` (including checkpoints and normalization statistics)
- **Animated Demonstration:** `assets/sac_reorient_demo.gif`
- **Cluster Deployment Scripts:** `scripts/slurm_drac_rl_benchmark.sh`
