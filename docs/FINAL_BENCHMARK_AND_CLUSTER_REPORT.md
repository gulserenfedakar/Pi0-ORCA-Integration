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

---

## 2. Quantitative Results & Comparison

### Part A: VLA Inference & Hierarchical Coupling (Local RTX 4070 Laptop GPU, 8 GB VRAM)

*Task criteria: Reorient cube inside the palm until red face is world-up within $15^\circ$ error cone and maintain stable hold for 25 consecutive steps ($0.25\text{ s}$) without dropping ($Z < 0.05\text{ m}$).*

| Benchmark Configuration | Control Mode | Viewpoint | Inference Latency | E2E Latency | Success Rate | Drop Rate | Final Angle Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Level 1: Standalone $\pi_0$ (Direct)** | `direct_17dof` | `palm_front` | 302.50 ms | 459.18 ms | **0.0%** | 100.0% | 30.09° |
| **Level 2: Standalone $\pi_0$ (Synergy)** | `synergy_reorient`| `palm_front` | 287.13 ms | 443.12 ms | **0.0%** | **0.0%** | 90.20° |
| **Level 2: Standalone $\pi_0$ (Top View)** | `synergy_reorient`| `palm_top` | 289.35 ms | 446.57 ms | **0.0%** | **0.0%** | 90.41° |
| **Level 3: Hybrid $\pi_0$ + RL (Nominal)** | `pi0_rl_hybrid` | `palm_front` | 289.16 ms | 444.09 ms | **100.0%** | **0.0%** | **13.81°** |
| **Level 3: Hybrid $\pi_0$ + RL (Randomized)** | `pi0_rl_hybrid` | `palm_front` | 295.62 ms | 450.47 ms | **80.0%** | 20.0% | **18.98°** |

---

### Part B: Matched Multi-Algorithm RL Benchmark (DRAC `Nibi` Cluster — 10,000,000 Steps)

*Trained under identical environments (`RelativeJointTargetAction`, `max_delta_frac=0.05`, `VecNormalize`, 16 parallel CPU workers) across randomized initial orientations (`init=random`), evaluated over 20 unseen seeds.*

| Algorithm | Training Steps | Total Wall Time | Throughput | Goal Align ($<15^\circ$) | Mean Hold Streak (/25) | Held Success Rate | Cube Drop Rate | Final Angle Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SAC** | 10,000,000 | 352.4 min (5.87 h) | 473 FPS | **85.0%** | **21.0 / 25** | **80.0%** | **0.0%** | **18.37°** |
| **PPO** | 10,000,000 | **69.2 min (1.15 h)** | **2,409 FPS** | **85.0%** | 16.6 / 25 | **50.0%** *(Eval: 75%)*| **0.0%** | 19.04° |
| **TD3** | 10,000,000 | 316.8 min (5.28 h) | 527 FPS | 60.0% | 13.8 / 25 | **55.0%** | 5.0% | 46.36° |

---

## 3. Key Engineering & Algorithmic Findings

1. **Why Pure Zero-Shot VLA Fails on Dexterous Hands:**
   Direct actuator mapping causes catastrophic wrist deflection within 25 steps. Adding a relative joint boundary (`RelativeJointTargetAction`) eliminates drops (0% drop rate) and begins rotation, but flow-matching outputs cannot sustain the high-frequency contact gaiting needed to complete the turn.
2. **Why SAC Outperforms PPO & TD3 in Sustained Holding:**
   While on-policy `PPO` trains over 5x faster due to synchronous rollout collection, off-policy `SAC` achieves superior asymptotic stability (**80% held success**, 0% drops). The maximum-entropy objective acts as an automatic exploration and damping regularizer, preventing finger chattering once the cube enters the $15^\circ$ success cone.
3. **The Role of Hierarchical Coupling:**
   By combining $\pi_0$'s semantic scene interpretation (phase identification: *flip, roll, hold*) with 20 Hz low-level RL execution, the hybrid policy achieves **100% success** on nominal 180° reorientation and **80% success** under arbitrary 3D initial orientations.

---

## 4. Deliverables & Artifacts

- **Source Code Repository:** [https://github.com/gulserenfedakar/Pi0-ORCA-Integration](https://github.com/gulserenfedakar/Pi0-ORCA-Integration)
- **Trained Model Weights:** `models/drac_sac_10M/`, `models/drac_ppo_10M/`, `models/drac_td3_10M/` (including checkpoints and normalization statistics)
- **Animated Demonstration:** `assets/sac_reorient_demo.gif`
- **Cluster Deployment Scripts:** `scripts/slurm_drac_rl_benchmark.sh`
