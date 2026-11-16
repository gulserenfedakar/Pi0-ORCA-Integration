# System Architecture and Control Design

## 1. Problem Formulation

The ETH ORCA hand is a 17-DoF tendon-driven anthropomorphic hand comprising a 1-DoF wrist pitch actuator, four 3-DoF fingers (index, middle, ring, pinky; abduction, MCP flexion, PIP flexion), and a 4-DoF thumb (CMC opposition, abduction, MCP flexion, PIP flexion). In contrast, open-source Vision-Language-Action checkpoints in the `openpi` ecosystem (`pi0_libero` and `pi0_droid`) are pretrained with a 7D action space designed for 1-DoF parallel-jaw grippers:

* **Input Observation Space (`pi0_libero`):**
  * `observation/image`: Exterior RGB camera (`224 x 224 x 3`, `uint8`)
  * `observation/wrist_image`: Wrist or top-down RGB camera (`224 x 224 x 3`, `uint8`)
  * `observation/state`: 8-dimensional proprioceptive vector $[x, y, z, r_x, r_y, r_z, q_{\text{grip}}, -q_{\text{grip}}]$
  * `prompt`: Natural language task instruction
* **Output Action Space:**
  * Action chunk $A \in \mathbb{R}^{H \times 32}$, where the first 7 dimensions represent $[\Delta x, \Delta y, \Delta z, \Delta r_x, \Delta r_y, \Delta r_z, u_{\text{grip}}]$ over prediction horizon $H$.

Directly wiring $A_{t, 0:17}$ to the 17 ORCA joint position controllers violates actuator range scaling and produces rapid bang-bang oscillations that eject the manipulated object from the palm.

---

## 2. Observation Pipeline (`orca_vla/cameras.py`, `orca_vla/observations.py`)

### 2.1 Multi-View Offscreen Rendering
`VLACameraRenderer` wraps `mujoco.Renderer` (`224 x 224`) to provide virtual `MjvCamera` viewpoints for the 17-DoF `scene_right_cube_orientation.xml` environment:
* `palm_front`: Oblique frontal view focused on the palm and cube (`lookat=[0.165, -0.015, 0.18]`, `distance=0.28`, `azimuth=145°`, `elevation=-32°`).
* `palm_top`: Overhead view directly above the palm (`lookat=[0.17, -0.015, 0.185]`, `distance=0.22`, `azimuth=180°`, `elevation=-75°`).

### 2.2 Proprioceptive State Formatting
`OrcaVLAObservationBuilder` constructs the 8-dimensional state vector expected by `pi0_libero` by extracting the 3D position and axis-angle orientation of `task_cube` relative to the world frame alongside normalized hand closure state.

---

## 3. Action-Space Adaptation (`orca_vla/actions.py`)

### 3.1 Parametric 17-DoF Hand Synergy (`hand_synergy_pose`)
Maps a scalar closure ratio $c \in [0, 1]$ to all 17 ORCA actuators using calibrated opposition angles (`max_finger_curl = 0.65`, `thumb_cmc = 0.0`, `thumb_abd = 0.0`).

### 3.2 In-Hand Reorientation Synergy (`in_hand_synergy_delta`)
For the 17-DoF cube reorientation task wrapped with `RelativeJointTargetAction` (`max_delta_frac = 0.05`), `in_hand_synergy_delta` projects $\pi_0$'s 7D output $[\Delta x, \Delta y, \Delta z, \Delta r_x, \Delta r_y, \Delta r_z, u_{\text{grip}}]$ onto six anatomical in-hand manipulation synergies:
1. Wrist pitch stabilization (upward bias preventing the cube from sliding off the distal fingertips),
2. Uniform MCP flexion (forming a palmar containment wall),
3. Uniform PIP flexion (distal rolling contact),
4. Radial-ulnar finger wave (differential flexion and abduction across index, middle, ring, and pinky to roll the cube about the $Y$ axis),
5. Thumb opposition (CMC and abduction adjustment),
6. Thumb distal flexion (MCP and PIP opposition).

---

## 4. $\pi_0$-Integrated + Low-Level RL / Skill Architecture (`scripts/run_cube_reorient.py`)

To evaluate both standalone VLA behavior and closed-loop integration alongside high-frequency (`100 Hz`, $\Delta t = 0.01\text{ s}$) finger gaiting, `run_cube_reorient.py` (`mode="pi0_rl_hybrid"`) couples two layers:

1. **Closed-Loop $\pi_0$ Visual Inference & Simulator-State Phase Latch (`classify_high_level_reorient_skill`):**
   * Every 10 simulation steps, $\pi_0$ (`pi0_libero`) renders and processes the live `palm_front` and `palm_top` RGB observations (`224 x 224`) on the GPU (`~274–300 ms` per chunk).
   * Skill phase selection (`ROLL_X_POS`, `ROLL_X_NEG`, `ROLL_Y_POS`, `ROLL_Y_NEG`, `FLIP_180`) and the terminal stability latch (`HOLD_STILL` when $\theta \le 14.5^\circ$ and $\|\omega\| \le 0.95\text{ rad/s}$) are determined from simulator state (`info["red_face_world_normal"]` and `info["red_face_up_angle_rad"]`), while extracting a bounded residual vector from $\pi_0$'s chunk (which can be ablated to zero via `--residual-scale 0.0`).
   * Once `HOLD_STILL` is latched, relative joint deltas are zeroed so the hand maintains a stationary cradle for the required 25 consecutive control steps ($0.25\text{ s}$ at $100\text{ Hz}$ control).
2. **Low-Level 17-DoF Controller (`LowLevelReorientRLController`):**
   * Supports both **trained 10M-step RL policies** (`--low-level ppo_10m` or `sac_10m`, running `models/drac_{ppo,sac}_10M/best_model.zip` normalized via `vecnormalize.pkl` without any scripted trajectories) and **scripted skill primitives** (`--low-level scripted`, replaying `orca_vla/reorient_expert_trajs.npz` with `models/ppo_cube_reorient.zip` fallback).
   * Ablating $\pi_0$'s residual (`--residual-scale 0.0`) confirms that zero-shot $\pi_0$ does not drive low-level finger gaiting; physical reorientation is executed by the 100 Hz low-level RL policy or skill trajectory while $\pi_0$ operates in the visual inference loop.
