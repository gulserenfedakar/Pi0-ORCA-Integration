# Real2Sim Multi-Modal Sensor Calibration & Hardware-Aligned RL Fine-Tuning

**Author:** Gulseren Fedakar Gonul  
**Platform:** 17-DoF Tendon-Driven ETH ORCA Dexterous Hand (`OrcaHandRightCubeOrientation` in MuJoCo 3.x)  

---

## 1. Motivation: The Sensor & Kinematic Sim-to-Real Gap

Deep reinforcement learning policies (`PPO`, `SAC`) trained under idealized simulation states achieve high in-hand cube reorientation proficiency (`67.0%–79.0%` held success over 200 seeds on DRAC `Nibi`). However, transferring ideal-trained policies to a physical 17-DoF tendon-driven hand equipped with external RGB cameras and a palm-base ultrasonic proximity sensor exposes four concrete Sim-to-Real discrepancies:

1. **Dual-Camera AprilTag Pose & Occlusion Noise:** External overhead (`top`) and lateral (`side`) $1920 \times 1080$ cameras tracking $27\text{ mm}$ AprilTags (`36h11`) on a $35\text{ mm}$ cube exhibit a $4.5\text{--}10\text{ mm}$ inter-camera depth inconsistency and $2.5^\circ\text{--}4.3^\circ$ static rotational bias, paired with $0.4\text{ mm}$ / $0.7^\circ$ per-frame jitter at $30\text{ Hz}$ and brief self-occlusion bursts (`~0.8%–4.0%`) when flexing fingers hide the fiducial tags.
2. **Ultrasonic Acoustic Cone, Ringing Dead Zone, and Specular Tilt Dropouts:** Unlike MuJoCo's default single-ray optical `<rangefinder>`, a physical $40\text{ kHz}$ piezoelectric ultrasonic rangefinder (HC-SR04) radiates a $\sim 15^\circ$ half-angle cone (modeled with 25 rays and per-ray specular filtering), clamps or misses inside a $\le 22\text{ mm}$ piezoelectric ringing dead zone ($\{17, 20\}\text{ mm}$ discrete steps), quantizes distances in $\sim 3.2\text{ mm}$ increments at $10\text{ Hz}$ ($100\text{ ms}$ period vs. $100\text{ Hz}$ RL control), and loses primary echoes when flat cube facets tilt past $\theta_{\max} \approx 25^\circ$—causing raw readings to jump to background wall reflections ($\sim 58.4\text{ cm}$ or $\sim 128.9\text{ cm}$).
3. **Curved-Finger Specular Deflection vs. Echo Ambiguity:** While flexing fingers entering the $15^\circ$ cone preempt the cube echo, curved 3D-printed finger shells deflect approximately $98.5\%$ of oblique ultrasonic pulses away from the receiver ($p_{\text{survive}} \approx 0.015$, measured at `6/446 = 1.35%` in physical twin replays).
4. **Locked-Wrist and Physical Thumb Range-of-Motion (ROM) Constraints:** Keeping the palm-base ultrasonic sensor aligned with the manipulation workspace requires locking the 1-DoF wrist pitch actuator ($\Delta q_{\text{wrist}} = 0$, restricting active control to the 16 finger degrees of freedom), while the physical `thumb_mcp` (`right_t-mcp_actuator`) operates within $[-5^\circ, +40^\circ]$ rather than the unconstrained $[-25^\circ, +100^\circ]$ simulation range.

---

## 2. Calibrated Real2Sim Sensor Parameters (`orca_vla/real2sim_sensors.py`)

System identification across 10 synchronized dual-camera + ultrasonic recordings ($4,243$ time-aligned samples including static palm benchmarks, $\pm 45^\circ$ incidence-angle sweeps, and time-logged physical policy replays) yields the parametric digital-twin specification implemented in [`orca_vla/real2sim_sensors.py`](../orca_vla/real2sim_sensors.py):

| Modality / Parameter | Config Field | Calibrated Value | Empirical Basis |
| :--- | :--- | :--- | :--- |
| **Camera Intrinsics (`top`)** | `fx, fy, cx, cy` | `1960.1, 1956.5, 981.5, 525.6 px` | ChArUco calibration at $1920 \times 1080$ ($0.675\text{ px}$ RMS; $0.42\text{ px}$ extrinsic reproj.) |
| **Camera Intrinsics (`side`)** | `fx, fy, cx, cy` | `1935.7, 1931.2, 945.9, 502.0 px` | ChArUco calibration at $1920 \times 1080$ ($0.587\text{ px}$ RMS; $0.53\text{ px}$ extrinsic reproj.) |
| **AprilTag Pose & Angular Noise** | `pos/rot_step_jitter_*` | `4.5 mm / 2.5°` bias, `0.4 mm / 0.7°` jitter, `0.8%–4.0%` occ. | Measured directly from consecutive static-palm frames (`0.21–0.42 mm/axis`) and `100%` in-hand visibility |
| **AprilTag Velocity Noise** | `vel_lin/ang_jitter_*` | `0.015 m/s` linear, `0.45 rad/s` angular | Matches finite-difference velocity noise of 30 Hz AprilTag tracking |
| **Ultrasonic Acoustic Cone** | `25-ray cluster` | `15.0°` half-angle (`1 axis + 3×8 rings`), per-ray filter | Triangulated at palm base `[0.1069, -0.0107, 0.2010] m`, pointing along `[0.965, -0.262, 0.0]` |
| **Range Offset & Quantization** | `range_offset_mm` | `+2.9 mm` offset, `3.2 mm` step @ `10 Hz` | Static bench MAE $= 3.3\text{ mm}$ (RMS $3.9\text{ mm}$, held-out $+3.9\text{ mm}$); 10-step sample-and-hold |
| **Ringing Dead-Zone Floor** | `dead_zone_cutoff_mm` | `≤ 22.0 mm` (`{17, 20} mm` / `8% None`) | Verified across $217$ near-transducer samples; zero-masked to `(0.0, 0.0)` for RL |
| **Bistatic Specular Tilt Limit** | `logistic_theta50_*` | $\theta_{50} = 26.5^\circ \to 18.5^\circ$, $k = 0.38/\text{deg}$ | Raises sim-vs-real cube echo agreement from $46\%$ (naive cone) to $88\%\text{--}91\%$ |
| **Curved-Finger Deflection** | `hand_echo_survival_prob` | `1.5%` survival (`98.5%` deflected) | Matches empirical `6/446 (1.35%)` in-range hand echo rate during physical replays |
| **Outlier Wall Jumps on Miss** | `prob_wall_jump_on_miss` | `90.0%` wall jump (`58.4 / 128.9 cm`), `10.0%` timeout | Filtered via 3-tap median + $150\text{ mm}$ workspace gate to `(0.0, 0.0)` to protect `VecNormalize` |

---

## 3. Real-vs-Sim Quantitative Validation Summary

![Ultrasonic Real2Sim Validation Plots](../assets/ultrasonic_real2sim_validation_plots.svg)

| Benchmark Dataset | Evaluation Window | Real Valid Rate ($\le 150\text{ mm}$) | Raw Monostatic Cone Rate | Calibrated Real2Sim Rate (Gap) | Sim $-$ Real Inlier Bias (Median / Mean / MAE) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Static Bench (Pos 2–5 + Pos 6)** | 5 cal + 1 held-out pos | `100.0%` | `100.0%` | **`100.0%` (`0.0 pp`)** | Mean `-1.3 mm` / MAE **`3.3 mm`** (Held-out **`+3.9 mm`**) |
| **Stationary Palm Sweep (`static01`)** | $n = 269$ tracked ($N = 291$) | `98.1%` | `95.9%` | **`95.9%` (`2.2 pp`)** | Median **`+1.20 mm`** / Mean **`+0.53 mm`** / MAE `5.27 mm` |
| **$\pm 45^\circ$ Incidence Sweep (`angle02`)** | $n = 535$ tracked ($N = 567$) | `51.6%` | `98.9%` *(no tilt limit)* | **`53.1%` (`1.5 pp`)** | Median **`-0.10 mm`** / Mean `-2.17 mm` / MAE `7.31 mm` |
| **Short-Range Replay (`run01`, $4\text{ cm}$)** | $n = 223$ tracked ($N = 361$) | `35.9%` | `100.0%` *(no tilt limit)* | **`41.7%` (`5.8 pp`)** | Median `+9.85 mm` / Mean `+7.38 mm` / MAE `9.78 mm` |
| **Full-Twin Replay (`run07`, Wrist Locked)** | $n = 339$ replay ($N = 459$) | `14.2%` | `36.6%` *(`22.4 pp` gap)* | **`18.6%` (`4.5 pp`)** | Median `+12.90 mm` / Mean `+10.25 mm` / MAE `11.31 mm` |
| **Full-Twin Replay (`run08`, $124^\circ$ Turn)** | $n = 333$ replay ($N = 553$) | `15.3%` | `35.7%` *(`20.4 pp` gap)* | **`18.9%` (`3.6 pp`)** | Median `+6.40 mm` / Mean **`+2.55 mm`** / MAE **`7.05 mm`** |

---

## 4. Training & Evaluation on DRAC `Nibi`

```bash
# Submit the Slurm Real2Sim & locked-wrist ablation array on DRAC Nibi
sbatch --account=def-<pi_username> scripts/slurm_drac_rl_benchmark.sh

# Evaluate all trained checkpoints over 200 fixed seeds (20000-20199)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -u scripts/evaluate_rl_reorient.py --runs-dir runs --episodes 200 --out-csv results/rl_real2sim_ablation_benchmark.csv
```

---

## 5. DRAC `Nibi` Ablation Results (200 Fixed Seeds `20000–20199`)

| Run Configuration | Steps | Obs | Wrist Locked + Real ROMs | Real2Sim Noise | Goal Align ($<15^\circ$) | Held Success (%) | Latched Success (%) | Cube Drop (%) | Final Angle (All / Aligned) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ppo_ideal_locked_wrist_68d`** | 5M | 68D | ✅ | ❌ | **96.0%** | **92.5%** | **92.5%** | **3.5%** | **14.76°** *(14.21°)* |
| **`ppo_real2sim_camera_68d`** | 10M | 68D | ✅ | ✅ (Cam) | **83.5%** | **62.0%** | **65.0%** | 9.0% | 31.72° *(20.23°)* |
| **`ppo_real2sim_camera_ultrasonic_70d`** | 10M | 70D | ✅ | ✅ (Cam+US) | 63.0% | **55.5%** | **55.5%** | 9.0% | 32.26° *(**11.54°**)* |
| `ppo_reorient_random_10M` *(Unlocked)* | 10M | 68D | ❌ | ❌ | 21.0% | 0.0% | 0.0% | 16.0% | 79.85° *(90.23°)* |
| `sac_reorient_random_10M` *(Unlocked)* | 10M | 68D | ❌ | ❌ | 72.5% | 65.0% | 69.5% | 6.5% | 30.46° *(9.53°)* |

**Key Takeaways:**
1. **Locked Wrist + Physical Thumb ROMs Unlock 92.5% PPO Proficiency:** Restricting `right_wrist` to `0.0 rad` and clamping `right_t-mcp_actuator` to `[-5°, +40°]` during training eliminates wrist-flicking instability and pushes PPO from `0.0%` (`16.0%` drop) to **`92.5%` held success (`3.5%` drop)** in 5M steps.
2. **Real2Sim Sensor Robustness:** Under calibrated 30 Hz dual-camera noise, PPO reaches **`62.0%` pure / `65.0%` latched held success (`83.5%` goal alignment)** at 10M steps, while adding 10 Hz 25-ray ultrasonic observations achieves **`55.5%` held success** with the sharpest aligned final orientation (**`11.54°`** vs. `20.23°`).
