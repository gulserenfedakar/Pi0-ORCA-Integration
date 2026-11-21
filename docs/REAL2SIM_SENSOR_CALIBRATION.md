# Real2Sim Multi-Modal Sensor Calibration & Hardware-Aligned RL Fine-Tuning

**Author:** Gulseren Fedakar Gonul  
**Platform:** 17-DoF Tendon-Driven ETH ORCA Dexterous Hand (`OrcaHandRightCubeOrientation` in MuJoCo 3.x)  

---

## 1. Motivation: The Sensor & Kinematic Sim-to-Real Gap

Deep reinforcement learning policies (`PPO`, `SAC`) trained under idealized simulation states achieve high in-hand cube reorientation proficiency (`67.0%–79.0%` held success over 200 seeds on DRAC `Nibi`). However, transferring ideal-trained policies to a physical 17-DoF tendon-driven hand equipped with external RGB cameras and a palm-base ultrasonic proximity sensor exposes four concrete Sim-to-Real discrepancies:

1. **Dual-Camera AprilTag Pose & Occlusion Noise:** External overhead (`top`) and lateral (`side`) $1920 \times 1080$ cameras tracking $27\text{ mm}$ AprilTags (`36h11`) on a $35\text{ mm}$ cube exhibit a $5\text{--}10\text{ mm}$ inter-camera depth inconsistency, $2.6^\circ\text{--}3.0^\circ$ per-frame angular jitter, and visual self-occlusion dropouts (`~18%`) when flexing fingers hide the fiducial tags.
2. **Ultrasonic Acoustic Cone, Ringing Dead Zone, and Specular Tilt Dropouts:** Unlike MuJoCo's default single-ray optical `<rangefinder>`, a physical $40\text{ kHz}$ piezoelectric ultrasonic rangefinder (HC-SR04) radiates a $\sim 15^\circ$ half-angle cone, clamps or misses inside a $\le 22\text{ mm}$ piezoelectric ringing dead zone ($\{17, 20, 23\}\text{ mm}$ discrete steps), quantizes distances in $\sim 3.2\text{ mm}$ increments at $10\text{ Hz}$ ($100\text{ ms}$ period vs. $100\text{ Hz}$ RL control), and loses primary echoes when flat cube facets tilt past $\theta_{\max} \approx 25^\circ$—causing raw readings to jump to background wall reflections ($\sim 58.4\text{ cm}$ or $\sim 128.9\text{ cm}$).
3. **Curved-Finger Specular Deflection vs. Echo Ambiguity:** While flexing fingers entering the $15^\circ$ cone can preempt the cube echo, curved 3D-printed finger shells deflect approximately $78\%$ of oblique ultrasonic pulses away from the receiver ($p_{\text{survive}} \approx 0.22$).
4. **Locked-Wrist and Physical Thumb Range-of-Motion (ROM) Constraints:** Keeping the palm-base ultrasonic sensor aligned with the manipulation workspace requires locking the 1-DoF wrist pitch actuator ($\Delta q_{\text{wrist}} = 0$, restricting active control to the 16 finger degrees of freedom), while the physical `thumb_mcp` actuator operates within $[-5^\circ, +40^\circ]$ rather than the unconstrained $[-25^\circ, +100^\circ]$ simulation range.

---

## 2. Calibrated Real2Sim Sensor Parameters (`orca_vla/real2sim_sensors.py`)

System identification across 10 synchronized dual-camera + ultrasonic recordings ($4,243$ time-aligned samples including static palm benchmarks, $\pm 45^\circ$ incidence-angle sweeps, and time-logged physical policy replays) yields the parametric digital-twin specification implemented in [`orca_vla/real2sim_sensors.py`](../orca_vla/real2sim_sensors.py):

| Modality / Parameter | Config Field | Calibrated Value | Empirical Basis |
| :--- | :--- | :--- | :--- |
| **Camera Intrinsics (`top`)** | `fx, fy, cx, cy` | `1960.1, 1956.5, 981.5, 525.6 px` | ChArUco calibration at $1920 \times 1080$ ($0.675\text{ px}$ RMS; $0.42\text{ px}$ extrinsic reproj.) |
| **Camera Intrinsics (`side`)** | `fx, fy, cx, cy` | `1935.7, 1931.2, 945.9, 502.0 px` | ChArUco calibration at $1920 \times 1080$ ($0.587\text{ px}$ RMS; $0.53\text{ px}$ extrinsic reproj.) |
| **AprilTag Pose & Angular Noise** | `rot_step_jitter_deg` | `5–10 mm` bias, `2.6°` jitter, `18%` occlusion | Fused dual-camera PnP (`effective_tag_mm = 26.25 mm`); holds last pose on occlusion |
| **Ultrasonic Acoustic Cone** | `25-ray cluster` | `15.0°` half-angle (`1 axis + 3×8 rings`) | Triangulated at palm base `[0.1069, -0.0107, 0.2010] m`, pointing along `[0.965, -0.262, 0.0]` |
| **Range Offset & Quantization** | `range_offset_mm` | `+2.9 mm` offset, `3.2 mm` step @ `10 Hz` | Static bench MAE $= 3.3\text{ mm}$ (RMS $3.9\text{ mm}$, held-out $+3.9\text{ mm}$); 10-step sample-and-hold |
| **Ringing Dead-Zone Floor** | `dead_zone_cutoff_mm` | `≤ 22.0 mm` (`{17, 20, 23} mm` / `8% None`) | Verified across $217$ near-transducer samples; zero-masked to `(0.0, 0.0)` for RL |
| **Bistatic Specular Tilt Limit** | `logistic_theta50_*` | $\theta_{50} = 26.5^\circ$ ($7.4\text{ cm}$) $\to 18.5^\circ$ ($3.5\text{ cm}$), $k = 0.12$ | Raises sim-vs-real cube echo agreement from $46\%$ (naive cone) to $88\%\text{--}91\%$ |
| **Curved-Finger Deflection** | `hand_echo_survival_prob` | `22.0%` survival (`78.0%` deflected) | Eliminates phantom finger echoes during post-drop frames, reducing full-replay rate gap from $21\%$ to $<5\%$ |
| **Outlier Wall Jumps on Miss** | `prob_wall_jump_on_miss` | `67.0%` wall jump (`58.4 / 128.9 cm`), `33.0%` timeout | Filtered via 3-tap median + $150\text{ mm}$ workspace gate to `(0.0, 0.0)` to protect `VecNormalize` |

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

## 4. Training & Fine-Tuning with Hardware Constraints on DRAC `Nibi`

To train or warm-start fine-tune `PPO` and `SAC` policies with a locked wrist, physical thumb ROM bounds, and calibrated Real2Sim sensor noise:

```bash
# Warm-start fine-tune 10M-step PPO checkpoint under locked wrist + physical ROM + calibrated Real2Sim noise (51D compatible)
python scripts/train_rl_reorient.py \
    --algo ppo \
    --steps 5000000 \
    --n-envs 16 \
    --lock-wrist \
    --real-joint-roms \
    --real2sim-noise \
    --resume models/drac_ppo_10M/best_model.zip \
    --out runs/ppo_reorient_real2sim_5M

# Submit the 4-job Slurm comparison array (PPO & SAC, Ideal Locked-Wrist vs. Calibrated Real2Sim) on DRAC Nibi
sbatch --account=def-<pi_username> scripts/slurm_drac_rl_benchmark.sh
```
