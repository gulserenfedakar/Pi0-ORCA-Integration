"""Generate 4 widescreen (1920x1080) JPEG presentation slides summarizing the
17-DoF ORCA Hand In-Hand Cube Reorientation architecture, Pi0 vs. Hybrid VLA+RL
benchmarks, DRAC Nibi 10M-step RL (PPO, SAC, TD3) learning curves, and rollout
keyframe trajectory analysis.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
for _cand in sorted(REPO_ROOT.parent.glob("*Dexterous-Hand-Simulation*")):
    sys.path.insert(0, str((_cand / "src").resolve()))
sys.path.insert(0, str(REPO_ROOT))

OUT_DIRS = [
    REPO_ROOT / "assets" / "slides",
]

# Clean engineering color palette
MAROON = "#7A003C"
GOLD = "#FDBF57"
NAVY = "#1E293B"
SLATE = "#475569"
LIGHT_BG = "#F8FAFC"
CARD_BG = "#FFFFFF"
BORDER = "#CBD5E1"
BLUE = "#2563EB"
TEAL = "#0D9488"
GREEN = "#16A34A"
ORANGE = "#EA580C"
RED = "#DC2626"
PURPLE = "#7C3AED"


def add_header(fig, title: str, subtitle: str, slide_num: int):
    header_ax = fig.add_axes([0.0, 0.90, 1.0, 0.10])
    header_ax.set_facecolor(MAROON)
    header_ax.set_xlim(0, 1)
    header_ax.set_ylim(0, 1)
    header_ax.axis("off")
    header_ax.add_patch(patches.Rectangle((0, 0), 1, 1, color=MAROON, transform=header_ax.transAxes))
    header_ax.add_patch(patches.Rectangle((0, 0), 1, 0.06, color=GOLD, transform=header_ax.transAxes))
    header_ax.text(
        0.025, 0.64, title,
        color="white", fontsize=17.0, fontweight="bold", va="center", ha="left",
        family="DejaVu Sans",
    )
    header_ax.text(
        0.025, 0.23, subtitle,
        color="#FDE68A", fontsize=11.5, va="center", ha="left",
        family="DejaVu Sans",
    )
    header_ax.text(
        0.975, 0.64, f"π₀ + 17-DoF ORCA Hand  |  Slide {slide_num}/4",
        color="white", fontsize=10.5, fontweight="bold", va="center", ha="right",
        family="DejaVu Sans",
    )


def save_slide(fig, filename: str):
    out_primary = OUT_DIRS[0] / filename
    out_primary.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_primary), format="jpg", dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    # Ensure exact 1920x1080 JPEG with high quality
    img = Image.open(out_primary).convert("RGB").resize((1920, 1080), Image.Resampling.LANCZOS)
    img.save(out_primary, "JPEG", quality=95)
    for d in OUT_DIRS[1:]:
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_primary, d / filename)
    print(f"Saved {filename} -> {out_primary}")


def make_slide_1():
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=LIGHT_BG)
    add_header(
        fig,
        "1. System Architecture: Bridging Pretrained VLA (π₀) to 17-DoF ORCA Hand",
        "Resolving the 7D-to-17D Embodiment Mismatch and 3.5 Hz vs. 20 Hz Contact Control Bottleneck in MuJoCo",
        1,
    )

    # Left column: 3 camera viewpoints
    cam_ax = fig.add_axes([0.03, 0.06, 0.31, 0.80])
    cam_ax.set_facecolor(CARD_BG)
    cam_ax.set_xlim(0, 1)
    cam_ax.set_ylim(0, 1)
    for spine in cam_ax.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(1.5)
    cam_ax.set_xticks([])
    cam_ax.set_yticks([])
    cam_ax.text(
        0.5, 0.96, "Multi-View Palmar Perception (224×224 RGB)",
        fontsize=13, fontweight="bold", color=NAVY, ha="center", va="top",
    )

    cam_specs = [
        ("cube_reorient_default_cam.png", "Default Free View (default_free)", 0.60),
        ("cube_reorient_front_cam.png", "Oblique Palmar View (palm_front)", 0.30),
        ("cube_reorient_top_cam.png", "Overhead Palmar View (palm_top)", 0.00),
    ]
    for fname, label, y0 in cam_specs:
        img_path = REPO_ROOT / "assets" / fname
        if img_path.exists():
            sub = fig.add_axes([0.05, 0.06 + y0 * 0.80 + 0.02, 0.27, 0.19])
            sub.imshow(Image.open(img_path))
            sub.set_xticks([])
            sub.set_yticks([])
            for sp in sub.spines.values():
                sp.set_edgecolor(SLATE)
            sub.set_title(label, fontsize=9.8, fontweight="bold", color=SLATE, pad=3)

    # Right column: Architecture diagram + comparison of 3 control pathways
    diag_ax = fig.add_axes([0.36, 0.06, 0.61, 0.80])
    diag_ax.set_facecolor(CARD_BG)
    diag_ax.set_xlim(0, 100)
    diag_ax.set_ylim(0, 100)
    for spine in diag_ax.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(1.5)
    diag_ax.set_xticks([])
    diag_ax.set_yticks([])

    diag_ax.text(
        50, 96, "Three Control Paradigms Evaluated on 17-DoF In-Hand Cube Reorientation",
        fontsize=14, fontweight="bold", color=NAVY, ha="center", va="top",
    )

    def draw_box(ax, x, y, w, h, title, body, facecolor, edgecolor, title_color="white"):
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.5",
            facecolor=facecolor, edgecolor=edgecolor, linewidth=2.0,
        )
        ax.add_patch(rect)
        hdr = patches.FancyBboxPatch(
            (x, y + h - 5.5), w, 5.5, boxstyle="round,pad=0.3,rounding_size=1.2",
            facecolor=edgecolor, edgecolor=edgecolor, linewidth=1.0,
        )
        ax.add_patch(hdr)
        ax.text(x + w / 2, y + h - 2.8, title, fontsize=9.2, fontweight="bold", color=title_color, ha="center", va="center")
        ax.text(x + w / 2, y + (h - 5.5) / 2, body, fontsize=9.1, color=NAVY, ha="center", va="center", linespacing=1.35)

    # Pathway A: Standalone VLA Direct
    draw_box(
        diag_ax, 3, 66, 27, 22,
        "Pathway A: Naive Standalone VLA",
        "π₀ (3B Flow-Matching)\n~289 ms/chunk (~3.5 Hz)\n7D Parallel-Gripper Output\n(Mismatch for 17 Tendons)",
        "#FEF2F2", RED,
    )
    diag_ax.annotate("", xy=(36, 77), xytext=(30.5, 77), arrowprops=dict(arrowstyle="->", lw=2.5, color=RED))
    draw_box(
        diag_ax, 36.5, 66, 27, 22,
        "Direct 17-DoF Mapping",
        "Raw joint targets applied\nwithout relative clamping\nor finger coordination",
        "#FEF2F2", RED,
    )
    diag_ax.annotate("", xy=(69.5, 77), xytext=(64, 77), arrowprops=dict(arrowstyle="->", lw=2.5, color=RED))
    draw_box(
        diag_ax, 70, 66, 27, 22,
        "Outcome: Immediate Drop",
        "Wrist tilts down on step 1\n100% Cube Drop Rate\n0.0% Task Success",
        "#FEE2E2", RED,
    )

    # Pathway B: Standalone VLA + Synergy + Relative Clamping
    draw_box(
        diag_ax, 3, 37, 27, 22,
        "Pathway B: Synergy-Adapted VLA",
        "π₀ (pi0_libero, bfloat16)\nDual Palmar RGB + 8D State\n6.34 GB / 8 GB VRAM",
        "#EFF6FF", BLUE,
    )
    diag_ax.annotate("", xy=(36, 48), xytext=(30.5, 48), arrowprops=dict(arrowstyle="->", lw=2.5, color=BLUE))
    draw_box(
        diag_ax, 36.5, 37, 27, 22,
        "17-DoF In-Hand Synergy",
        "in_hand_synergy_delta\n+ RelativeJointTargetAction\n(max_delta_frac = 0.05)",
        "#EFF6FF", BLUE,
    )
    diag_ax.annotate("", xy=(69.5, 48), xytext=(64, 48), arrowprops=dict(arrowstyle="->", lw=2.5, color=BLUE))
    draw_box(
        diag_ax, 70, 37, 27, 22,
        "Outcome: Stable Grasp, No Gait",
        "0.0% Cube Drop Rate\nRotates cube ~17° (90°→73°)\n0.0% Full 180° Hold Success",
        "#DBEAFE", BLUE,
    )

    # Pathway C: Hierarchical Hybrid VLA + RL (PPO / SAC)
    draw_box(
        diag_ax, 3, 8, 27, 23,
        "Pathway C: Hybrid VLA + RL",
        "High-Level π₀ + Phase Gate (~3.5 Hz)\nVisual Chunk + Bounded Residual:\n[FLIP_180 | ROLL_X | ROLL_Y | HOLD]",
        "#ECFDF5", GREEN,
    )
    diag_ax.annotate("", xy=(36, 19.5), xytext=(30.5, 19.5), arrowprops=dict(arrowstyle="->", lw=2.5, color=GREEN))
    draw_box(
        diag_ax, 36.5, 8, 27, 23,
        "Low-Level 20 Hz Skill / RL",
        "Reference Gait + PPO/SAC Policy\n(~0.3 ms CPU Step Execution)\nRelative 17-DoF Finger Gaiting",
        "#ECFDF5", GREEN,
    )
    diag_ax.annotate("", xy=(69.5, 19.5), xytext=(64, 19.5), arrowprops=dict(arrowstyle="->", lw=2.5, color=GREEN))
    draw_box(
        diag_ax, 70, 8, 27, 23,
        "Outcome: Reliable Reorientation",
        "100% Nominal / 80% Random\nHold: 25/25 (Nom) | 21.0/25 (SAC)\nAligned Error: 6.6°–13.8° (<15°)",
        "#DCFCE7", GREEN,
    )
    # Closed-loop feedback annotation at bottom of Pathway C
    diag_ax.annotate(
        "Closed-Loop Feedback: 20 Hz Proprioception (43D) + 3.5 Hz Dual-Camera RGB Re-Planning",
        xy=(16.5, 7.5), xytext=(50, 2.2),
        ha="center", va="center", fontsize=8.8, fontweight="bold", color=GREEN,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#ECFDF5", edgecolor=GREEN, linestyle="--"),
    )

    save_slide(fig, "slide1_architecture_pipeline.jpg")


def make_slide_2():
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=LIGHT_BG)
    add_header(
        fig,
        "2. Pretrained VLA (π₀) vs. Hybrid π₀ + RL Benchmark Results",
        "Evaluated on 17-DoF ORCA In-Hand Cube Reorientation (RTX 4070 Laptop GPU, 8 GB VRAM, bfloat16)",
        2,
    )

    # Top-left: Bar chart comparing Task Success, Goal Alignment, and Drop Rate
    ax1 = fig.add_axes([0.05, 0.45, 0.52, 0.39])
    ax1.set_facecolor(CARD_BG)
    labels = [
        "1. Direct π₀\n(No Synergy)",
        "2. Synergy π₀\n(palm_front)",
        "3. Synergy π₀\n(palm_top)",
        "4. Hybrid π₀+RL\n(Nominal 180°)",
        "5. Hybrid π₀+RL\n(Random 5 Poses)",
    ]
    x = np.arange(len(labels))
    w = 0.25
    align = [0.0, 0.0, 0.0, 100.0, 100.0]
    success = [0.0, 0.0, 0.0, 100.0, 80.0]
    drop = [100.0, 0.0, 0.0, 0.0, 20.0]

    b1 = ax1.bar(x - w, align, width=w, label="Goal Alignment (<15°)", color=BLUE)
    b2 = ax1.bar(x, success, width=w, label="Held Task Success (25/25 steps)", color=GREEN)
    b3 = ax1.bar(x + w, drop, width=w, label="Cube Drop Rate (%)", color=RED)

    ax1.set_ylim(0, 118)
    ax1.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold", color=NAVY)
    ax1.set_title("Manipulation Reliability Across Adaptation Levels", fontsize=13, fontweight="bold", color=NAVY, pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9.5, fontweight="bold", color=NAVY)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left", fontsize=9.5, frameon=True)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2, h + 2,
                f"{h:.0f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=NAVY,
            )
    ax1.text(
        1.5, 34,
        "Configs 2 & 3 (Synergy π₀):\n0% Drop Rate & 17° Partial Rotation (90°→72.9°),\nbut 3.5 Hz VLA alone cannot finger-gait to <15°",
        ha="center", va="center", fontsize=8.6, fontweight="bold", color=BLUE,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#EFF6FF", edgecolor=BLUE),
    )

    # Top-right: VLA Inference Latency Comparison (OpenVLA-7B vs Pi0 vs RL CPU)
    ax2 = fig.add_axes([0.62, 0.45, 0.34, 0.39])
    ax2.set_facecolor(CARD_BG)
    models = ["OpenVLA-7B\n(Autoregressive)", "π₀ (pi0_libero)\n(Flow-Matching)", "Low-Level RL\n(PPO / SAC CPU)"]
    latencies = [2144.69, 289.16, 0.33]
    colors = [ORANGE, BLUE, GREEN]
    bars = ax2.bar(models, latencies, color=colors, width=0.5)
    ax2.set_yscale("log")
    ax2.set_ylim(0.1, 12000)
    ax2.set_ylabel("Inference Latency per Call (ms, log scale)", fontsize=10.5, fontweight="bold", color=NAVY)
    ax2.set_title("Control Latency: 7.4× Faster VLA + 20 Hz RL", fontsize=12.5, fontweight="bold", color=NAVY, pad=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, val in zip(bars, latencies):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, val * 1.28,
            f"{val:.2f} ms" if val < 1 else f"{val:.1f} ms",
            ha="center", va="bottom", fontsize=10, fontweight="bold", color=NAVY,
        )
    ax2.text(
        0.54, 0.87, "π₀ is 7.4× faster than OpenVLA-7B\nand fits in 6.34 GB / 8 GB VRAM",
        transform=ax2.transAxes, ha="center", va="center",
        fontsize=9.2, fontweight="bold", color=MAROON,
        bbox=dict(boxstyle="round,pad=0.38", facecolor="#FEF3C7", edgecolor=GOLD),
    )

    # Bottom: Full quantitative table
    tab_ax = fig.add_axes([0.04, 0.06, 0.92, 0.32])
    tab_ax.axis("off")
    col_labels = [
        "Benchmark Configuration", "Mode", "Camera", "Infer (ms)", "E2E (ms)",
        "Goal Align", "Hold Streak", "Held Success", "Drop Rate", "Min Angle", "Final Angle",
    ]
    table_data = [
        ["1. Standalone π₀ — Naive Direct Mapping", "direct_17dof", "palm_front", "302.50", "459.18", "0.0%", "0.0 / 25", "0.0%", "100.0%", "30.09°*", "30.09° (Drop)"],
        ["2. Standalone π₀ — Relative + 17-DoF Synergy", "synergy_reorient", "palm_front", "287.13", "443.12", "0.0%", "0.0 / 25", "0.0%", "0.0%", "72.93°", "90.20°"],
        ["3. Standalone π₀ — Relative + 17-DoF Synergy", "synergy_reorient", "palm_top", "289.35", "446.57", "0.0%", "0.0 / 25", "0.0%", "0.0%", "72.96°", "90.41°"],
        ["4. Hybrid π₀ + RL (PPO) — Nominal 180° Flip", "pi0_rl_hybrid", "palm_front", "289.16", "444.09", "100.0%", "25.0 / 25", "100.0%", "0.0%", "2.83°", "13.81°"],
        ["5. Hybrid π₀ + RL (PPO) — Randomized 5 Poses", "pi0_rl_hybrid", "palm_front", "295.62", "450.47", "100.0%", "20.6 / 25", "80.0%", "20.0%", "7.26°", "18.98°"],
    ]
    col_widths = [0.25, 0.11, 0.08, 0.07, 0.07, 0.07, 0.08, 0.08, 0.06, 0.06, 0.07]
    tbl = tab_ax.table(
        cellText=table_data, colLabels=col_labels, colWidths=col_widths,
        loc="upper center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.2)
    tbl.scale(1.0, 1.75)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(BORDER)
        if row == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color="white", fontweight="bold")
        elif row in (4, 5):
            cell.set_facecolor("#ECFDF5")
            if col in (0, 7):
                cell.set_text_props(fontweight="bold", color=GREEN)
        else:
            cell.set_facecolor(CARD_BG)
            if col == 0:
                cell.set_text_props(fontweight="bold", color=NAVY)

    tab_ax.text(
        0.5, 0.02,
        "*Note on Config 1 (30.09° Drop): Uncoordinated joint targets eject the cube at step 25 (100% drop rate), causing uncontrolled free-fall tumbling. Configs 2–3 maintain a 0% drop rate.",
        ha="center", va="center", fontsize=8.8, style="italic", color=SLATE,
    )

    save_slide(fig, "slide2_pi0_standalone_vs_hybrid.jpg")


def make_slide_3():
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=LIGHT_BG)
    add_header(
        fig,
        "3. DRAC Nibi Cluster: 10M-Step Matched RL Benchmark (PPO vs. SAC vs. TD3)",
        "Resolving the 200K-Step Short-Horizon Bottleneck via 16-Core Parallel MuJoCo Training under RelativeJointTargetAction",
        3,
    )

    # Load real training progress CSVs from DRAC Nibi
    ax_curve = fig.add_axes([0.05, 0.44, 0.45, 0.40])
    ax_curve.set_facecolor(CARD_BG)

    algo_meta = [
        ("sac", "SAC (10M Steps — 80% Test Success)", GREEN),
        ("ppo", "PPO (10M Steps — 69.2 min, 2,409 FPS)", BLUE),
        ("td3", "TD3 (10M Steps — 55% Test Success)", ORANGE),
    ]
    for algo, label, color in algo_meta:
        csv_path = REPO_ROOT / "models" / f"drac_{algo}_10M" / "progress.csv"
        df = pd.read_csv(csv_path)
        sub = df[["time/total_timesteps", "rollout/success_rate"]].dropna()
        steps_m = sub["time/total_timesteps"].values / 1e6
        sr = pd.Series(sub["rollout/success_rate"].values * 100.0).rolling(window=max(1, len(sub) // 40), min_periods=1).mean()
        ax_curve.plot(steps_m, sr, label=label, color=color, linewidth=2.4)

    # Mark the 200K-step short-horizon cutoff
    ax_curve.axvline(0.2, color=RED, linestyle="--", linewidth=1.8)
    ax_curve.text(
        0.35, 72, "Short-Horizon Cutoff (200K Steps = 0.2M)\nWhy SAC appears stuck at 16% early on!",
        color=RED, fontsize=9.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF2F2", edgecolor=RED),
    )

    ax_curve.set_xlim(0, 10.0)
    ax_curve.set_ylim(0, 88)
    ax_curve.set_xlabel("Environment Training Steps (Millions)", fontsize=11, fontweight="bold", color=NAVY)
    ax_curve.set_ylabel("Rollout Held Success Rate (%)", fontsize=11, fontweight="bold", color=NAVY)
    ax_curve.set_title("DRAC Nibi 10M-Step Training Curves (init=random)", fontsize=12.5, fontweight="bold", color=NAVY, pad=10)
    ax_curve.grid(True, linestyle="--", alpha=0.4)
    ax_curve.legend(loc="lower right", fontsize=9.5, frameon=True)

    # Top-right: 20-Episode Unseen Evaluation & Peak Eval Bar Chart
    ax_eval = fig.add_axes([0.55, 0.44, 0.41, 0.40])
    ax_eval.set_facecolor(CARD_BG)
    algos = ["SAC (10M)", "PPO (10M)", "TD3 (10M)"]
    x = np.arange(len(algos))
    w = 0.25
    goal_align = [85.0, 85.0, 60.0]
    eval_best = [90.0, 80.0, 70.0]
    held_succ = [80.0, 50.0, 55.0]

    b1 = ax_eval.bar(x - w, goal_align, width=w, label="Goal Align (<15°, 20 Seeds)", color=BLUE)
    b2 = ax_eval.bar(x, eval_best, width=w, label="Best EvalCallback Success (%)", color=TEAL)
    b3 = ax_eval.bar(x + w, held_succ, width=w, label="Strict 25-Step Hold (20 Seeds)", color=GREEN)

    ax_eval.set_ylim(0, 108)
    ax_eval.set_xticks(x)
    ax_eval.set_xticklabels(algos, fontsize=11, fontweight="bold", color=NAVY)
    ax_eval.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold", color=NAVY)
    ax_eval.set_title("Alignment (<15°) vs. Sustained 25-Step Hold", fontsize=12.5, fontweight="bold", color=NAVY, pad=10)
    ax_eval.grid(axis="y", linestyle="--", alpha=0.4)
    ax_eval.legend(loc="upper right", fontsize=8.8, frameon=True)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax_eval.text(
                bar.get_x() + bar.get_width() / 2, h + 1.8,
                f"{h:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color=NAVY,
            )

    # Bottom: Comparison Table + Key Findings
    tab_ax = fig.add_axes([0.03, 0.05, 0.94, 0.32])
    tab_ax.axis("off")
    col_labels = [
        "RL Algorithm (DRAC Nibi)", "Total Steps", "Wall-Clock Time", "Throughput",
        "Goal Align (<15°)", "EvalCallback (Best/Peak)", "20-Seed Hold Streak", "20-Seed Held Success", "Drop Rate", "Final Angle (All / Aligned)",
    ]
    rows = [
        ["SAC (Max-Entropy Off-Policy)", "10,000,000", "352.4 min (5.87 h)", "473 FPS", "85.0%", "90.0% / 90.0%", "21.0 / 25 steps", "80.0%", "0.0%", "18.37° (6.6° aligned)"],
        ["PPO (Clipped On-Policy)", "10,002,432", "69.2 min (1.15 h)", "2,409 FPS", "85.0%", "80.0% / 95.0%", "16.6 / 25 steps", "50.0% (75% Final Eval)", "0.0%", "19.04° (12.0° aligned)"],
        ["TD3 (Deterministic Off-Policy)", "10,000,000", "316.8 min (5.28 h)", "527 FPS", "60.0%", "70.0% / 70.0%", "13.8 / 25 steps", "55.0%", "5.0%", "46.36° (11.7° aligned)"],
    ]
    col_widths = [0.18, 0.08, 0.11, 0.07, 0.09, 0.12, 0.10, 0.11, 0.05, 0.12]
    tbl = tab_ax.table(cellText=rows, colLabels=col_labels, colWidths=col_widths, loc="upper center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.8)
    tbl.scale(1.0, 1.95)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(BORDER)
        if row == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color="white", fontweight="bold")
        elif row == 1:
            cell.set_facecolor("#ECFDF5")
            if col in (0, 5, 7, 9):
                cell.set_text_props(fontweight="bold", color=GREEN)
        else:
            cell.set_facecolor(CARD_BG)
            if col == 0:
                cell.set_text_props(fontweight="bold", color=NAVY)

    tab_ax.text(
        0.5, 0.07,
        "Key Finding: PPO & SAC both reach <15° in 85% of episodes (0% drops; aligned final error: 6.6° SAC, 12.0° PPO; 20-seed mean: 18.4°–19.0°).\n"
        "SAC's entropy regularization damps angular velocity (||ω||≤1 rad/s) to hold 25/25 steps (1.25 s) in 80% of seeds, while PPO trains 5.1× faster (69.2 min).",
        ha="center", va="center", fontsize=9.1, fontweight="bold", color=MAROON,
        bbox=dict(boxstyle="round,pad=0.38", facecolor="#FEF3C7", edgecolor=GOLD),
    )

    save_slide(fig, "slide3_drac_rl_10m_benchmark.jpg")


def make_slide_4():
    # Render 6 keyframes + angle trajectory from the 10M-step SAC policy
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from orca_sim import OrcaHandRightCubeOrientation
    from orca_sim.wrappers import RelativeJointTargetAction
    from orca_vla.cameras import VLACameraRenderer

    env = OrcaHandRightCubeOrientation(version="v2", initial_red_face="down")
    env = RelativeJointTargetAction(env, max_delta_frac=0.05)
    env.reset(seed=7)
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize.load(str(REPO_ROOT / "models/drac_sac_10M/vecnormalize.pkl"), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    model = SAC.load(str(REPO_ROOT / "models/drac_sac_10M/best_model.zip"), env=vec_env)

    import mujoco
    renderer = VLACameraRenderer(env.unwrapped.model, env.unwrapped.data, width=360, height=270)
    high_cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(env.unwrapped.model, high_cam)
    high_cam.lookat[:] = [0.165, -0.015, 0.185]
    high_cam.distance = 0.30
    high_cam.azimuth = 140.0
    high_cam.elevation = -55.0

    obs = vec_env.reset()
    frames = []
    angles_deg = []
    streaks = []

    for step in range(120):
        renderer.renderer.update_scene(env.unwrapped.data, camera=high_cam)
        frame = renderer.renderer.render().copy()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        info_d = info[0]
        if "red_face_up_angle_rad" in info_d:
            ang = float(np.degrees(info_d["red_face_up_angle_rad"]))
        else:
            cos_val = float(np.clip(info_d.get("red_face_up_alignment", -1.0), -1.0, 1.0))
            ang = float(np.degrees(np.arccos(cos_val)))
        st_val = int(info_d.get("success_streak", 0))
        frames.append((step, frame, ang, st_val))
        angles_deg.append(ang)
        streaks.append(st_val)
        if info_d.get("is_success", False) or st_val >= 25 or done[0]:
            break

    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=LIGHT_BG)
    add_header(
        fig,
        "4. Closed-Loop 17-DoF Cube Reorientation Rollout & Final Takeaways",
        "180° In-Hand Flip (Red Face Down → World-Up <15° & ||ω||≤1 rad/s) by 10M-Step SAC Policy",
        4,
    )

    # Top row: 6 action-focused keyframes capturing the rotation phases + final 25/25 hold
    cand_idxs = [0, 9, 17, 25, 35, len(frames) - 1]
    idxs = [min(max(0, i), len(frames) - 1) for i in cand_idxs]
    for k, idx in enumerate(idxs):
        step_num, img_arr, ang_val, st_val = frames[idx]
        ax_f = fig.add_axes([0.03 + k * 0.158, 0.56, 0.148, 0.28])
        ax_f.imshow(img_arr)
        ax_f.set_xticks([])
        ax_f.set_yticks([])
        border_col = GREEN if ang_val < 15.0 else (BLUE if ang_val < 90.0 else MAROON)
        for sp in ax_f.spines.values():
            sp.set_edgecolor(border_col)
            sp.set_linewidth(2.5)
        ax_f.set_title(
            f"Step {step_num+1} (t={(step_num+1)*0.05:.2f}s)\nθ = {ang_val:.1f}° | Hold: {st_val}/25",
            fontsize=9.2, fontweight="bold", color=NAVY, pad=5,
        )

    # Bottom-left: Angle error & hold streak twin-axis trajectory plot
    ax_traj = fig.add_axes([0.05, 0.08, 0.45, 0.40])
    ax_traj.set_facecolor(CARD_BG)
    t_axis = np.arange(1, len(angles_deg) + 1) * 0.05
    l1 = ax_traj.plot(t_axis, angles_deg, color=BLUE, linewidth=2.6, label="Red-Face Angle Error θ(t) [deg]")
    l2 = ax_traj.axhline(15.0, color=GREEN, linestyle="--", linewidth=2.0, label="15° Goal Cone Threshold")
    ax_traj.fill_between(t_axis, 0, 15.0, color="#DCFCE7", alpha=0.6)
    ax_traj.set_ylim(0, 190)
    ax_traj.set_xlabel("Simulation Time (s) [20 Hz Control]", fontsize=10.8, fontweight="bold", color=NAVY)
    ax_traj.set_ylabel("Angle Error to World-Up (°)", fontsize=10.8, fontweight="bold", color=BLUE)
    ax_traj.set_title("Orientation Error θ(t) & Stationary Hold Streak (||ω||≤1 rad/s)", fontsize=11.8, fontweight="bold", color=NAVY, pad=8)
    ax_traj.grid(True, linestyle="--", alpha=0.4)

    ax_st = ax_traj.twinx()
    l3 = ax_st.plot(t_axis, streaks, color=MAROON, linestyle="-.", linewidth=2.2, label="Stationary Hold Streak (/25 steps)")
    ax_st.set_ylim(0, 28)
    ax_st.set_ylabel("Hold Streak (steps)", fontsize=10.5, fontweight="bold", color=MAROON)

    lines = l1 + [l2] + l3
    labels_leg = [l.get_label() for l in lines]
    ax_traj.legend(lines, labels_leg, loc="upper center", bbox_to_anchor=(0.56, 0.97), fontsize=8.4, frameon=True)

    # Bottom-right: Summary card for Final Report integration
    card = fig.add_axes([0.55, 0.08, 0.41, 0.40])
    card.set_facecolor(CARD_BG)
    card.set_xlim(0, 100)
    card.set_ylim(0, 100)
    for sp in card.spines.values():
        sp.set_edgecolor(MAROON)
        sp.set_linewidth(2.0)
    card.set_xticks([])
    card.set_yticks([])

    card.add_patch(patches.Rectangle((0, 86), 100, 14, color=NAVY))
    card.text(50, 93, "Key Engineering & Experimental Contributions", fontsize=12.5, fontweight="bold", color="white", ha="center", va="center")

    bullets = [
        ("1. Diagnosed Pretrained VLA Bottlenecks on 17-DoF Hand:",
         "   • π₀ cuts inference latency 7.4× vs. OpenVLA-7B (289 ms vs. 2,145 ms)\n"
         "   • RelativeJointTargetAction + synergy prevents drops (100% → 0%),\n"
         "     proving why low-level 20 Hz finger gaiting is still required."),
        ("2. Completed 10M-Step Matched RL Benchmark on DRAC Nibi:",
         "   • Overcame the 200K-step short-horizon SAC bottleneck using 16 CPU workers\n"
         "   • SAC reaches 80% held success (85% align); PPO trains in 69.2 min."),
        ("3. Validated Hierarchical VLA + RL Hybrid Architecture:",
         "   • High-level π₀ phase guidance + 20 Hz RL achieves 100% nominal\n"
         "     and 80% randomized 3D cube reorientation success."),
    ]
    y_pos = 79
    for head, body in bullets:
        card.text(4, y_pos, head, fontsize=10.2, fontweight="bold", color=MAROON, ha="left", va="top")
        card.text(4, y_pos - 6.5, body, fontsize=9.2, color=NAVY, ha="left", va="top", linespacing=1.3)
        y_pos -= 26.5

    save_slide(fig, "slide4_rollout_filmstrip_and_takeaways.jpg")


if __name__ == "__main__":
    make_slide_1()
    make_slide_2()
    make_slide_3()
    make_slide_4()
