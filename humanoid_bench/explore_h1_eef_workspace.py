"""
Explore the reachable workspace of H1 left_hand / right_hand sites
within the action_scale=0.75 window around the home keyframe.

Outputs a PNG with top / side / front / 3-D scatter views and the
wb_cmd target box overlaid for reference.

Usage
-----
    python humanoid_bench/explore_h1_eef_workspace.py
    python humanoid_bench/explore_h1_eef_workspace.py --n_samples 30000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mujoco

_XML = Path(__file__).parent / "assets/mjx/scene_mjx_h1_lowlevel.xml"
_ACTION_SCALE_ARMS_PITCH = 1.0
_ACTION_SCALE_ARMS       = 0.75  # roll, yaw, elbow

# (qpos_index, default_val, joint_lo, joint_hi, action_scale)
ARM_JOINTS: dict[str, tuple[int, float, float, float, float]] = {
    "left_shoulder_pitch":  (18,  0.0, -2.87, 2.87, 1.00),
    "left_shoulder_roll":   (19,  0.3, -0.34, 3.11, 0.75),
    "left_shoulder_yaw":    (20,  0.0, -1.30, 4.45, 0.75),
    "left_elbow":           (21,  0.8, -1.25, 2.61, 0.75),
    "right_shoulder_pitch": (22,  0.0, -2.87, 2.87, 1.00),
    "right_shoulder_roll":  (23, -0.3, -3.11, 0.34, 0.75),
    "right_shoulder_yaw":   (24,  0.0, -4.45, 1.30, 0.75),
    "right_elbow":          (25,  0.8, -1.25, 2.61, 0.75),
}

# wb_cmd target ranges from G1 config (shown as reference boxes)
_LEFT_TARGET  = dict(x=(0.05, 0.35), y=(0.0,  0.40), z=(0.0, 0.40))
_RIGHT_TARGET = dict(x=(0.05, 0.35), y=(-0.40, 0.0), z=(0.0, 0.40))


def _effective_ranges():
    names = list(ARM_JOINTS.keys())
    idx   = np.array([ARM_JOINTS[n][0] for n in names])
    lo    = np.array([max(ARM_JOINTS[n][1] - ARM_JOINTS[n][4], ARM_JOINTS[n][2]) for n in names])
    hi    = np.array([min(ARM_JOINTS[n][1] + ARM_JOINTS[n][4], ARM_JOINTS[n][3]) for n in names])
    return idx, lo, hi


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    """MuJoCo quaternion [w,x,y,z] → 3×3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y)      ],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z),   2*(y*z - w*x)      ],
        [2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x*x + y*y)  ],
    ])


def _to_torso(pos_w, torso_pos, torso_quat):
    return _quat_to_mat(torso_quat).T @ (pos_w - torso_pos)


def sample_workspace(n_samples: int):
    m = mujoco.MjModel.from_xml_path(str(_XML))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)

    torso_id      = m.body("torso_link").id
    left_site_id  = m.site("left_hand").id
    right_site_id = m.site("right_hand").id

    idx, lo, hi = _effective_ranges()
    rng     = np.random.default_rng(42)
    configs = rng.uniform(lo, hi, size=(n_samples, len(idx)))

    base_qpos = d.qpos.copy()
    left_pts  = np.zeros((n_samples, 3))
    right_pts = np.zeros((n_samples, 3))

    for i in range(n_samples):
        d.qpos[:]   = base_qpos
        d.qpos[idx] = configs[i]
        mujoco.mj_kinematics(m, d)

        torso_pos  = d.xpos[torso_id].copy()
        torso_quat = d.xquat[torso_id].copy()
        left_pts[i]  = _to_torso(d.site_xpos[left_site_id],  torso_pos, torso_quat)
        right_pts[i] = _to_torso(d.site_xpos[right_site_id], torso_pos, torso_quat)

        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{n_samples}")

    return left_pts, right_pts


def _default_ee():
    m = mujoco.MjModel.from_xml_path(str(_XML))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_kinematics(m, d)
    torso_id = m.body("torso_link").id
    torso_pos, torso_quat = d.xpos[torso_id].copy(), d.xquat[torso_id].copy()
    left  = _to_torso(d.site_xpos[m.site("left_hand").id],  torso_pos, torso_quat)
    right = _to_torso(d.site_xpos[m.site("right_hand").id], torso_pos, torso_quat)
    return left, right


def _add_target_rect(ax, box, xi, yi, color):
    x0, x1 = box["xyz"[xi]]
    y0, y1 = box["xyz"[yi]]
    ax.add_patch(plt.Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        fill=False, edgecolor=color, linewidth=1.5, linestyle="--",
    ))


def plot_and_save(left_pts, right_pts, default_l, default_r, out_base: str):
    p   = 5
    ds  = max(1, len(left_pts) // 3000)

    print("\n=== Workspace bounds (5/95 pct) ===")
    for side, pts in [("LEFT ", left_pts), ("RIGHT", right_pts)]:
        lo = np.percentile(pts, p,     axis=0)
        hi = np.percentile(pts, 100-p, axis=0)
        print(f"  {side}  x=[{lo[0]:.3f}, {hi[0]:.3f}]"
              f"  y=[{lo[1]:.3f}, {hi[1]:.3f}]"
              f"  z=[{lo[2]:.3f}, {hi[2]:.3f}]")

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(
        "H1 hand workspace in torso frame  (pitch=1.0, roll/yaw/elbow=0.75)",
        fontsize=13,
    )

    def scatter2d(ax, xi, yi, xlabel, ylabel, title):
        ax.scatter(left_pts[::ds,  xi], left_pts[::ds,  yi], s=1, alpha=0.2, c="royalblue", label="left")
        ax.scatter(right_pts[::ds, xi], right_pts[::ds, yi], s=1, alpha=0.2, c="tomato",    label="right")
        ax.scatter(default_l[xi], default_l[yi], s=80, c="blue", marker="*", zorder=6, label="L default")
        ax.scatter(default_r[xi], default_r[yi], s=80, c="red",  marker="*", zorder=6, label="R default")
        _add_target_rect(ax, _LEFT_TARGET,  xi, yi, "blue")
        _add_target_rect(ax, _RIGHT_TARGET, xi, yi, "red")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.axhline(0, c="k", lw=0.5); ax.axvline(0, c="k", lw=0.5)
        ax.grid(True, lw=0.3); ax.set_aspect("equal")
        ax.legend(markerscale=5, fontsize=7)

    fig.add_subplot(2, 3, 1); scatter2d(plt.gca(), 0, 1, "x fwd (m)", "y left (m)", "Top view  (x–y)")
    fig.add_subplot(2, 3, 2); scatter2d(plt.gca(), 0, 2, "x fwd (m)", "z up (m)",   "Side view (x–z)")
    fig.add_subplot(2, 3, 3); scatter2d(plt.gca(), 1, 2, "y left (m)", "z up (m)",  "Front view (y–z)")

    ax4 = fig.add_subplot(2, 3, 4, projection="3d")
    ax4.scatter(left_pts[::ds, 0], left_pts[::ds, 1], left_pts[::ds, 2],
                s=1, alpha=0.15, c=left_pts[::ds, 2], cmap="Blues_r")
    ax4.scatter(*default_l, s=80, c="blue", marker="*", zorder=6)
    ax4.set_title("LEFT hand (3D)"); ax4.set_xlabel("x"); ax4.set_ylabel("y"); ax4.set_zlabel("z")

    ax5 = fig.add_subplot(2, 3, 5, projection="3d")
    ax5.scatter(right_pts[::ds, 0], right_pts[::ds, 1], right_pts[::ds, 2],
                s=1, alpha=0.15, c=right_pts[::ds, 2], cmap="Reds_r")
    ax5.scatter(*default_r, s=80, c="red", marker="*", zorder=6)
    ax5.set_title("RIGHT hand (3D)"); ax5.set_xlabel("x"); ax5.set_ylabel("y"); ax5.set_zlabel("z")

    # Joint range bar chart
    ax6 = fig.add_subplot(2, 3, 6)
    names = list(ARM_JOINTS.keys())
    _, lo_arr, hi_arr = _effective_ranges()
    y = np.arange(len(names))
    colors = ["royalblue"] * 4 + ["tomato"] * 4
    ax6.barh(y, hi_arr - lo_arr, left=lo_arr, color=colors, alpha=0.75)
    ax6.set_yticks(y)
    ax6.set_yticklabels([n.replace("_", "\n") for n in names], fontsize=7)
    ax6.axvline(0, c="k", lw=0.8)
    ax6.set_xlabel("joint angle (rad)")
    ax6.set_title("Effective arm joint ranges  (pitch=1.0, roll/yaw/elbow=0.75)")
    ax6.grid(True, axis="x", lw=0.3)

    plt.tight_layout()
    out = Path(out_base)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out) + ".png", dpi=150)
    print(f"\nSaved → {out}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=20000)
    parser.add_argument(
        "--output", type=str,
        default=str(Path(__file__).parent / "assets" / "h1_eef_workspace"),
    )
    args = parser.parse_args()

    default_l, default_r = _default_ee()
    print(f"Default LEFT  EE in torso frame: {default_l.round(3)}")
    print(f"Default RIGHT EE in torso frame: {default_r.round(3)}")

    print(f"\nSampling {args.n_samples} configurations within action_scale={_ACTION_SCALE_ARMS}...")
    left_pts, right_pts = sample_workspace(args.n_samples)

    plot_and_save(left_pts, right_pts, default_l, default_r, args.output)


if __name__ == "__main__":
    main()
