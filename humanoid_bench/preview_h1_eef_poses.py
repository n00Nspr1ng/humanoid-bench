"""
Interactive MuJoCo viewer for H1 arm configurations.

Shows the H1 in its home keyframe pose first, then cycles through
random arm configs within the action_scale=0.75 window.
EEF positions in the torso frame are printed to console on each resample.

Usage
-----
    python humanoid_bench/preview_h1_eef_poses.py
    python humanoid_bench/preview_h1_eef_poses.py --hold_secs 3.0 --seed 1
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

_XML = Path(__file__).parent / "assets/mjx/scene_mjx_h1_lowlevel.xml"
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


def _effective_ranges():
    names = list(ARM_JOINTS.keys())
    idx   = np.array([ARM_JOINTS[n][0] for n in names])
    lo    = np.array([max(ARM_JOINTS[n][1] - ARM_JOINTS[n][4], ARM_JOINTS[n][2]) for n in names])
    hi    = np.array([min(ARM_JOINTS[n][1] + ARM_JOINTS[n][4], ARM_JOINTS[n][3]) for n in names])
    return idx, lo, hi


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)    ],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)    ],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def _ee_in_torso(d, torso_id, left_sid, right_sid):
    R         = _quat_to_mat(d.xquat[torso_id])
    torso_pos = d.xpos[torso_id]
    left  = R.T @ (d.site_xpos[left_sid]  - torso_pos)
    right = R.T @ (d.site_xpos[right_sid] - torso_pos)
    return left, right


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hold_secs", type=float, default=2.0,
                        help="Seconds to display each pose before resampling.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    m = mujoco.MjModel.from_xml_path(str(_XML))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)

    torso_id  = m.body("torso_link").id
    left_sid  = m.site("left_hand").id
    right_sid = m.site("right_hand").id

    idx, lo, hi = _effective_ranges()
    base_qpos   = d.qpos.copy()

    print("Arm action_scale: pitch=1.0, roll/yaw/elbow=0.75")
    print("Effective arm joint ranges:")
    names = list(ARM_JOINTS.keys())
    for i, name in enumerate(names):
        print(f"  {name:26s} [{lo[i]:+.3f}, {hi[i]:+.3f}] rad")
    print(f"\nHolding each pose {args.hold_secs}s — close the viewer to exit.\n")

    step = 0
    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
        viewer.cam.distance  = 3.5
        viewer.cam.elevation = -20
        viewer.cam.azimuth   = 160

        while viewer.is_running():
            # Build pose
            d.qpos[:] = base_qpos
            if step > 0:
                d.qpos[idx] = rng.uniform(lo, hi)

            d.qvel[:] = 0.0
            mujoco.mj_kinematics(m, d)

            left, right = _ee_in_torso(d, torso_id, left_sid, right_sid)
            label = "default" if step == 0 else f"sample {step}"
            print(f"[{label}]")
            print(f"  LEFT  EE (torso frame): x={left[0]:+.3f}  y={left[1]:+.3f}  z={left[2]:+.3f}")
            print(f"  RIGHT EE (torso frame): x={right[0]:+.3f}  y={right[1]:+.3f}  z={right[2]:+.3f}")
            if step > 0:
                print(f"  arm joints: {d.qpos[idx].round(3).tolist()}")

            viewer.sync()

            t0 = time.time()
            while viewer.is_running() and (time.time() - t0) < args.hold_secs:
                viewer.sync()
                time.sleep(0.016)

            step += 1


if __name__ == "__main__":
    main()
