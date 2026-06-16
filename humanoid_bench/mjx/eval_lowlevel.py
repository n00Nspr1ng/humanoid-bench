"""Evaluate a trained lowlevel stand/loco policy on CPU MuJoCo.

Usage:
    cd humanoid_bench/mjx
    # Interactive viewer (passive):
    python eval_lowlevel.py --folder data/lowlevel_stand_v1 --step 57344

    # Render to video:
    python eval_lowlevel.py --folder data/lowlevel_stand_v1 --step 57344 --render

    # Latest checkpoint (no --step):
    python eval_lowlevel.py --folder data/lowlevel_stand_v1
"""

import argparse
import os
import numpy as np
import mujoco
import mujoco.viewer
import tqdm
from pathlib import Path

from flax_to_torch import TorchModel, TorchPolicy

def save_numpy_as_video(array: np.ndarray, filename: str, fps: int = 50) -> None:
    """Save (T, H, W, 3) uint8 array as mp4 using OpenCV."""
    import cv2
    T, H, W, _ = array.shape
    writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for frame in array:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

# ------------------------------------------------------------------ #
# Shared constants — must stay in sync with lowlevel_stand.py / loco.py
# ------------------------------------------------------------------ #
_XML_PATH = str(
    Path(__file__).parents[2]
    / "humanoid_bench/assets/mjx/scene_mjx_h1_lowlevel.xml"
)
_SIM_DT   = 0.004
_N_FRAMES = 5          # ctrl_dt = 0.02 s

_ACTION_SCALE_LEGS       = 0.5
_ACTION_SCALE_ARMS_PITCH = 1.0
_ACTION_SCALE_ARMS       = 0.75

# wb_cmd sampling ranges (must match lowlevel_stand.py)
_WB_LEFT_EE_LOW   = np.array([0.05,  0.0,  -0.19])
_WB_LEFT_EE_HIGH  = np.array([0.55,  0.70,  0.20])
_WB_RIGHT_EE_LOW  = np.array([0.05, -0.70, -0.19])
_WB_RIGHT_EE_HIGH = np.array([0.55,  0.0,   0.20])
_WB_HEIGHT_LOW    = 0.6
_WB_HEIGHT_HIGH   = 1.0

# Observation dims per policy type
_OBS_DIMS = {
    # last_act_start = gyro(3)+upvec(3)+wb(7)+qpos(19)+qvel(19) = 51 for stand
    #                  gyro(3)+upvec(3)+cmd(3)+wb(10)+qpos(19)+qvel(19) = 57 for loco
    "h1_lowlevel_stand": {"actor": 70, "action": 19, "last_act_start": 51},
    "h1_lowlevel_loco":  {"actor": 76, "action": 19, "last_act_start": 57},
}


# ------------------------------------------------------------------ #
# CPU observation builder
# ------------------------------------------------------------------ #

class LowlevelCPUEnv:
    """Thin CPU MuJoCo wrapper that replicates MJX policy observations."""

    def __init__(self, env_name: str):
        self.model = mujoco.MjModel.from_xml_path(_XML_PATH)
        self.model.opt.timestep = _SIM_DT
        self.data  = mujoco.MjData(self.model)
        self.env_name = env_name

        kf = self.model.keyframe("home")
        self._default_pose = kf.qpos[7:].copy()  # 19D
        self._init_q       = kf.qpos.copy()

        lowers = self.model.jnt_range[1:, 0]
        uppers = self.model.jnt_range[1:, 1]
        self._lowers = lowers
        self._uppers = uppers

        # Body IDs
        self._torso_id   = self.model.body("torso_link").id
        self._pelvis_id  = self.model.body("pelvis").id
        self._lhand_site = self.model.site("left_hand").id
        self._rhand_site = self.model.site("right_hand").id

        # Per-joint action scale (same logic as _post_init)
        arm_names   = ["left_shoulder_pitch",  "left_shoulder_roll",
                       "left_shoulder_yaw",    "left_elbow",
                       "right_shoulder_pitch", "right_shoulder_roll",
                       "right_shoulder_yaw",   "right_elbow"]
        pitch_names = ["left_shoulder_pitch", "right_shoulder_pitch"]

        arm_idx   = np.array([self.model.joint(n).qposadr - 7 for n in arm_names])
        pitch_idx = np.array([self.model.joint(n).qposadr - 7 for n in pitch_names])

        scale = np.full(self.model.nu, _ACTION_SCALE_LEGS)
        scale[arm_idx]   = _ACTION_SCALE_ARMS
        scale[pitch_idx] = _ACTION_SCALE_ARMS_PITCH
        self._action_scale = scale

        # Loco command (zeros for stand; random for loco)
        self._loco_cmd = np.zeros(3)

        # Wb_cmd: [left_ee(3), right_ee(3), roll, pitch, yaw, height]
        # Use midpoint of training ranges as default; overridden after first reset.
        self._wb_cmd = np.array([0.3, 0.35, 0.0,
                                  0.3, -0.35, 0.0,
                                  0.0, 0.0, 0.0, 0.8])

        self._init_last_act = np.zeros(self.model.nu)
        self._last_act = self._init_last_act.copy()
        self._step = 0

    def _get_imu(self):
        R = self.data.xmat[self._torso_id].reshape(3, 3)
        gyro         = R.T @ self.data.cvel[self._torso_id, :3]
        upvector     = R[:, 2]
        local_linvel = R.T @ self.data.cvel[self._torso_id, 3:]
        return gyro, upvector, local_linvel

    def _get_obs(self) -> np.ndarray:
        """Return FULL critic obs (75D stand / 81D loco) matching NormalizeVecObservation."""
        gyro, upvector, local_linvel = self._get_imu()
        wb = self._wb_cmd

        if self.env_name == "h1_lowlevel_stand":
            # actor_obs (70D): gyro(3)+upvec(3)+wb_7(7)+qpos_dev(19)+qvel(19)+act(19)
            actor_obs = np.concatenate([
                gyro,
                upvector,
                np.concatenate([wb[:6], wb[9:10]]),   # 7D wb
                self.data.qpos[7:] - self._default_pose,
                self.data.qvel[6:],
                self._last_act,
            ])
            # full critic obs (75D) = actor_obs + local_linvel(3) + feet_contact(2)
            feet = self._get_feet_contact().astype(np.float32)
            return np.concatenate([actor_obs, local_linvel, feet])
        else:  # h1_lowlevel_loco — actor 76D, full 81D
            actor_obs = np.concatenate([
                gyro,
                upvector,
                self._loco_cmd,
                self._wb_cmd,
                self.data.qpos[7:] - self._default_pose,
                self.data.qvel[6:],
                self._last_act,
            ])
            feet = self._get_feet_contact().astype(np.float32)
            return np.concatenate([actor_obs, local_linvel, feet])

    def _get_feet_contact(self) -> np.ndarray:
        left_site_z  = self.data.site_xpos[self.model.site("left_foot").id,  2]
        right_site_z = self.data.site_xpos[self.model.site("right_foot").id, 2]
        return np.array([left_site_z < 0.05, right_site_z < 0.05])

    def reset(self, rng: np.random.Generator | None = None,
              last_act_init: np.ndarray | None = None) -> np.ndarray:
        self.data.qpos[:] = self._init_q
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        if rng is not None:
            self._wb_cmd = self._sample_wb_cmd(rng)
        # else keep the previous wb_cmd (shouldn't happen in normal use)

        if self.env_name == "h1_lowlevel_loco" and rng is not None:
            self._loco_cmd = rng.uniform([-1.0, -0.5, -0.5], [1.5, 0.5, 0.5])

        # Use training mean for last_act if provided: prevents OOD normalisation at reset
        # when the policy has learned a non-zero steady-state action.
        self._last_act = last_act_init.copy() if last_act_init is not None else np.zeros(self.model.nu)
        self._step = 0
        return self._get_obs()

    def _sample_wb_cmd(self, rng: np.random.Generator) -> np.ndarray:
        left_ee  = rng.uniform(_WB_LEFT_EE_LOW,  _WB_LEFT_EE_HIGH)
        right_ee = rng.uniform(_WB_RIGHT_EE_LOW, _WB_RIGHT_EE_HIGH)
        height   = rng.uniform(_WB_HEIGHT_LOW,   _WB_HEIGHT_HIGH)
        return np.concatenate([left_ee, right_ee, np.zeros(3), [height]])

    def step(self, action: np.ndarray, rng: np.random.Generator | None = None):
        motor_targets = np.clip(
            self._default_pose + action * self._action_scale,
            self._lowers, self._uppers,
        )
        self.data.ctrl[:] = motor_targets
        mujoco.mj_step(self.model, self.data, nstep=_N_FRAMES)

        # Store effective action (matches training fix in lowlevel_stand.py).
        self._last_act = (motor_targets - self._default_pose) / self._action_scale

        self._step += 1
        if self._step > 300 and rng is not None:
            self._wb_cmd = self._sample_wb_cmd(rng)
            self._step = 0

        torso_z  = self.data.xpos[self._torso_id, 2]
        R        = self.data.xmat[self._torso_id].reshape(3, 3)
        upvector = R[:, 2]

        lfoot_y = self.data.site_xpos[self.model.site("left_foot").id,  1]
        rfoot_y = self.data.site_xpos[self.model.site("right_foot").id, 1]

        done = (
            upvector[2] < 0.0
            or torso_z < 0.5
            or lfoot_y < rfoot_y
            or np.isnan(self.data.qpos).any()
            or np.isnan(self.data.qvel).any()
        )
        obs = self._get_obs()
        return obs, done


# ------------------------------------------------------------------ #
# Main evaluation loop
# ------------------------------------------------------------------ #

def load_policy(folder: str, env_name: str, step: int | None):
    """Return (policy, mean, var) where mean/var cover the full critic obs."""
    actor_dim  = _OBS_DIMS[env_name]["actor"]
    action_dim = _OBS_DIMS[env_name]["action"]

    suffix = f"_{step}" if step is not None else ""
    model_path = os.path.join(folder, f"torch_model{suffix}.pt")
    mean_path  = os.path.join(folder, f"mean{suffix}.npy")
    var_path   = os.path.join(folder, f"var{suffix}.npy")

    import torch
    torch_model = TorchModel(actor_dim, action_dim)
    torch_model.load_state_dict(torch.load(model_path, weights_only=True))
    torch_model.eval()

    mean = np.load(mean_path)[0]   # shape (full_obs_dim,)
    var  = np.load(var_path)[0]

    print(f"Loaded policy from {model_path}  (mean shape={mean.shape})")
    return torch_model, mean, var


def infer(model, obs_full: np.ndarray, mean: np.ndarray, var: np.ndarray,
          actor_dim: int) -> np.ndarray:
    """Normalize full obs, slice actor obs, run forward pass."""
    import torch
    obs_norm = (obs_full - mean) / np.sqrt(var + 1e-8)
    x = torch.from_numpy(obs_norm[:actor_dim]).float()
    with torch.no_grad():
        action = model(x).numpy()
    return action


def run_rollouts(env: LowlevelCPUEnv, model, mean, var,
                 actor_dim: int, n_rollouts: int, max_steps: int, render: bool,
                 last_act_init: np.ndarray | None = None):
    rng = np.random.default_rng(0)
    all_rewards = []
    all_lengths = []
    frames_list = []

    renderer = None
    if render:
        renderer = mujoco.Renderer(env.model, height=480, width=480)

    for ep in tqdm.tqdm(range(n_rollouts), desc="rollouts"):
        obs = env.reset(rng, last_act_init=last_act_init)
        total_r = 0.0
        frames  = []

        for t in range(max_steps):
            action = infer(model, obs, mean, var, actor_dim)
            obs, done = env.step(action, rng)

            R = env.data.xmat[env._torso_id].reshape(3, 3)
            total_r += float(R[2, 2])   # upright reward proxy

            if render and renderer is not None:
                renderer.update_scene(env.data, camera="cam_default")
                frames.append(renderer.render().copy())

            if done:
                break

        all_rewards.append(total_r)
        all_lengths.append(t + 1)
        if render:
            frames_list.append(np.array(frames))

    print(f"\nEpisode returns:  mean={np.mean(all_rewards):.1f}  "
          f"min={np.min(all_rewards):.1f}  max={np.max(all_rewards):.1f}")
    print(f"Episode lengths:  mean={np.mean(all_lengths):.0f}  "
          f"min={np.min(all_lengths)}  max={np.max(all_lengths)}")

    return frames_list


def _draw_ee_targets(v, env: LowlevelCPUEnv) -> None:
    """Draw EE command targets and actual EE positions as spheres + coordinate axes."""
    R         = env.data.xmat[env._torso_id].reshape(3, 3)
    torso_pos = env.data.xpos[env._torso_id]
    wb        = env._wb_cmd

    # World-frame target positions from wb_cmd
    l_target = torso_pos + R @ wb[:3]
    r_target = torso_pos + R @ wb[3:6]

    # Actual EE positions
    l_actual = env.data.site_xpos[env._lhand_site]
    r_actual = env.data.site_xpos[env._rhand_site]

    AXIS_LEN   = 0.08   # length of each coordinate axis arrow
    AXIS_RAD   = 0.008
    TARGET_RAD = 0.025
    ACTUAL_RAD = 0.015

    # Per-axis unit vectors and colours
    axes = [
        (np.array([1, 0, 0]), np.array([1.0, 0.1, 0.1, 1.0])),  # X red
        (np.array([0, 1, 0]), np.array([0.1, 1.0, 0.1, 1.0])),  # Y green
        (np.array([0, 0, 1]), np.array([0.1, 0.1, 1.0, 1.0])),  # Z blue
    ]

    geoms = v.user_scn.geoms
    n = 0

    def _add(geom_type, size, pos, mat, rgba):
        nonlocal n
        if n >= v.user_scn.maxgeom:
            return
        mujoco.mjv_initGeom(geoms[n], geom_type, size, pos, mat, rgba)
        n += 1

    identity = np.eye(3).flatten()

    for target_pos, actual_pos, target_rgba, actual_rgba in [
        (l_target, l_actual,
         np.array([1.0, 0.3, 0.3, 0.9]), np.array([1.0, 0.6, 0.6, 0.6])),
        (r_target, r_actual,
         np.array([0.3, 0.3, 1.0, 0.9]), np.array([0.6, 0.6, 1.0, 0.6])),
    ]:
        # Target sphere
        _add(mujoco.mjtGeom.mjGEOM_SPHERE,
             np.array([TARGET_RAD, 0, 0]), target_pos, identity, target_rgba)

        # Coordinate frame axes at target (world-aligned)
        for axis_dir, axis_rgba in axes:
            start = target_pos
            end   = target_pos + axis_dir * AXIS_LEN
            _add(mujoco.mjtGeom.mjGEOM_CAPSULE,
                 np.array([AXIS_RAD, AXIS_LEN / 2, 0]),
                 (start + end) / 2,
                 # rotation matrix: z-axis of capsule → axis_dir
                 _look_along(axis_dir),
                 axis_rgba)

        # Actual EE position (smaller, transparent sphere)
        _add(mujoco.mjtGeom.mjGEOM_SPHERE,
             np.array([ACTUAL_RAD, 0, 0]), actual_pos, identity, actual_rgba)

    v.user_scn.ngeom = n


def _look_along(direction: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix whose Z column points along direction, flattened."""
    z = direction / np.linalg.norm(direction)
    # pick an arbitrary perpendicular x-axis
    x = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(np.cross(z, x), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z]).flatten()


def run_interactive(env: LowlevelCPUEnv, model, mean, var, actor_dim: int,
                    last_act_init: np.ndarray | None = None):
    """Open the MuJoCo passive viewer and run the policy in real time."""
    rng = np.random.default_rng(0)
    obs = env.reset(rng, last_act_init=last_act_init)
    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        print("Passive viewer open — close the window to exit.")
        print("  Red sphere/axes  = left  EE command target")
        print("  Blue sphere/axes = right EE command target")
        print("  Faint spheres    = actual EE positions")
        step = 0
        while v.is_running():
            action = infer(model, obs, mean, var, actor_dim)
            obs, done = env.step(action, rng)
            _draw_ee_targets(v, env)
            v.sync()
            step += 1
            if done or step > 2000:
                obs = env.reset(rng, last_act_init=last_act_init)
                step = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder",   type=str, required=True,
                        help="checkpoint folder, e.g. data/lowlevel_stand_v1")
    parser.add_argument("--env_name", type=str, default="h1_lowlevel_stand",
                        choices=list(_OBS_DIMS.keys()))
    parser.add_argument("--step",     type=int, default=None,
                        help="checkpoint step (omit to use final torch_model.pt)")
    parser.add_argument("--render",   action="store_true",
                        help="render rollouts to video instead of interactive viewer")
    parser.add_argument("--n_rollouts", type=int, default=5)
    parser.add_argument("--max_steps",  type=int, default=1000)
    args = parser.parse_args()

    model, mean, var = load_policy(args.folder, args.env_name, args.step)
    actor_dim = _OBS_DIMS[args.env_name]["actor"]
    last_act_start = _OBS_DIMS[args.env_name]["last_act_start"]
    last_act_init = mean[last_act_start : last_act_start + 19]
    env = LowlevelCPUEnv(args.env_name)

    if args.render:
        frames_list = run_rollouts(env, model, mean, var, actor_dim,
                                   args.n_rollouts, args.max_steps, render=True,
                                   last_act_init=last_act_init)
        out_path = os.path.join(args.folder, "evaluation.mp4")
        all_frames = np.concatenate(frames_list, axis=0)
        save_numpy_as_video(all_frames, out_path, fps=50)
        print(f"Video saved to {out_path}")
    else:
        run_interactive(env, model, mean, var, actor_dim, last_act_init=last_act_init)


if __name__ == "__main__":
    main()
