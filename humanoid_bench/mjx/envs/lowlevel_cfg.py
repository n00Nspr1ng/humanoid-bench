"""Shared configuration for H1 lowlevel_stand and lowlevel_loco envs.

Edit this file to tune reward scales, obs-noise levels, domain-randomisation
ranges, or command sampling without touching the env implementations.
"""
import numpy as np

# ── Simulation ────────────────────────────────────────────────────────────────
SIM_DT   = 0.004          # physics timestep (s)
N_FRAMES = 5              # control decimation; ctrl_dt = SIM_DT * N_FRAMES = 0.02 s

# ── Action scales ─────────────────────────────────────────────────────────────
ACTION_SCALE_LEGS       = 0.5
ACTION_SCALE_ARMS_PITCH = 1.0
ACTION_SCALE_ARMS       = 0.75   # roll, yaw, elbow

# ── Reward tracking sigmas ────────────────────────────────────────────────────
# Velocity (lin_vel_xy, ang_vel_z): errors in m/s or rad/s
TRACKING_SIGMA_VEL    = 0.25
# Pelvis height: typical converged error ~0.05-0.10 m
# σ=0.05 → err=0.05m: 0.95 reward; err=0.10m: 0.82; err=0.25m: 0.29
TRACKING_SIGMA_HEIGHT = 0.05
# EE position: initial errors 0.3-0.6 m/arm — need gradient even at large errors
# σ=0.35 → combined 0.6m/arm: 0.13 reward; 0.3m/arm: 0.60; 0.1m/arm: 0.95
TRACKING_SIGMA_EE     = 0.35
MAX_CONTACT_FORCE     = 500.0   # N — used for feet_force clip

# ── Observation noise (uniform ±, applied during training only) ───────────────
OBS_NOISE_GYRO   = 0.2
OBS_NOISE_UPVEC  = 0.05
OBS_NOISE_QPOS   = 0.01
OBS_NOISE_QVEL   = 1.5
OBS_NOISE_LINVEL = 0.1

# ── Domain randomisation ──────────────────────────────────────────────────────
DR_FLOOR_FRICTION_LOW  = 0.4
DR_FLOOR_FRICTION_HIGH = 1.0
DR_ARMATURE_LOW        = 1.0
DR_ARMATURE_HIGH       = 1.05
DR_MASS_SCALE_LOW      = 0.9
DR_MASS_SCALE_HIGH     = 1.1
DR_TORSO_MASS_OFFSET   = 1.0    # ± kg added to torso
DR_QPOS_JITTER         = 0.05   # rad — initial qpos noise
DR_DAMPING_LOW         = 0.5    # scale × base damping
DR_DAMPING_HIGH        = 2.0

# ── Velocity command ranges ───────────────────────────────────────────────────
LIN_VEL_X   = (-1.0,  1.5)   # m/s
LIN_VEL_Y   = (-0.8,  0.8)
ANG_VEL_YAW = (-0.7,  0.7)   # rad/s

# ── Whole-body EE command ranges (in torso frame, height in world frame) ──────
WB_LEFT_EE_LOW   = np.array([0.05,  0.0,  -0.19]) # [0.05,  0.0,  -0.19]
WB_LEFT_EE_HIGH  = np.array([0.45,  0.60,  0.20]) # [0.55,  0.70,  0.20]
WB_RIGHT_EE_LOW  = np.array([0.05, -0.60, -0.19]) # [0.05, -0.70, -0.19]
WB_RIGHT_EE_HIGH = np.array([0.45,  0.0,   0.20]) # [0.55,  0.0,   0.20]
WB_HEIGHT_LOW    = 0.6    # m
WB_HEIGHT_HIGH   = 1.0

# ── Reward scales — lowlevel_stand ───────────────────────────────────────────
STAND_REWARD_SCALES = {
    # Exp-tracking
    "track_lin_vel_xy_exp":   1.5,
    "track_ang_vel_z_exp":    2.0,
    "track_ee_pos_exp":       3.0,
    "track_height_exp":       1.5,
    # Orientation
    "body_orientation_l2":   -0.2,   # torso projected gravity
    "flat_orientation_l2":   -0.1,   # pelvis projected gravity
    # Base motion
    "lin_vel_z_l2":          -0.1,
    "ang_vel_xy_l2":         -0.05,
    # Feet
    "feet_slide":            -0.25,
    "feet_force":            -3e-3,
    # Pose
    "joint_deviation_hip":   -0.15,
    "joint_deviation_torso": -0.2,
    "joint_deviation_legs":  -0.02,
    "joint_deviation_arms":  -0.01,
    "dof_pos_limits":        -2.0,
    # EE regularisation
    "penalize_ee_jitter":    -1e-4,
    # Energy / regularisation
    "action_rate_l2":        -0.01,
    "energy":                -1e-3,
    "dof_acc_l2":            -2.5e-7,
    "torques":                0.0,
    # Termination / alive
    "termination_penalty":   -2000.0,
    "alive":                  0.1,
}

# ── Reward scales — lowlevel_loco ─────────────────────────────────────────────
LOCO_REWARD_SCALES = {
    # Exp-tracking
    "track_lin_vel_xy_exp":   1.5,
    "track_ang_vel_z_exp":    2.0,
    "track_ee_pos_exp":       3.0,
    "track_height_exp":       1.5,
    "track_body_roll_exp":    1.75,
    # Orientation
    "body_ori_l1":           -0.2,
    "flat_ori_l2":           -0.1,
    # Base motion
    "lin_vel_z_l2":          -0.1,
    "ang_vel_xy_l2":         -0.05,
    # Feet
    "feet_air_time":          0.15,
    "feet_slide":            -0.25,
    "feet_force":            -3e-3,
    # Pose
    "stand_still":            -1.0,
    "joint_deviation_hip":   -0.15,
    "joint_deviation_torso": -0.2,
    "joint_deviation_legs":  -0.02,
    "joint_deviation_arms":  -0.01,
    "dof_pos_limits":        -2.0,
    # EE regularisation
    "penalize_ee_jitter":    -1e-4,
    # Energy / regularisation
    "action_rate_l2":        -0.01,
    "energy":                -1e-3,
    "dof_acc_l2":            -2.5e-7,
    "torques":                0.0,
    # Termination
    "termination_penalty":   -200.0,
}
