"""H1 lowlevel locomotion + whole-body policy (Brax/MJX)."""

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from brax.envs.base import MjxEnv, State

from humanoid_bench.mjx.envs.utils import perturbed_pipeline_step

from pathlib import Path
_XML_PATH = str(Path(__file__).parents[3] / "humanoid_bench/assets/mjx/scene_mjx_h1_lowlevel.xml")
_SIM_DT = 0.004
_N_FRAMES = 5        # ctrl_dt = 0.02

_ACTION_SCALE_LEGS        = 0.5
_ACTION_SCALE_ARMS_PITCH  = 1.0
_ACTION_SCALE_ARMS        = 0.75   # roll, yaw, elbow
_TRACKING_SIGMA = 0.25
_MAX_CONTACT_FORCE = 500.0

_GRAVITY = jp.array([0.0, 0.0, -1.0])

_LIN_VEL_X   = (-1.0, 1.5)
_LIN_VEL_Y   = (-0.8, 0.8)
_ANG_VEL_YAW = (-0.7, 0.7)

# Whole-body command ranges (EE positions in torso frame, height in world frame)
# Default EE at home pose ≈ left=[0.243, 0.387, -0.123], right=[0.243, -0.387, -0.123]
_WB_LEFT_EE_LOW  = np.array([0.05,  0.0,  -0.19])  # x, y, z
_WB_LEFT_EE_HIGH = np.array([0.55,  0.70,  0.20])
_WB_RIGHT_EE_LOW  = np.array([0.05, -0.70,  -0.19])
_WB_RIGHT_EE_HIGH = np.array([0.55, 0.0,  0.20])
_WB_HEIGHT_LOW  = 0.6
_WB_HEIGHT_HIGH = 1.0

# Reward scales (positive = bonus, negative = penalty)
# Exp-tracking weights match Isaac Lab G1RewardCfg exactly.
# Newly-added terms (not in Isaac Lab) use mujoco_playground G1 weights.
REWARD_SCALES = {
    # Exp tracking rewards — Isaac Lab weights
    # always track zero velocity (stand in place)
    "track_lin_vel_xy_exp":   1.5,
    "track_ang_vel_z_exp":    2.0,
    "track_ee_pos_exp":       3.0,
    "track_height_exp":       1.5,
    # "track_body_roll_exp":    1.75, # NOTE: not for lowlevel_stand
    "body_orientation_l2":  -0.2, # torso projected gravity
    "flat_orientation_l2":  -0.1, # pelvis projected gravity
    # Base motion costs — Isaac Lab weights
    "lin_vel_z_l2":          -0.1,
    "ang_vel_xy_l2":         -0.05,
    # Feet — Isaac Lab weights 
    # "feet_air_time":          0.15, # NOTE: not for lowlevel_stand
    "feet_slide":            -0.25,
    "feet_force":            -3e-3, 
    # Pose — Isaac Lab for existing; mujoco_playground for new
    "joint_deviation_hip":    -0.15,
    "joint_deviation_torso":  -0.2,    # H1 torso yaw ≡ G1 waist_yaw
    "joint_deviation_legs":   -0.02,
    "joint_deviation_arms":   -0.01,
    "dof_pos_limits":         -2.0,
    # EE regularisation
    "penalize_ee_jitter":     -1e-4,   # L2 norm of finite-differenced EE linear velocity
    # Energy / regularisation — Isaac Lab weights
    "action_rate_l2":        -0.01,
    "energy":                -1e-3,
    "dof_acc_l2":            -2.5e-7,
    "torques":                0.0,
    # Termination — Isaac Lab weight
    "termination_penalty":   -200.0,
}


class H1LowLevelStand(MjxEnv):
    """H1 lowlevel locomotion + whole-body command tracking."""

    def __init__(self, **kwargs):
        mj_model = mujoco.MjModel.from_xml_path(_XML_PATH)
        mj_model.opt.timestep = _SIM_DT
        kwargs["n_frames"] = kwargs.get("n_frames", _N_FRAMES)
        super().__init__(model=mj_model, **kwargs)
        self._post_init(mj_model)

    def _post_init(self, mj_model: mujoco.MjModel) -> None:
        home_kf = mj_model.keyframe("home")
        self._init_q       = jp.array(home_kf.qpos)
        self._default_pose = jp.array(home_kf.qpos[7:])  # 19D

        # Joint limits (skip free joint at index 0)
        self._lowers = jp.array(mj_model.jnt_range[1:, 0])
        self._uppers = jp.array(mj_model.jnt_range[1:, 1])

        # Soft joint limits — 10% inset from each hard limit
        _jnt_margin = 0.1
        self._soft_lowers = self._lowers + _jnt_margin * (self._uppers - self._lowers)
        self._soft_uppers = self._uppers - _jnt_margin * (self._uppers - self._lowers)

        # Site / body IDs
        self._left_foot_site_id  = mj_model.site("left_foot").id
        self._right_foot_site_id = mj_model.site("right_foot").id
        self._left_hand_site_id  = mj_model.site("left_hand").id
        self._right_hand_site_id = mj_model.site("right_hand").id
        self._pelvis_body_id     = mj_model.body("pelvis").id
        self._torso_body_id      = mj_model.body("torso_link").id

        # Hip joint indices within qpos[7:] for joint_deviation_hip
        # order: left_hip_yaw(0), left_hip_roll(1), right_hip_yaw(5), right_hip_roll(6)
        hip_names = [
            "left_hip_yaw", "left_hip_roll",
            "right_hip_yaw", "right_hip_roll",
        ]
        self._hip_indices = jp.array(
            [mj_model.joint(n).qposadr - 7 for n in hip_names]
        )

        # Leg indices (pitch / knee / ankle) for joint_deviation_legs
        leg_names = [
            "left_hip_pitch",  "left_knee",  "left_ankle",
            "right_hip_pitch", "right_knee", "right_ankle",
        ]
        self._leg_indices = jp.array(
            [mj_model.joint(n).qposadr - 7 for n in leg_names]
        )

        # Arm indices (shoulders + elbows) for joint_deviation_arms
        arm_names = [
            "left_shoulder_pitch",  "left_shoulder_roll",  "left_shoulder_yaw",  "left_elbow",
            "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
        ]
        self._arm_indices = jp.array(
            [mj_model.joint(n).qposadr - 7 for n in arm_names]
        )

        # Per-joint action scale
        _pitch_names = ["left_shoulder_pitch", "right_shoulder_pitch"]
        _pitch_idx   = jp.array([mj_model.joint(n).qposadr - 7 for n in _pitch_names])
        _scale = jp.full(mj_model.nu, _ACTION_SCALE_LEGS)
        _scale = _scale.at[self._arm_indices].set(_ACTION_SCALE_ARMS)
        self._action_scale = _scale.at[_pitch_idx].set(_ACTION_SCALE_ARMS_PITCH)

        # Torso joint index for joint_deviation_torso (equivalent of G1 waist_yaw)
        self._torso_jnt_idx = int(mj_model.joint("torso").qposadr) - 7

        # Foot body IDs (for velocity / contact queries via MJX data)
        self._left_ankle_body_id  = mj_model.body("left_ankle_link").id
        self._right_ankle_body_id = mj_model.body("right_ankle_link").id

        # Foot collision geom IDs — find first non-mesh geom on each ankle body
        def _foot_geom(body_id):
            for g in range(mj_model.ngeom):
                if (mj_model.geom_bodyid[g] == body_id
                        and mj_model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH):
                    return g
            raise ValueError(f"No collision geom on body {body_id}")

        self._left_foot_geom_id  = _foot_geom(self._left_ankle_body_id)
        self._right_foot_geom_id = _foot_geom(self._right_ankle_body_id)

        # wb_cmd is 10D: [left_ee(3), right_ee(3), roll, pitch, yaw, height]
        # obs uses 7D: [left_ee(3), right_ee(3), height] — roll/pitch/yaw excluded
        self.actor_obs_dim = 3 + 3 + 7 + 19 + 19 + 19  # 70
        # critic obs appends local_linvel(3) + feet_contact(2)
        self.state_dim  = self.actor_obs_dim + 3 + 2    # 75
        self.action_dim = self.sys.nu                        # 19

    # ------------------------------------------------------------------ #
    #  Command sampling                                                    #
    # ------------------------------------------------------------------ #

    def sample_command(self, rng: jax.Array) -> jax.Array:
        """Always zero — stand policy tracks zero velocity."""
        del rng
        return jp.zeros(3)

    def sample_wb_cmd(self, rng: jax.Array) -> jax.Array:
        """Sample a 10D whole-body command [left_ee(3), right_ee(3), roll, pitch, yaw, height]."""
        rng1, rng2, rng3 = jax.random.split(rng, 3)
        left_ee  = jax.random.uniform(rng1, shape=(3,),
                       minval=jp.array(_WB_LEFT_EE_LOW),
                       maxval=jp.array(_WB_LEFT_EE_HIGH))
        right_ee = jax.random.uniform(rng2, shape=(3,),
                       minval=jp.array(_WB_RIGHT_EE_LOW),
                       maxval=jp.array(_WB_RIGHT_EE_HIGH))
        height   = jax.random.uniform(rng3, shape=(1,),
                       minval=_WB_HEIGHT_LOW, maxval=_WB_HEIGHT_HIGH)
        return jp.concatenate([left_ee, right_ee, jp.zeros(3), height])

    # ------------------------------------------------------------------ #
    #  Reset / Step                                                        #
    # ------------------------------------------------------------------ #

    def reset(self, rng: jax.Array) -> State:
        rng, cmd_rng, wb_rng = jax.random.split(rng, 3)

        data     = self.pipeline_init(self._init_q, jp.zeros(self.sys.nv))
        loco_cmd = self.sample_command(cmd_rng)
        wb_cmd   = self.sample_wb_cmd(wb_rng)

        info = {
            "rng":              rng,
            "step":             jp.zeros((), dtype=jp.int32),
            "command":          loco_cmd,
            "wb_cmd":           wb_cmd,
            "last_act":         jp.zeros(self.sys.nu),
            "feet_air_time":    jp.zeros(2),
            "last_contact":     jp.zeros(2, dtype=bool),
            "last_left_ee_pos": data.data.site_xpos[self._left_hand_site_id],
            "last_right_ee_pos":data.data.site_xpos[self._right_hand_site_id],
        }
        metrics = {k: jp.zeros(()) for k in REWARD_SCALES}

        obs         = self._get_obs(data.data, info)
        reward, done = jp.zeros(2)
        return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        rng, cmd_rng, wb_rng = jax.random.split(state.info["rng"], 3)

        motor_targets = jp.clip(
            self._default_pose + action * self._action_scale,
            self._lowers, self._uppers,
        )
        xfrc_applied = jp.zeros((self.sys.nbody, 6))
        data = perturbed_pipeline_step(
            self.sys, state.pipeline_state, motor_targets, xfrc_applied, self._n_frames
        )

        contact = self._get_foot_contact(data.data)
        contact_filt  = contact | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
        feet_air_time = (state.info["feet_air_time"] + self.dt) * ~contact

        obs = self._get_obs(data.data, state.info)

        # Termination conditions
        upvector = data.data.xmat[self._torso_body_id, :, 2]
        torso_z  = data.data.xpos[self._torso_body_id, 2]
        lfoot_y  = data.data.site_xpos[self._left_foot_site_id, 1]
        rfoot_y  = data.data.site_xpos[self._right_foot_site_id, 1]
        done = (
            (upvector[2] < 0.0)                    # fallen over
            | (torso_z < _WB_HEIGHT_LOW - 0.1)     # torso below minimum commanded height
            | (lfoot_y < rfoot_y)         # legs crossed
            | jp.isnan(data.data.qpos).any()
            | jp.isnan(data.data.qvel).any()
        )

        rewards = self._get_reward(
            data.data, action, state.info, first_contact, contact, done
        )
        rewards = {k: v * REWARD_SCALES[k] for k, v in rewards.items()}
        reward  = jp.clip(sum(rewards.values()) * self.dt, -10000.0, 10000.0)

        # Update info in-place
        state.info["rng"]               = rng
        state.info["last_act"]          = action
        state.info["feet_air_time"]     = feet_air_time
        state.info["last_contact"]      = contact
        state.info["last_left_ee_pos"]  = data.data.site_xpos[self._left_hand_site_id]
        state.info["last_right_ee_pos"] = data.data.site_xpos[self._right_hand_site_id]
        state.info["step"]              = state.info["step"] + 1
        state.info["command"] = jp.where(
            state.info["step"] > 500,
            self.sample_command(cmd_rng),
            state.info["command"],
        )
        state.info["wb_cmd"] = jp.where(
            state.info["step"] > 500,
            self.sample_wb_cmd(wb_rng),
            state.info["wb_cmd"],
        )
        state.info["step"] = jp.where(
            done | (state.info["step"] > 500),
            jp.zeros((), dtype=jp.int32),
            state.info["step"],
        )

        for k, v in rewards.items():
            state.metrics[k] = v

        done = done.astype(jp.float32)
        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)

    # ------------------------------------------------------------------ #
    #  IMU / contact helpers (replace MJX-unavailable sensordata)         #
    # ------------------------------------------------------------------ #

    def _get_imu(self, data):
        """Return (gyro, upvector, local_linvel) computed from MJX data."""
        R = data.xmat[self._torso_body_id]           # (3, 3) in MJX
        gyro        = R.T @ data.cvel[self._torso_body_id, :3]  # body-frame ang vel
        upvector    = R[:, 2]                                    # torso z-axis in world
        local_linvel = R.T @ data.cvel[self._torso_body_id, 3:] # body-frame lin vel
        return gyro, upvector, local_linvel

    def _get_foot_contact(self, data) -> jax.Array:
        """Return (2,) bool array [left, right] foot in contact."""
        geoms  = data.contact.geom   # (ncon_max, 2)
        active = data.contact.dist < 1e-3
        l = jp.any(jp.any(geoms == self._left_foot_geom_id,  axis=-1) & active)
        r = jp.any(jp.any(geoms == self._right_foot_geom_id, axis=-1) & active)
        return jp.array([l, r])

    # ------------------------------------------------------------------ #
    #  Observation                                                         #
    # ------------------------------------------------------------------ #

    def _get_obs(self, data, info: dict) -> jax.Array:
        """Returns 75D critic obs = actor_obs(70) + local_linvel(3) + feet_contact(2)."""
        gyro, upvector, local_linvel = self._get_imu(data)
        feet_contact = self._get_foot_contact(data).astype(jp.float32)
        wb_cmd = info["wb_cmd"]
        actor_obs = jp.concatenate([
            gyro,                                           #  3 – body-frame angular velocity
            upvector,                                       #  3 – torso z-axis in world (gravity proxy)
            jp.concatenate([wb_cmd[:6], wb_cmd[9:10]]),     #  7 – wb_cmd: [l_ee(3), r_ee(3), height]
            data.qpos[7:] - self._default_pose,             # 19 – joint position deviation
            data.qvel[6:],                                  # 19 – joint velocities
            info["last_act"],                               # 19 – last action
        ])  # 70D
        return jp.concatenate([actor_obs, local_linvel, feet_contact])  # 75D

    # ------------------------------------------------------------------ #
    #  Reward computation                                                  #
    # ------------------------------------------------------------------ #

    def _get_reward(
        self, data, action, info, first_contact, contact, done
    ) -> dict:
        gyro, upvector, local_linvel = self._get_imu(data)
        cmd         = info["command"]
        wb_cmd      = info["wb_cmd"]
        base_pos    = data.qpos[:3]

        return {
            "track_lin_vel_xy_exp":  self._reward_track_lin_vel_xy_exp(cmd, local_linvel),
            "track_ang_vel_z_exp":   self._reward_track_ang_vel_z_exp(cmd, gyro),
            "track_ee_pos_exp":      self._reward_track_ee_pos_exp(data, wb_cmd),
            "track_height_exp":      self._reward_track_height_exp(base_pos[2], wb_cmd[9]),
            # "track_body_roll_exp":   self._reward_track_body_roll_exp(upvector, wb_cmd[6]),  # NOTE: not for lowlevel_stand
            "body_orientation_l2":   self._cost_body_orientation_l2(data),
            "flat_orientation_l2":   self._cost_flat_orientation_l2(data),
            "lin_vel_z_l2":          self._cost_lin_vel_z_l2(data),
            "ang_vel_xy_l2":         self._cost_ang_vel_xy_l2(gyro),
            # "feet_air_time":         self._reward_feet_air_time(  # NOTE: not for lowlevel_stand
            #     info["feet_air_time"], first_contact, cmd
            # ),
            "feet_slide":            self._cost_feet_slide(data, contact),
            "feet_force":            self._cost_feet_force(data),
            "joint_deviation_hip":    self._cost_joint_deviation_hip(data.qpos[7:], cmd),
            "joint_deviation_torso":  self._cost_joint_deviation_torso(data.qpos[7:]),
            "joint_deviation_legs":   self._cost_joint_deviation_legs(data.qpos[7:]),
            "joint_deviation_arms":   self._cost_joint_deviation_arms(data.qpos[7:]),
            "dof_pos_limits":         self._cost_dof_pos_limits(data.qpos[7:]),
            "penalize_ee_jitter":     self._cost_penalize_ee_jitter(
                data.site_xpos[self._left_hand_site_id],
                data.site_xpos[self._right_hand_site_id],
                info["last_left_ee_pos"],
                info["last_right_ee_pos"],
            ),
            "action_rate_l2":        self._cost_action_rate_l2(action, info["last_act"]),
            "energy":                self._cost_energy(data.qvel[6:], data.qfrc_actuator),
            "dof_acc_l2":            self._cost_dof_acc_l2(data.qacc[6:]),
            "torques":               self._cost_torques(data.qfrc_actuator[6:]),
            "termination_penalty":   done.astype(jp.float32),
        }

    # --- tracking (exp rewards) ---

    def _reward_track_lin_vel_xy_exp(self, cmd, local_linvel) -> jax.Array:
        err = jp.sum(jp.square(cmd[:2] - local_linvel[:2]))
        return jp.exp(-err / _TRACKING_SIGMA)

    def _reward_track_ang_vel_z_exp(self, cmd, gyro) -> jax.Array:
        err = jp.square(cmd[2] - gyro[2])
        return jp.exp(-err / _TRACKING_SIGMA)

    def _reward_track_height_exp(self, height, target) -> jax.Array:
        return jp.exp(-jp.square(height - target) / _TRACKING_SIGMA)

    def _reward_track_body_roll_exp(self, upvector, target_roll) -> jax.Array:
        err = jp.square(upvector[1] - jp.sin(target_roll))
        return jp.exp(-err / _TRACKING_SIGMA)

    def _reward_track_ee_pos_exp(self, data, wb_cmd) -> jax.Array:
        R = data.xmat[self._torso_body_id].reshape(3, 3)
        torso_pos = data.xpos[self._torso_body_id]
        left_ee_torso  = R.T @ (data.site_xpos[self._left_hand_site_id]  - torso_pos)
        right_ee_torso = R.T @ (data.site_xpos[self._right_hand_site_id] - torso_pos)
        l_err = jp.sum(jp.square(left_ee_torso  - wb_cmd[:3]))
        r_err = jp.sum(jp.square(right_ee_torso - wb_cmd[3:6]))
        return jp.exp(-(l_err + r_err) / _TRACKING_SIGMA)

    # --- orientation costs ---

    def _cost_body_orientation_l2(self, data) -> jax.Array:
        """Torso projected gravity — matches Isaac Lab body_orientation_l2 w/ body_names='.*torso.*'."""
        R = data.xmat[self._torso_body_id].reshape(3, 3)
        proj_grav = R.T @ _GRAVITY
        return jp.sum(jp.square(proj_grav[:2]))

    def _cost_flat_orientation_l2(self, data) -> jax.Array:
        """Pelvis projected gravity — matches Isaac Lab flat_orientation_l2 on root body."""
        R = data.xmat[self._pelvis_body_id].reshape(3, 3)
        proj_grav = R.T @ _GRAVITY
        return jp.sum(jp.square(proj_grav[:2]))

    # --- base motion costs ---

    def _cost_lin_vel_z_l2(self, data) -> jax.Array:
        global_linvel = data.cvel[self._torso_body_id, 3:]
        return jp.square(global_linvel[2])

    def _cost_ang_vel_xy_l2(self, gyro) -> jax.Array:
        return jp.sum(jp.square(gyro[:2]))

    # --- feet ---

    def _reward_feet_air_time(
        self,
        air_time: jax.Array,
        first_contact: jax.Array,
        cmd: jax.Array,
        threshold_min: float = 0.2,
        threshold_max: float = 0.5,
    ) -> jax.Array:
        moving = jp.linalg.norm(cmd) > 0.01
        air_clipped = jp.clip(
            air_time - threshold_min, 0.0, threshold_max - threshold_min
        )
        return jp.sum(air_clipped * first_contact) * moving

    def _cost_feet_slide(self, data, contact) -> jax.Array:
        feet_vel  = data.cvel[jp.array([self._left_ankle_body_id, self._right_ankle_body_id]), 3:]  # (2, 3)
        vel_xy_sq = jp.sum(jp.square(feet_vel[..., :2]), axis=-1)  # (2,)
        return jp.sum(vel_xy_sq * contact)

    def _cost_feet_force(self, data) -> jax.Array:
        # MJX does not expose per-foot contact forces; return zero (scale -3e-3, negligible)
        return jp.zeros(())

    # --- pose / regularisation ---

    def _cost_stand_still(self, cmd, qpos) -> jax.Array:
        return jp.sum(jp.abs(qpos - self._default_pose)) * (jp.linalg.norm(cmd) < 0.01)

    def _cost_joint_deviation_hip(self, qpos, cmd) -> jax.Array:
        error = qpos[self._hip_indices] - self._default_pose[self._hip_indices]
        weight = jp.where(
            cmd[1] > 0.1,
            jp.array([1.0, 0.0, 1.0, 0.0]),
            jp.array([1.0, 1.0, 1.0, 1.0]),
        )
        return jp.sum(jp.abs(error) * weight)

    def _cost_joint_deviation_legs(self, qpos) -> jax.Array:
        error = qpos[self._leg_indices] - self._default_pose[self._leg_indices]
        return jp.sum(jp.abs(error))

    def _cost_joint_deviation_arms(self, qpos) -> jax.Array:
        error = qpos[self._arm_indices] - self._default_pose[self._arm_indices]
        return jp.sum(jp.abs(error))

    def _cost_joint_deviation_torso(self, qpos) -> jax.Array:
        return jp.abs(qpos[self._torso_jnt_idx] - self._default_pose[self._torso_jnt_idx])

    def _cost_penalize_ee_jitter(
        self,
        left_ee: jax.Array,
        right_ee: jax.Array,
        last_left_ee: jax.Array,
        last_right_ee: jax.Array,
    ) -> jax.Array:
        left_vel  = (left_ee  - last_left_ee)  / self.dt
        right_vel = (right_ee - last_right_ee) / self.dt
        return jp.linalg.norm(left_vel) + jp.linalg.norm(right_vel)

    def _cost_dof_pos_limits(self, qpos) -> jax.Array:
        out_lower = jp.clip(self._soft_lowers - qpos, 0.0)
        out_upper = jp.clip(qpos - self._soft_uppers, 0.0)
        return jp.sum(out_lower + out_upper)

    def _cost_action_rate_l2(self, act, last_act) -> jax.Array:
        return jp.sum(jp.square(act - last_act))

    def _cost_energy(self, qvel, qfrc_actuator) -> jax.Array:
        return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator[6:]))

    def _cost_dof_acc_l2(self, qacc) -> jax.Array:
        return jp.sum(jp.square(qacc))

    def _cost_torques(self, qfrc_actuator) -> jax.Array:
        return jp.sum(jp.abs(qfrc_actuator))
