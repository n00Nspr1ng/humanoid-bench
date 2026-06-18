"""H1 lowlevel locomotion + whole-body policy (Brax/MJX)."""

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from brax.envs.base import MjxEnv, State

from humanoid_bench.mjx.envs.utils import perturbed_pipeline_step

from pathlib import Path
import humanoid_bench.mjx.envs.lowlevel_cfg as _cfg

_XML_PATH = str(Path(__file__).parents[3] / "humanoid_bench/assets/mjx/scene_mjx_h1_lowlevel.xml")
_SIM_DT              = _cfg.SIM_DT
_N_FRAMES            = _cfg.N_FRAMES
_ACTION_SCALE_LEGS       = _cfg.ACTION_SCALE_LEGS
_ACTION_SCALE_ARMS_PITCH = _cfg.ACTION_SCALE_ARMS_PITCH
_ACTION_SCALE_ARMS       = _cfg.ACTION_SCALE_ARMS
_TRACKING_SIGMA_VEL      = _cfg.TRACKING_SIGMA_VEL
_TRACKING_SIGMA_HEIGHT   = _cfg.TRACKING_SIGMA_HEIGHT
_TRACKING_SIGMA_EE       = _cfg.TRACKING_SIGMA_EE
_MAX_CONTACT_FORCE       = _cfg.MAX_CONTACT_FORCE
_DEFAULT_HEIGHT          = 0.98
_LIN_VEL_X               = _cfg.LIN_VEL_X
_LIN_VEL_Y               = _cfg.LIN_VEL_Y
_ANG_VEL_YAW             = _cfg.ANG_VEL_YAW
REWARD_SCALES            = _cfg.LOCO_REWARD_SCALES


class H1LowLevelLoco(MjxEnv):
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

        # Sensor address lookup helper
        def _adr(name):
            sid = mj_model.sensor(name).id
            a   = int(mj_model.sensor_adr[sid])
            d   = int(mj_model.sensor_dim[sid])
            return a, d

        # Foot touch sensors (scalar each)
        self._lfoot_touch_adr, _ = _adr("left_foot_sensor")
        self._rfoot_touch_adr, _ = _adr("right_foot_sensor")

        # IMU sensors (3D each)
        a, d = _adr("gyro");               self._gyro_adr         = jp.arange(a, a + d)
        a, d = _adr("local_linvel_torso"); self._local_linvel_adr  = jp.arange(a, a + d)
        a, d = _adr("upvector_torso");     self._upvector_adr      = jp.arange(a, a + d)
        a, d = _adr("global_linvel_torso");self._global_linvel_adr = jp.arange(a, a + d)
        a, d = _adr("global_angvel_torso");self._global_angvel_adr = jp.arange(a, a + d)

        # Foot linvel sensors — shape (2, 3) index array for vectorised gather
        foot_linvel_adr = []
        for name in ["left_foot_global_linvel", "right_foot_global_linvel"]:
            a, d = _adr(name)
            foot_linvel_adr.append(list(range(a, a + d)))
        self._foot_linvel_adr = jp.array(foot_linvel_adr)  # (2, 3)

        # Foot force z-component addresses
        a, _ = _adr("left_foot_force");  self._lfoot_fz_adr = a + 2
        a, _ = _adr("right_foot_force"); self._rfoot_fz_adr = a + 2

        # Default EE positions relative to robot base XY (for wb_cmd targets)
        mj_data = mujoco.MjData(mj_model)
        mj_data.qpos[:] = home_kf.qpos
        mujoco.mj_forward(mj_model, mj_data)
        base_xy = mj_data.qpos[:2].copy()
        self._default_left_ee = jp.array(
            mj_data.site_xpos[self._left_hand_site_id]
            - np.array([base_xy[0], base_xy[1], 0.0])
        )
        self._default_right_ee = jp.array(
            mj_data.site_xpos[self._right_hand_site_id]
            - np.array([base_xy[0], base_xy[1], 0.0])
        )

        # wb_cmd is 10D: [left_ee(3), right_ee(3), roll, pitch, yaw, height]
        self.actor_obs_dim = 3 + 3 + 3 + 10 + 19 + 19 + 19  # 76
        # critic obs appends local_linvel(3) + feet_contact(2)
        self.state_dim  = self.actor_obs_dim + 3 + 2         # 81
        self.action_dim = self.sys.nu                        # 19

    # ------------------------------------------------------------------ #
    #  Command sampling                                                    #
    # ------------------------------------------------------------------ #

    def sample_command(self, rng: jax.Array) -> jax.Array:
        """Sample a 3D locomotion command [vx, vy, wz]; 10% chance of zeros."""
        rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
        vx  = jax.random.uniform(rng1, minval=_LIN_VEL_X[0],   maxval=_LIN_VEL_X[1])
        vy  = jax.random.uniform(rng2, minval=_LIN_VEL_Y[0],   maxval=_LIN_VEL_Y[1])
        wz  = jax.random.uniform(rng3, minval=_ANG_VEL_YAW[0], maxval=_ANG_VEL_YAW[1])
        cmd = jp.array([vx, vy, wz])
        return jp.where(jax.random.bernoulli(rng4, p=0.1), jp.zeros(3), cmd)

    def sample_wb_cmd(self, rng: jax.Array) -> jax.Array:
        """Sample a 10D whole-body command [left_ee(3), right_ee(3), roll, pitch, yaw, height].

        During standalone training the EE targets are the default arm positions
        and the height/orientation targets are at rest values.
        """
        del rng
        return jp.concatenate([
            self._default_left_ee,
            self._default_right_ee,
            jp.zeros(3),                    # roll=0, pitch=0, yaw=0
            jp.array([_DEFAULT_HEIGHT]),
        ])

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

        # Foot contact via touch sensors (> 0.5 N = in contact)
        sd = data.data.sensordata
        contact = jp.array([
            sd[self._lfoot_touch_adr] > 0.5,
            sd[self._rfoot_touch_adr] > 0.5,
        ])
        contact_filt  = contact | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
        feet_air_time = (state.info["feet_air_time"] + self.dt) * ~contact

        obs = self._get_obs(data.data, state.info)

        # Termination conditions
        upvector = sd[self._upvector_adr]
        torso_z  = data.data.xpos[self._torso_body_id, 2]
        lfoot_y  = data.data.site_xpos[self._left_foot_site_id, 1]
        rfoot_y  = data.data.site_xpos[self._right_foot_site_id, 1]
        done = (
            (upvector[2] < 0.0)           # fallen over
            | (torso_z < 0.92)            # torso too low
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
    #  Observation                                                         #
    # ------------------------------------------------------------------ #

    def _get_obs(self, data, info: dict) -> jax.Array:
        """Returns 81D critic obs = actor_obs(76) + local_linvel(3) + feet_contact(2).

        The actor obs (first 76 dims) matches the G1 lowlevel actor observation exactly.
        The extra 5 dims are privileged information used only by the critic.
        """
        sd = data.sensordata
        # --- actor obs (76D) ---
        actor_obs = jp.concatenate([
            sd[self._gyro_adr],                 #  3 – body-frame angular velocity
            sd[self._upvector_adr],             #  3 – torso z-axis in world (gravity proxy)
            info["command"],                    #  3 – locomotion command [vx, vy, wz]
            info["wb_cmd"],                     # 10 – whole-body command [l_ee(3), r_ee(3), roll, pitch, yaw, height]
            data.qpos[7:] - self._default_pose, # 19 – joint position deviation
            data.qvel[6:],                      # 19 – joint velocities
            info["last_act"],                   # 19 – last action
        ])  # 76D
        # --- privileged critic extras (5D) ---
        local_linvel  = sd[self._local_linvel_adr]          # 3 – body-frame linear velocity
        feet_contact  = jp.array([                           # 2 – binary foot contact
            sd[self._lfoot_touch_adr] > 0.5,
            sd[self._rfoot_touch_adr] > 0.5,
        ]).astype(jp.float32)
        return jp.concatenate([actor_obs, local_linvel, feet_contact])  # 81D

    # ------------------------------------------------------------------ #
    #  Reward computation                                                  #
    # ------------------------------------------------------------------ #

    def _get_reward(
        self, data, action, info, first_contact, contact, done
    ) -> dict:
        sd          = data.sensordata
        local_linvel = sd[self._local_linvel_adr]
        upvector    = sd[self._upvector_adr]
        gyro        = sd[self._gyro_adr]
        cmd         = info["command"]
        wb_cmd      = info["wb_cmd"]
        base_pos    = data.qpos[:3]

        return {
            "track_lin_vel_xy_exp":  self._reward_track_lin_vel_xy_exp(cmd, local_linvel),
            "track_ang_vel_z_exp":   self._reward_track_ang_vel_z_exp(cmd, gyro),
            "track_ee_pos_exp":      self._reward_track_ee_pos_exp(data, base_pos, wb_cmd),
            "track_height_exp":      self._reward_track_height_exp(base_pos[2], wb_cmd[9]),
            "track_body_roll_exp":   self._reward_track_body_roll_exp(upvector, wb_cmd[6]),
            "body_ori_l1":           self._cost_body_ori_l1(upvector),
            "flat_ori_l2":           self._cost_flat_ori_l2(upvector),
            "lin_vel_z_l2":          self._cost_lin_vel_z_l2(data),
            "ang_vel_xy_l2":         self._cost_ang_vel_xy_l2(gyro),
            "feet_air_time":         self._reward_feet_air_time(
                info["feet_air_time"], first_contact, cmd
            ),
            "feet_slide":            self._cost_feet_slide(data, contact),
            "feet_force":            self._cost_feet_force(data),
            "stand_still":            self._cost_stand_still(cmd, data.qpos[7:]),
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
            "torques":               self._cost_torques(data.qfrc_actuator),
            "termination_penalty":   done.astype(jp.float32),
        }

    # --- tracking (exp rewards) ---

    def _reward_track_lin_vel_xy_exp(self, cmd, local_linvel) -> jax.Array:
        err = jp.sum(jp.square(cmd[:2] - local_linvel[:2]))
        return jp.exp(-err / _TRACKING_SIGMA_VEL)

    def _reward_track_ang_vel_z_exp(self, cmd, gyro) -> jax.Array:
        err = jp.square(cmd[2] - gyro[2])
        return jp.exp(-err / _TRACKING_SIGMA_VEL)

    def _reward_track_height_exp(self, height, target) -> jax.Array:
        return jp.exp(-jp.square(height - target) / _TRACKING_SIGMA_HEIGHT)

    def _reward_track_body_roll_exp(self, upvector, target_roll) -> jax.Array:
        err = jp.square(upvector[1] - jp.sin(target_roll))
        return jp.exp(-err / _TRACKING_SIGMA_EE)

    def _reward_track_ee_pos_exp(self, data, base_pos, wb_cmd) -> jax.Array:
        base_xy = jp.array([base_pos[0], base_pos[1], 0.0])
        l_err = jp.sum(jp.square(
            data.site_xpos[self._left_hand_site_id]  - base_xy - wb_cmd[:3]
        ))
        r_err = jp.sum(jp.square(
            data.site_xpos[self._right_hand_site_id] - base_xy - wb_cmd[3:6]
        ))
        return jp.exp(-(l_err + r_err) / _TRACKING_SIGMA_EE)

    # --- orientation costs ---

    def _cost_body_ori_l1(self, upvector) -> jax.Array:
        return jp.sum(jp.abs(upvector[:2]))

    def _cost_flat_ori_l2(self, upvector) -> jax.Array:
        return jp.sum(jp.square(upvector[:2]))

    # --- base motion costs ---

    def _cost_lin_vel_z_l2(self, data) -> jax.Array:
        global_linvel = data.sensordata[self._global_linvel_adr]
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
        feet_vel  = data.sensordata[self._foot_linvel_adr]  # (2, 3)
        vel_xy_sq = jp.sum(jp.square(feet_vel[..., :2]), axis=-1)  # (2,)
        return jp.sum(vel_xy_sq * contact)

    def _cost_feet_force(self, data) -> jax.Array:
        lz = jp.abs(data.sensordata[self._lfoot_fz_adr])
        rz = jp.abs(data.sensordata[self._rfoot_fz_adr])
        return (
            jp.clip(lz - _MAX_CONTACT_FORCE, 0.0)
            + jp.clip(rz - _MAX_CONTACT_FORCE, 0.0)
        )

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
        return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

    def _cost_dof_acc_l2(self, qacc) -> jax.Array:
        return jp.sum(jp.square(qacc))

    def _cost_torques(self, qfrc_actuator) -> jax.Array:
        return jp.sum(jp.abs(qfrc_actuator))
