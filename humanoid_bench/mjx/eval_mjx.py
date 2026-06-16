"""Evaluate a trained lowlevel stand policy directly in the MJX (Brax/JAX) environment.

Usage (from humanoid_bench/mjx/):
    python eval_mjx.py --folder data/lowlevel_stand_v2 --step 1998848000
    python eval_mjx.py --folder data/lowlevel_stand_v2  # latest checkpoint
    python eval_mjx.py --folder data/lowlevel_stand_v2 --zero_action  # sanity check
"""

import argparse
import os
import numpy as np
import jax
import jax.numpy as jnp

from brax import envs
from humanoid_bench.mjx.envs.lowlevel_stand import H1LowLevelStand, _WB_HEIGHT_LOW
from eval_lowlevel import load_policy, infer, _OBS_DIMS

envs.register_environment('h1_lowlevel_stand', H1LowLevelStand)


def _recompute_obs(env, state, last_act_init):
    """Recompute obs with patched last_act so step-0 inference is in-distribution."""
    new_info = dict(state.info)
    new_info["last_act"] = jnp.array(last_act_init)
    new_obs = env._get_obs(state.pipeline_state.data, new_info)
    return state.replace(obs=new_obs, info=new_info)


def _termination_reason(state, env):
    """Return string describing which termination fired."""
    data = state.pipeline_state.data
    R = data.xmat[env._torso_body_id].reshape(3, 3)
    upvec   = R[:, 2]
    torso_z = data.xpos[env._torso_body_id, 2]
    lfoot_y = data.site_xpos[env._left_foot_site_id,  1]
    rfoot_y = data.site_xpos[env._right_foot_site_id, 1]
    reasons = []
    if float(upvec[2]) < 0.0:
        reasons.append(f"fallen (upvec_z={float(upvec[2]):.3f})")
    if float(torso_z) < _WB_HEIGHT_LOW - 0.1:
        reasons.append(f"torso_z={float(torso_z):.3f} < {_WB_HEIGHT_LOW - 0.1}")
    if float(lfoot_y) < float(rfoot_y):
        reasons.append(f"legs crossed (L={float(lfoot_y):.3f} R={float(rfoot_y):.3f})")
    if np.isnan(np.array(data.qpos)).any():
        reasons.append("NaN in qpos")
    if np.isnan(np.array(data.qvel)).any():
        reasons.append("NaN in qvel")
    return ", ".join(reasons) if reasons else "unknown"


def run_mjx_rollouts(folder, env_name, step, n_rollouts=5, max_steps=1000,
                     zero_action=False, verbose=False):
    if not zero_action:
        model, mean, var = load_policy(folder, env_name, step)
        actor_dim      = _OBS_DIMS[env_name]["actor"]
        last_act_start = _OBS_DIMS[env_name]["last_act_start"]
        last_act_init  = mean[last_act_start : last_act_start + 19]
    else:
        mean = var = last_act_init = None
        actor_dim = _OBS_DIMS[env_name]["actor"]

    env      = H1LowLevelStand(physics_randomization=False, obs_noise=False)
    step_jit = jax.jit(env.step)
    reset_jit = jax.jit(env.reset)

    lengths = []
    for ep in range(n_rollouts):
        rng   = jax.random.PRNGKey(ep)
        state = reset_jit(rng)

        # Only patch last_act; wb_cmd is already randomly sampled by the env reset.
        if last_act_init is not None:
            new_info = dict(state.info)
            new_info["last_act"] = jnp.array(last_act_init)
            new_obs = env._get_obs(state.pipeline_state.data, new_info)
            state = state.replace(obs=new_obs, info=new_info)

        for t in range(max_steps):
            obs_np = np.array(state.obs)

            if zero_action:
                action_np = np.zeros(19)
            else:
                action_np = np.clip(
                    infer(model, obs_np, mean, var, actor_dim), -1.0, 1.0
                )

            if t == 0 and verbose:
                print(f"  ep {ep} step 0 action: {np.round(action_np, 3)}")

            action_jax = jnp.array(action_np)
            state = step_jit(state, action_jax)

            done = bool(state.done > 0.5)
            if done:
                if verbose or t < 50:
                    print(f"  ep {ep} done at t={t+1}: {_termination_reason(state, env)}")
                break

        ep_len = t + 1
        lengths.append(ep_len)
        if not verbose:
            print(f"  ep {ep+1}: {ep_len} steps")

    print(f"\nEpisode lengths — mean={np.mean(lengths):.1f}  "
          f"min={np.min(lengths)}  max={np.max(lengths)}")
    return lengths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder",      type=str, required=True)
    parser.add_argument("--env_name",    type=str, default="h1_lowlevel_stand")
    parser.add_argument("--step",        type=int, default=None)
    parser.add_argument("--n_rollouts",  type=int, default=5)
    parser.add_argument("--max_steps",   type=int, default=1000)
    parser.add_argument("--zero_action", action="store_true",
                        help="ignore policy, apply zero action (sanity check)")
    parser.add_argument("--verbose",     action="store_true")
    args = parser.parse_args()

    run_mjx_rollouts(args.folder, args.env_name, args.step,
                     args.n_rollouts, args.max_steps,
                     zero_action=args.zero_action, verbose=args.verbose)


if __name__ == "__main__":
    main()
