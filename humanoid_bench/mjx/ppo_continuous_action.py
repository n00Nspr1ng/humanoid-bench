import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple
from flax.training.train_state import TrainState
import distrax
from wrappers import (
    LogWrapper,
    BraxGymnaxWrapper,
    VecEnv,
    NormalizeVecObservation,
    NormalizeVecReward,
    ClipAction,
)
from brax import envs
from humanoid_bench.mjx.envs.reach_continual import HumanoidReachContinual
from humanoid_bench.mjx.envs.reach_continual_two_hands import HumanoidReachContinualTwoHands
from humanoid_bench.mjx.envs.lowlevel_loco import H1LowLevelLoco
from humanoid_bench.mjx.envs.lowlevel_stand import H1LowLevelStand

from flax_to_torch import flax_to_torch, TorchModel, TorchPolicy

import os
import wandb

from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('job_name', None, 'Name of the job to be launched', required=True)
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'h1_lowlevel', 'Environment name to train.')
flags.DEFINE_integer('num_envs', 2048, 'Number of parallel environments.')

envs.register_environment('h1_reach_continual', HumanoidReachContinual)
envs.register_environment('h1_reach_continual_two_hands', HumanoidReachContinualTwoHands)
envs.register_environment('h1_lowlevel_loco', H1LowLevelLoco)
envs.register_environment('h1_lowlevel_stand', H1LowLevelStand)

class ActorCritic(nn.Module):
    action_dim: int
    actor_obs_dim: int   # first N dims of obs go to the actor; full obs goes to the critic
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        activation = nn.tanh if self.activation != "relu" else nn.relu

        # Actor head — sees only the actor obs slice
        actor_obs = x[..., :self.actor_obs_dim]
        actor_mean = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_obs)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        # Critic head — sees full obs (privileged)
        critic = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic = activation(critic)
        critic = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return pi, jnp.squeeze(critic, axis=-1)

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

def make_train(config):
    config["NUM_UPDATES"] = (
            config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
            config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    env, env_params = BraxGymnaxWrapper(config["ENV_NAME"], **config["ENV_KWARGS"]), None
    env = LogWrapper(env)
    env = ClipAction(env)
    env = VecEnv(env)
    if config["NORMALIZE_ENV"]:
        env = NormalizeVecObservation(env)
        env = NormalizeVecReward(env, config["GAMMA"])

    def linear_schedule(count):
        frac = (
                1.0
                - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
                / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    # Build network once at make_train time so _update_step can close over it
    network = ActorCritic(
        action_dim=config['DIMU'],
        actor_obs_dim=config['DIMO_ACTOR'],
        activation=config["ACTIVATION"],
    )

    def _update_step(runner_state, unused):
            
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = env.step(
                    rng_step, env_state, action, env_params
                )

                def filter_nan_state(state):
                    return jnp.where(jnp.isnan(state), jnp.zeros_like(state), state)

                obsv = jax.tree_map(filter_nan_state, obsv)
                env_state = jax.tree_map(filter_nan_state, env_state)

                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng)

                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                            delta
                            + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                                value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                                0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                                jnp.clip(
                                    ratio,
                                    1.0 - config["CLIP_EPS"],
                                    1.0 + config["CLIP_EPS"],
                                )
                                * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                                loss_actor
                                + config["VF_COEF"] * value_loss
                                - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                        batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]
            if config.get("DEBUG"):
                def callback(info, env_state, train_state):
                    return_values = info["returned_episode_returns"][info["returned_episode"]]
                    length_values = info["returned_episode_lengths"][info["returned_episode"]]
                    timesteps = info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    for t in range(min(len(timesteps), 5)):
                        print(
                            f"global step={timesteps[t]}, episodic return={return_values[t]}, "
                            f"episodic length={length_values[t]}")

                    if len(timesteps) > 0:
                        step = int(timesteps[-1])
                        wandb.log({
                            "train/episode_return": float(return_values.mean()),
                            "train/episode_length": float(length_values.mean()),
                        }, step=step)

                        # Reach-task-specific metrics
                        if config["ENV_NAME"].startswith("h1_reach"):
                            state_info = env_state.env_state.env_state.env_state.info
                            num_samples = 500
                            wandb.log({
                                "train/target_dist_left":  wandb.Histogram(np.array(state_info['target_dist_left'][:num_samples])),
                                "train/target_dist_right": wandb.Histogram(np.array(state_info['target_dist_right'][:num_samples])),
                            }, step=step)
                            total_successes = state_info['total_successes'][info["returned_episode"][-1, :]]
                            if len(total_successes) > 0:
                                wandb.log({"train/total_successes": wandb.Histogram(np.array(total_successes))}, step=step)

                        if step // (config["NUM_STEPS"] * config["NUM_ENVS"]) % 100 == 0:
                            print("Saving model")
                            save_folder = config["SAVE_FOLDER"]
                            torch_model = TorchModel(config['DIMO_ACTOR'], config['DIMU'])
                            torch_model = flax_to_torch(train_state, torch_model)
                            torch_policy = TorchPolicy(torch_model)
                            torch_policy.save(os.path.join(save_folder, "torch_model_{}.pt".format(step)))
                            mean = env_state.env_state.mean
                            var = env_state.env_state.var
                            np.save(os.path.join(save_folder, "mean_{}.npy".format(step)), mean)
                            np.save(os.path.join(save_folder, "var_{}.npy".format(step)), var)


                jax.debug.callback(callback, metric, env_state, train_state)

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

    def init(rng):
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(config['DIMO'])
        network_params = network.init(_rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = env.reset(reset_rng, env_params)
        rng, _rng = jax.random.split(rng)
        return (train_state, env_state, obsv, _rng)

    def update(runner_state, num_steps):
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, num_steps
        )
        return runner_state, metric

    return init, update

def main(_):
    exp_name = FLAGS.job_name
    save_folder = os.path.join('./data', exp_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    else:
        os.system(f"rm -rf {save_folder}/*")

    env_name = FLAGS.env_name
    if env_name == 'h1_reach_continual':
        dimU = 19
        dimO = 55
        dimO_actor = 55   # symmetric: no privileged critic obs
        env_kwargs = {
            'collisions': 'feet', 'act_control': 'pos', 'hands': 'both',
            'reward_weights_dict': {'alive': 1.0, 'vel': 1.0,
                                    'l1_weight': 1.0, 'l1_dist': 0.05,
                                    'l2_weight': 2.0, 'l2_dist': 0.3},
        }
    elif env_name == 'h1_reach_continual_two_hands':
        dimU = 19
        dimO = 61
        dimO_actor = 61   # symmetric: no privileged critic obs
        env_kwargs = {
            'collisions': 'feet', 'act_control': 'pos', 'hands': 'both',
            'reward_weights_dict': {'alive': 1.0, 'vel': 1.0,
                                    'l1_weight': 1.0, 'l1_dist': 0.05,
                                    'l2_weight': 2.0, 'l2_dist': 0.3},
        }
    elif env_name == 'h1_lowlevel_loco':
        dimU = 19
        dimO = 81         # critic obs: actor_obs(76) + local_linvel(3) + feet_contact(2)
        dimO_actor = 76   # actor obs: gyro(3)+upvec(3)+cmd(3)+wb_cmd(10)+qpos(19)+qvel(19)+act(19)
        env_kwargs = {}
    elif env_name == 'h1_lowlevel_stand':
        dimU = 19
        dimO = 75         # critic obs: actor_obs(70) + local_linvel(3) + feet_contact(2)
        dimO_actor = 70   # actor obs: gyro(3)+upvec(3)+wb_cmd(7)+qpos(19)+qvel(19)+act(19)
        env_kwargs = {}
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    config = {
        'DIMU': dimU,
        'DIMO': dimO,
        'DIMO_ACTOR': dimO_actor,
        "SAVE_FOLDER": save_folder,
        "LR": 3e-4,
        "NUM_ENVS": FLAGS.num_envs,
        "NUM_STEPS": 16,
        "TOTAL_TIMESTEPS": 4e9,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 32,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.001,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ENV_NAME": env_name,
        "ENV_KWARGS": env_kwargs,
        "ANNEAL_LR": True,
        "NORMALIZE_ENV": True,
        "DEBUG": True,
    }
    wandb.init(project="humanoid-bench", name=exp_name, config=config)

    rng = jax.random.PRNGKey(FLAGS.seed)
    init_fn, update_fn = make_train(config)
    init_jit   = jax.jit(init_fn)
    update_jit = jax.jit(update_fn, static_argnums=(1,))

    runner_state = init_jit(rng)

    # For loop is in Python so each scan compiles independently (1x memory, not split_scan_n x)
    split_scan_n = int(config["TOTAL_TIMESTEPS"] // 5e8 + 1)
    steps_per_split = config["NUM_UPDATES"] // split_scan_n
    metric = None
    for i in range(split_scan_n):
        runner_state, metric = update_jit(runner_state, steps_per_split)
        print(f"Split {i+1}/{split_scan_n} done", flush=True)

    train_state, env_state = runner_state[0], runner_state[1]

    print("mean: ", env_state.env_state.mean.shape, flush=True)
    print("var: ", env_state.env_state.var.shape, flush=True)

    # Save model — torch policy uses actor obs only
    torch_model = TorchModel(dimO_actor, dimU)
    torch_model = flax_to_torch(train_state, torch_model)
    torch_policy = TorchPolicy(torch_model)
    torch_policy.save(os.path.join(save_folder, "torch_model.pt"))
    # Save mean and var
    mean = env_state.env_state.mean
    var = env_state.env_state.var
    np.save(os.path.join(save_folder, "mean.npy"), mean)
    np.save(os.path.join(save_folder, "var.npy"), var)

if __name__ == "__main__":
    app.run(main)
    