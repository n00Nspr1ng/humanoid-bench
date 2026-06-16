"""Sweep over checkpoints in a folder and report episode lengths.

Usage (from humanoid_bench/mjx/):
    python sweep_eval.py --folder data/lowlevel_stand_v2
    python sweep_eval.py --folder data/lowlevel_stand_v2 --every 5 --n_rollouts 2
"""

import argparse
import os
import glob
import re
import numpy as np

from eval_lowlevel import LowlevelCPUEnv, load_policy, infer, _OBS_DIMS

def _last_act_init(mean, env_name):
    s = _OBS_DIMS[env_name]["last_act_start"]
    return mean[s : s + 19]


def true_step(stored: int) -> int:
    """Convert int32-overflowed step value back to true uint32 step."""
    if stored >= 0:
        return stored
    return stored + (1 << 32)


def run_single(env, model, mean, var, actor_dim, max_steps=1000, last_act_init=None):
    rng = np.random.default_rng(0)
    obs = env.reset(rng, last_act_init=last_act_init)
    for t in range(max_steps):
        action = infer(model, obs, mean, var, actor_dim)
        obs, done = env.step(action, rng)
        if done:
            return t + 1
    return max_steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder",      type=str, required=True)
    parser.add_argument("--env_name",    type=str, default="h1_lowlevel_stand",
                        choices=list(_OBS_DIMS.keys()))
    parser.add_argument("--every",       type=int, default=10,
                        help="evaluate every N-th checkpoint (by true step order)")
    parser.add_argument("--n_rollouts",  type=int, default=1,
                        help="rollouts per checkpoint")
    parser.add_argument("--max_steps",   type=int, default=1000)
    args = parser.parse_args()

    # Collect all per-step checkpoints (torch_model_<step>.pt)
    pattern = os.path.join(args.folder, "torch_model_*.pt")
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"No checkpoints found in {args.folder}")

    # Parse stored step from filename
    entries = []
    for p in paths:
        m = re.search(r"torch_model_(-?\d+)\.pt$", p)
        if m:
            stored = int(m.group(1))
            entries.append((true_step(stored), stored, p))

    entries.sort(key=lambda x: x[0])
    sampled = entries[::args.every]
    print(f"Total checkpoints: {len(entries)}, evaluating {len(sampled)} "
          f"(every {args.every}-th)")

    actor_dim = _OBS_DIMS[args.env_name]["actor"]
    env = LowlevelCPUEnv(args.env_name)

    results = []
    for i, (ts, stored, path) in enumerate(sampled):
        try:
            model, mean, var = load_policy(args.folder, args.env_name, stored)
        except Exception as e:
            print(f"[{i+1}/{len(sampled)}] true_step={ts:>12,}  stored={stored:>12}  LOAD ERROR: {e}")
            continue

        init = _last_act_init(mean, args.env_name)
        lengths = [run_single(env, model, mean, var, actor_dim, args.max_steps, init)
                   for _ in range(args.n_rollouts)]
        mean_len = np.mean(lengths)
        results.append((ts, stored, mean_len, lengths))
        print(f"[{i+1}/{len(sampled)}] true_step={ts:>12,}  stored={stored:>12}  "
              f"ep_len={mean_len:6.1f}  {lengths}")

    if not results:
        print("No results.")
        return

    best = max(results, key=lambda x: x[2])
    print(f"\nBest checkpoint: true_step={best[0]:,}  stored={best[1]}  "
          f"mean_ep_len={best[2]:.1f}  lengths={best[3]}")
    print(f"  → torch_model_{best[1]}.pt")


if __name__ == "__main__":
    main()
