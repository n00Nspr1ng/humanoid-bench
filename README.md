# HumanoidBench — Lowlevel Policy Training for H1

This fork of HumanoidBench is used to train and evaluate a **lowlevel locomotion policy for the Unitree H1** robot, which serves as the motor primitive for a hierarchical LLM-based control system in `humanoid_coordination`.

The original HumanoidBench README is preserved at `README_ORIGINAL.md`.

---

## Why HumanoidBench

The main `humanoid_coordination` codebase trains lowlevel policies in **Isaac Lab (PhysX, G1 robot)**. Moving to HumanoidBench for the H1 lowlevel policy avoids the sim-to-sim transfer problem: the policy is trained and evaluated in the same MuJoCo physics backend. HumanoidBench also provides GPU-parallel training via Brax/MJX for the lowlevel, and a library of CPU MuJoCo task environments for highlevel evaluation.

---

## Overall Pipeline

```
[1] Train lowlevel policy
    Brax/MJX (GPU-parallel)
    h1_mjx_feet_collisions_pos.xml
    humanoid_bench/mjx/envs/lowlevel.py
            |
            | JAX/Flax checkpoint
            v
[2] Convert to PyTorch
    humanoid_bench/mjx/flax_to_torch.py
    -> TorchModel (256-256 MLP) + TorchPolicy
    -> checkpoints/policy.pt
            |
            | TorchPolicy loaded at runtime
            v
[3] Highlevel task evaluation
    CPU MuJoCo gymnasium env (h1_pos_*.xml, h1hand_pos_*.xml, ...)
    Wrapper holds TorchPolicy, builds lowlevel obs from live sim state,
    calls policy every step, passes 19 joint targets to task env
    (same pattern as SingleReachWrapper in wrappers.py)
            |
            v
[4] LLM-generated highlevel policy
    humanoid_coordination/llm/ (lives outside this repo)
    generates Python policy code that issues commands to the wrapper
```

---

## Robot Variants and XML Files

### `assets/robots/` — pure robot definitions (used by CPU gymnasium env)

| File | DOF | Description |
|---|---|---|
| `h1_pos.xml` | 19 | Base H1, no hands |
| `h1hand_pos.xml` | 76 | H1 + Shadow Hand dexterous fingers |
| `h1simplehand_pos.xml` | 52 | H1 + simplified hands |
| `h1touch_pos.xml` | 76 | H1 + touch sensors + Shadow Hand |
| `h1strong_pos.xml` | 76 | H1 + stronger actuators |
| `h1gripper_pos.xml` | — | H1 + gripper |

### `assets/mjx/` — MJX-specific scenes (used by Brax training)

| File | Collision geometry | Used for |
|---|---|---|
| `h1_mjx_feet_collisions_pos.xml` | Feet only (2 boxes) | Lowlevel training — fast, contacts only where needed |
| `h1_mesh_collisions_hands_pos.xml` | Full body meshes + hands | Testing with full collision fidelity |

`h1_mjx_feet_collisions_pos.xml` is the right choice for lowlevel locomotion training. Feet contacts are all that matter for walking, and simplified geometry keeps MJX simulation fast.

### `assets/envs/` — task scene XMLs (used by CPU task envs)

Named `{robot}_{control}_{task}.xml`, e.g. `h1_pos_walk.xml`, `h1hand_pos_push.xml`. Each includes the robot + task-specific objects (obstacles, boxes, stairs, etc.). These are loaded by `HumanoidEnv` at task evaluation time.

### How training XML relates to task XMLs

The lowlevel policy is trained on the simplified MJX scene. At task evaluation time, the wrapper loads whatever task env XML is needed and runs the TorchPolicy directly on the live sim state (`qpos`, `qvel`, hand positions) — the policy never sees the task XML. If the task env has hands (`nu > 19`), the wrapper fills body joints from the policy and handles hand joints separately.

---

## Control Mode

All H1 XML files use **position control** (`<position>` actuators) with PD gains:

| Joint group | kp | kv | Force limit |
|---|---|---|---|
| Hip | 200 | 5 | ±200 N |
| Knee | 300 | 6 | ±300 N |
| Ankle | 40 | 2 | ±40 N |
| Torso | 300 | 6 | ±200 N |
| Shoulder (pitch/roll) | 100 | 2 | ±40 N |
| Shoulder (yaw) / Elbow | 100 | 2 | ±18 N |

Action space for the lowlevel policy: **19 joint position targets** (body only, no hands).

---

## New Files in This Fork

```
humanoid_bench/mjx/envs/
└── lowlevel.py          # H1 locomotion env for Brax/MJX training

checkpoints/
├── flax/                # Raw JAX/Flax checkpoint from training
└── policy.pt            # Converted TorchPolicy (PyTorch)
```

The training entry point is the existing `humanoid_bench/mjx/ppo_continuous_action.py` with `h1_lowlevel` registered as a new Brax env. No separate training script is needed.

---

## Training

### Lowlevel stand policy (whole-body, standing only)

```bash
cd humanoid_bench/mjx
python ppo_continuous_action.py --job_name lowlevel_stand_v1 --env_name h1_lowlevel_stand
```

### Lowlevel loco policy (locomotion + whole-body)

```bash
cd humanoid_bench/mjx
python ppo_continuous_action.py --job_name lowlevel_loco_v1 --env_name h1_lowlevel_loco
```

Checkpoints are saved to `./data/<job_name>/` every 100 update steps.

---

## Evaluation

```bash
cd humanoid_bench/mjx

# Interactive viewer — opens a MuJoCo window and runs the policy in real time, not specifying step will evaluate the final checkpoint
python eval_lowlevel.py --folder data/lowlevel_stand_v1 --step 57344

# Record 5 rollouts to video (saved as data/lowlevel_stand_v1/evaluation.mp4)
python eval_lowlevel.py --folder data/lowlevel_stand_v1 --step 57344 --render

# Sweep through all existing checkpoints and output each checkpoint's episode length
python sweep_eval.py --folder data/lowlevel_stand_v1 --every 1

# Evaluate in gpu MuJoCo
python eval_mjx.py --folder data/lowlevel_stand_v2 --step 1998848000
```

Checkpoints are written during training as `torch_model_<step>.pt` / `mean_<step>.npy` / `var_<step>.npy`, so evaluation can run alongside training.

> **Note:** If the robot falls immediately in CPU eval despite training succeeding, try an
> earlier checkpoint. Very long training runs may cause the policy to develop large raw action
> outputs that corrupt the `last_act` observation in deployment.

---

## Installation

```bash
# 1. Install HumanoidBench env
pip install -e .
# 2. Install more requirements
pip install -r requirements_lowlevel.txt
```

`requirements_lowlevel.txt` covers the Brax/JAX training stack (brax, flax, optax, distrax, chex, gymnax, absl-py). JAX must be installed before the rest to avoid the CPU-only `jax` wheel overwriting the CUDA version.

`gymnasium`, `torch`, and `tensorboard` are expected to already be installed in your environment and are intentionally excluded from both `setup.py` and `requirements_lowlevel.txt`.

---

## Relationship to `humanoid_coordination`

The `humanoid_coordination` package (parent repo) contains:
- The LLM code generation system (`llm/`) — simulator-agnostic, stays there
- The G1/Isaac Lab lowlevel policy and training pipeline — kept as-is
- Highlevel task wrappers that will call into this repo's TorchPolicy at evaluation time

This repo is responsible only for H1 lowlevel training and the task environments the highlevel policy is evaluated on.
