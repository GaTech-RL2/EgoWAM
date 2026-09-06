# 🤖 Simulation (RoboTwin)

The paper's robot-to-robot transfer results (supplementary) run on
[RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin), a SAPIEN bimanual benchmark. It is
third-party, and it is vendored at `external/RoboTwin`.

## 1. Install the simulation assets

The code arrives with the clone; RoboTwin's meshes, textures and `envs/curobo` do not (~200 MB,
upstream fetches them with its own scripts):

```bash
cd external/RoboTwin && bash script/_download_assets.sh    # see RoboTwin's own README
```

`external/README.md` lists what our bridge adds and which upstream files we patched. Point
`EGOWAM_ROBOTWIN_ROOT` at a different checkout if you keep one elsewhere.

## 2. Build the simulation environment

RoboTwin needs SAPIEN and mplib, which conflict with the training environment, but the bridge also
needs torch and `egowam` importable in the *same* process. So the sim environment is a superset,
built separately:

```bash
uv venv .venv-sim --python 3.11
source .venv-sim/bin/activate
uv pip install -e .                 # egowam + torch
uv pip install sapien==3.0.0b1 mplib==0.2.1
```

Point launchers at it with `EGOWAM_SIM_VENV=/path/to/.venv-sim`.

The renderer needs a working Vulkan/EGL setup. On a headless node set `DISPLAY=""` and
`PYOPENGL_PLATFORM=egl`. Nodes with a broken renderer fail inside RoboTwin with `Render Error`;
The eval launchers detect this and exit non-zero so the job requeues elsewhere.

## 3. Convert data

```bash
sbatch sbatch/robotwin/convert_dataset.sh <task> <robot> <variant>
```

Writes zarr under `$EGOWAM_ROBOTWIN_ROOT`. Tasks used in the paper: `pick_diverse_bottles`,
`stack_bowls_three`, `hanging_mug`; robots `aloha-agilex` (primary), `arx-x5`, `franka`, `ur5`.

## 4. Train and evaluate

```bash
sbatch sbatch/robotwin/train_rt.sh <single|cross> <pick|stack|hang> <hpt|vae|rae>
sbatch sbatch/robotwin/eval_seeds.sh <task> <ckpt> <results.json> <tag>
```

`eval_seeds` runs the paper's protocol: 100 held-out seeds from `seed_base=100000`, resumable via
per-seed JSON so a preempted job continues rather than restarting.

## How the bridge works

All robots predict a **14-D end-effector action** — per arm `[xyz, ZYX-euler, gripper]` — in the
head-camera frame, identical across embodiments. That is what makes cross-embodiment co-training
well-posed when arms differ in degrees of freedom.

At evaluation the predicted chunk is resampled to the control rate and streamed to a differential-IK
controller. Two details matter and were established the hard way:

- **Execute at the data's native rate.** Feeding the model's upsampled 100-step chunk straight to a
  plan-to-stop controller, one plan per ~27 ms waypoint, collapses closed-loop success to near zero.
  Downsampling back to the recorded rate and streaming targets fixes it.
- **Use mink for IK**, seeded from the current joint configuration, matching the physical robot.
  curobo's IK shares a CUDA graph that cannot be reset between goal types.

## Evaluation seeds

Every method is scored on the same 100 held-out seeds, listed in
`egowam/scripts/robotwin_process/seed.txt` (whitespace-separated, RoboTwin's own seed-file
format). `sbatch/robotwin/eval_seeds.sh` passes that file to
`sim_smoke.py --eval-seed-file` by default, so a rerun evaluates exactly this set rather than
re-deriving one; point `EVAL_SEED_FILE` elsewhere to override.

The eight evaluated policies are published, so the table below can be reproduced without
retraining:

```bash
hf download boeyyyy/EgoWAM-checkpoints --include 'robotwin_pick/*' --local-dir ./ckpts
sbatch sbatch/robotwin/eval_seeds.sh pick ./ckpts/robotwin_pick/rae_cross_1999.ckpt out.json dino_c
```

Scores on this list, from `epoch_epoch=1999.ckpt` for all eight variants:

| | BC s | BC c | Pixel s | Pixel c | DINO s | DINO c | 3D Flow s | 3D Flow c |
|---|---|---|---|---|---|---|---|---|
| this list (100 seeds) | 2% | 9% | 6% | 17% | 3% | 28% | 1% | 19% |

Training seeds are `< 200`, so the evaluation set never overlaps them.

## Quick gate

The simulator, planner and success check can be exercised without a policy by running the
scripted expert:

```bash
"${EGOWAM_SIM_VENV:-.venv-sim}/bin/python" -m egowam.scripts.robotwin_process.sim_smoke \
  --task pick_diverse_bottles --task-config demo_clean --robot aloha-agilex \
  --seeds 1 --mode expert
```

If the expert succeeds, the sim stack works on that node and any later failure is the policy's.
