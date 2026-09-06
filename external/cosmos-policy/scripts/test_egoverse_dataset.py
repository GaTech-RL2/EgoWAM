"""
Test EgoVerseDataset via a DataLoader.

Checks:
  1. Dataset loads without error.
  2. Batches contain both demo (rollout_data_mask=0) and
     rollout / world-model samples (rollout_data_mask=1, world_model_sample_mask=1).
  3. Normalised actions differ from raw actions, i.e. normalisation is active.
  4. Proprio is also normalised.

Run via:
  source /coc/flash7/bli678/Projects/EgoVerse/emimic/bin/activate
  srun -G l40s -c 8 --partition rl2-lab --account rl2-lab python test_egoverse_dataset.py
"""

import sys
import os

# Make sure the egomimic package is importable
sys.path.insert(0, "/coc/flash7/bli678/Projects/EgoVerse")
sys.path.insert(0, "/coc/flash7/bli678/Projects/EgoVerse/egomimic")

import numpy as np
import torch
from torch.utils.data import DataLoader

# ── Build dataset directly (mirrors egoverse_cosmos_policy config) ──────────
from cosmos_policy.datasets.egoverse_dataset import EgoVerseDataset

print("=" * 70)
print("Building EgoVerseDataset with demo + rollout train_datasets …")
print("=" * 70)

ds = EgoVerseDataset(
    # Shared infrastructure
    temp_root="/coc/flash7/scratch/egowm/egoverseS3Dataset",
    cache_root="/coc/flash7/scratch/egowm/.cache",
    wm_root="/coc/flash7/scratch/egowm/wmprocessedDataset",
    t5_text_embeddings_path="/coc/flash7/scratch/egowm/wmprocessedDataset/t5_embeddings.pkl",
    # Cosmos-policy rendering params
    chunk_size=25,
    use_image_aug=False,          # off for speed during testing
    use_stronger_image_aug=False,
    use_proprio=True,
    use_values=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    primary_camera_key="observations.images.front_img_1",
    left_wrist_camera_key="observations.images.left_wrist_img",
    right_wrist_camera_key="observations.images.right_wrist_img",
    proprio_key="observations.state.ee_pose",
    action_key="actions_cartesian",
    # Structured per-dataset config
    train_datasets={
        "demo": {
            "bucket_name": "rldb",
            "mode": "total",
            "embodiment": "eva_bimanual",
            "filters": {"episode_hash": "2026-01-22-18-57-54-150000"},
            "local_files_only": True,
            "use_future": True,
            "action_chunk": 25,
            "is_rollout": False,
        },
        "rollout": {
            "bucket_name": "rldb",
            "mode": "total",
            "embodiment": "eva_bimanual",
            "filters": {"episode_hash": "2026-01-22-18-57-54-150000"},
            "local_files_only": True,
            "use_future": True,
            "action_chunk": 25,
            "is_rollout": True,
            "success": True,
            "p_world_model": 0.5,
        },
    },
)

total = len(ds)
demo_len    = len(ds.demo_dataset)
rollout_len = len(ds.rollout_dataset) if ds.rollout_dataset is not None else 0

print(f"\nDataset total length : {total}")
print(f"  demo_dataset       : {demo_len}")
print(f"  rollout_dataset    : {rollout_len}")

# Print norm stats summary
print("\nNorm stats (min-max):")
for k, v in ds._norm_stats.items():
    mn_str = str([round(x, 4) for x in v['min'].tolist()])
    mx_str = str([round(x, 4) for x in v['max'].tolist()])
    print(f"  {k}:\n    min={mn_str}\n    max={mx_str}")

# ── DataLoader ─────────────────────────────────────────────────────────────────
BATCH_SIZE = 8
MAX_BATCHES = 10          # cap at 20 batches to keep the test fast
loader = DataLoader(
    ds,
    batch_size=BATCH_SIZE,
    shuffle=True,          # shuffle → samples from both demo and rollout halves
    num_workers=0,
    drop_last=False,
)

print(f"\nDataLoader: batch_size={BATCH_SIZE}, shuffle=True")
print(f"  Iterating up to {MAX_BATCHES} batches …\n")

demo_count     = 0
rollout_count  = 0
wm_count       = 0
vf_count       = 0
success_count  = 0

raw_action_means  = []
norm_action_means = []
raw_proprio_means = []
norm_proprio_means = []

for batch_idx, batch in enumerate(loader):
    if batch_idx >= MAX_BATCHES:
        break

    rdm  = batch["rollout_data_mask"]          # (B,)
    rdsm = batch["rollout_data_success_mask"]
    wmsm = batch["world_model_sample_mask"]
    vfsm = batch["value_function_sample_mask"]

    demo_count    += int((rdm == 0).sum())
    rollout_count += int((rdm == 1).sum())
    success_count += int((rdsm == 1).sum())
    wm_count      += int((wmsm == 1).sum())
    vf_count      += int((vfsm == 1).sum())

    raw_action_means.append(batch["actions_raw"].mean().item())
    norm_action_means.append(batch["actions"].mean().item())
    raw_proprio_means.append(batch["proprio_raw"].mean().item())
    norm_proprio_means.append(batch["proprio"].mean().item())

    print(
        f"  [batch {batch_idx:2d}]  "
        f"demo={int((rdm==0).sum())}  rollout={int((rdm==1).sum())}  "
        f"wm={int((wmsm==1).sum())}  vf={int((vfsm==1).sum())}  "
        f"action_raw_mean={batch['actions_raw'].mean():.4f}  "
        f"action_norm_mean={batch['actions'].mean():.4f}"
    )

batches_seen = batch_idx + 1

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Batches iterated       : {batches_seen}  (batch_size={BATCH_SIZE})")
print(f"  Total samples checked  : {demo_count + rollout_count}")
print(f"  Demo  (rollout_data_mask=0) : {demo_count}")
print(f"  Rollout (rollout_data_mask=1): {rollout_count}")
print(f"    ↳ success rollouts         : {success_count}")
print(f"    ↳ world-model samples      : {wm_count}")
print(f"    ↳ value-function samples   : {vf_count}")
print()
print(f"  Action  raw    mean (avg over {batches_seen} batches): {np.mean(raw_action_means):.4f}")
print(f"  Action  normed mean (avg over {batches_seen} batches): {np.mean(norm_action_means):.4f}")
print(f"  Proprio raw    mean (avg over {batches_seen} batches): {np.mean(raw_proprio_means):.4f}")
print(f"  Proprio normed mean (avg over {batches_seen} batches): {np.mean(norm_proprio_means):.4f}")

# ── Assertions ────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("ASSERTIONS")
print("=" * 70)

assert demo_count > 0, \
    "ERROR: No demo samples found (rollout_data_mask should be 0 for some)!"
print("  ✓  Demo samples present")

assert rollout_count > 0, \
    "ERROR: No rollout samples found! Check rollout sub-dataset is non-empty and " \
    "shuffle=True is propagating rollout indices."
print("  ✓  Rollout samples present")

assert wm_count > 0 or vf_count > 0, \
    "ERROR: No world-model or value-function samples found (check p_world_model > 0)!"
print(f"  ✓  World-model samples present     : {wm_count}")
print(f"  ✓  Value-function samples present  : {vf_count}")

# Normalization: normalised mean should differ from raw mean
raw_a   = np.mean(raw_action_means)
norm_a  = np.mean(norm_action_means)
assert abs(raw_a - norm_a) > 1e-4 or abs(raw_a) < 1e-6, \
    f"ERROR: Actions appear un-normalised — raw={raw_a:.4f}, norm={norm_a:.4f}"
print(f"  ✓  Action normalisation active  (raw={raw_a:.4f} → norm={norm_a:.4f})")

raw_p  = np.mean(raw_proprio_means)
norm_p = np.mean(norm_proprio_means)
assert abs(raw_p - norm_p) > 1e-4 or abs(raw_p) < 1e-6, \
    f"ERROR: Proprio appears un-normalised — raw={raw_p:.4f}, norm={norm_p:.4f}"
print(f"  ✓  Proprio normalisation active (raw={raw_p:.4f} → norm={norm_p:.4f})")

print("\nAll assertions passed ✓")
