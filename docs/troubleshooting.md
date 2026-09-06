# ❓ Troubleshooting

Failure modes we actually hit, with the shortest diagnosis for each.

## `GatedRepoError: 401` — over an hour into a run

```
Error in call to target 'egowam.models.cosmos_vae.CosmosVAEWrapper':
GatedRepoError('401 Client Error ... Cannot access gated repo for url
https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World/...')
```

The Wan/Cosmos tokenizer is gated. The VAE is built *after* the data sync and normalization pass,
so this surfaces long after the job starts.

**Fix:** request access, `hf auth login`, then verify in two seconds:

```bash
python egowam/scripts/check_gated_assets.py
```

Only the **Pixel** variant needs this. If you hit it on 3D-flow or RAE, your config is wrong —
those build their world-model heads from local assets.

## `ModuleNotFoundError: No module named 'egomimic'`

The package was renamed `egomimic` → `egowam`. Checkpoints trained before the rename pickle
references to the old module.

**Fix:** the training and evaluation entrypoints install the alias automatically. If you are
loading a checkpoint from your own script:

```python
from egowam.compat import install_legacy_module_alias
install_legacy_module_alias()
```

## `ModuleNotFoundError: No module named 'cosmos_policy'` when loading a checkpoint

Older world-model checkpoints pickle a class reference into `hyper_parameters`. Loading them needs
the vendored package importable:

```python
import sys; sys.path.insert(0, "external/cosmos-policy")
```

This affects loading pre-existing checkpoints only, not training new ones.

## Traceback storms about `.nfsXXXX` files at shutdown

```
OSError: [Errno 16] Device or resource busy: '.nfs00000000...'
```

`TMPDIR` is on NFS. A deleted-but-open file gets silly-renamed to `.nfsXXXX`, and cleanup then
fails with `EBUSY`, burying the real error under pages of traceback.

**Fix:** point `TMPDIR` at node-local storage. The launchers already do this
(`${SLURM_TMPDIR:-/tmp}`); override with `TMPDIR_OVERRIDE` if your cluster differs.

## `pymp-*` directories appearing in the repo root

When `/tmp` is full, Python's `tempfile` falls back to the current working directory, and every
launcher `cd`s to the repo root. Same root cause and same fix as above. They are ignored by
`.gitignore` but indicate a full scratch disk.

## A 3D-flow run sits silently at "computing/loading stats" for ~40 minutes

```
[flow preflight] computing/loading stats for eva_bimanual.future_flow_3d_points (cache=...)
```

Not a hang. Point-flow normalization statistics are computed by scanning every segment of every
training episode, the step emits no progress output, and the result is cached per flow key in
`$FLOW_STATS_CACHE_DIR` (default `.cache/flow_stats/`). Subsequent runs load it in milliseconds.

The cache is not distributed with the repository, so the first 3D-flow run on a new machine pays
this cost even though it is instant for anyone with a warm cache. Point `FLOW_STATS_CACHE_DIR` at
a shared location to pay it once across runs:

```bash
export FLOW_STATS_CACHE_DIR=/shared/egowam/flow_stats
```

Note the launchers default this to a per-run directory, so overriding it is what lets the cache
actually be reused.

## No epoch logged for the first hour

Not a hang. Normalization statistics are computed over the whole dataset before epoch 0 and are
I/O-bound — tens of minutes on the flagship splits, longer if the data still has to sync from
object storage.

Confirm progress by watching the log rather than the checkpoint directory:

```bash
tail -f sbatch_logs/<job>_<id>.out
```

`[NormStats] sampling N/N` means it is still in the statistics pass.

## No checkpoints appear until very late

Checkpointing is gated by `check_val_every_n_epoch`, so the first save happens at the first
validation epoch. With the default of 200 that is epoch 199, several hours in. Reduce it for a
quick sanity run:

```bash
sbatch sbatch/hpt_wm_3dflow.sh cup_saucer indomain trainer=debug
```

## Hydra rejects a config group starting with a digit

```
LexerNoViableAltException
```

Hydra's override grammar will not accept `3dflow=point3d`. The group is named `flow3d` for exactly
this reason:

```bash
python egowam/trainHydra.py ... flow3d=point3d
```

## `[point_flow_panel] skip future_flow_2d_points: key not in batch`

Benign. The 2D-flow visualization panel is skipped because the 3D-flow configs publish only
`future_flow_3d_points`. Training is unaffected.

## Escaping the `=` in a checkpoint path

Hydra parses `=` in override values. Checkpoint filenames contain one:

```bash
ckpt_path="<run>/checkpoints/epoch_epoch\=1399.ckpt"
```

## A human co-training config resolves to zero episodes

The SQL episode table dropped `robot_name` and gained `rig_name`. Legacy filters are translated by
`remap_legacy_filters`; if you write filters by hand, `human_bimanual` alone matches other capture
rigs too. See [data.md](data.md#a-note-on-sql-filters).
