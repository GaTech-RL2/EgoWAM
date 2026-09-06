import copy
import os
import signal
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import DictConfig, OmegaConf, open_dict
from tabulate import tabulate

from egowam.compat import install_legacy_module_alias, rewrite_legacy_targets
from egowam.rldb.zarr.utils import DataSchematic, set_global_seed
from egowam.scripts.evaluation.eval import Eval
from egowam.utils.aws.aws_data_utils import load_env
from egowam.utils.instantiators import instantiate_callbacks, instantiate_loggers
from egowam.utils.logging_utils import log_hyperparameters
from egowam.utils.pylogger import RankedLogger
from egowam.utils.utils import extras, task_wrapper

OmegaConf.register_new_resolver("eval", eval)
log = RankedLogger(__name__, rank_zero_only=True)


def _set_cuda_device_from_local_rank() -> None:
    """Pin each spawned rank to its local CUDA device before model construction."""
    if not torch.cuda.is_available():
        return

    local_rank_raw = os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID"))
    if local_rank_raw is None:
        return

    try:
        local_rank = int(local_rank_raw)
    except ValueError:
        log.warning(f"Ignoring invalid local rank value: {local_rank_raw!r}")
        return

    device_count = torch.cuda.device_count()
    if device_count <= 0:
        return

    device_index = local_rank % device_count
    torch.cuda.set_device(device_index)
    log.info(
        f"Set CUDA current device to cuda:{device_index} from local rank {local_rank}"
    )


def _load_ckpt_hydra_config(ckpt_path: str) -> Optional[DictConfig]:
    """Load Hydra config from the run directory that produced ckpt_path."""
    run_dir = os.path.dirname(os.path.dirname(ckpt_path))
    hydra_cfg_path = os.path.join(run_dir, ".hydra", "config.yaml")
    if not os.path.exists(hydra_cfg_path):
        log.warning(f"Could not find checkpoint Hydra config at: {hydra_cfg_path}")
        return None
    log.info(f"Loading checkpoint Hydra config from: {hydra_cfg_path}")
    # Checkpoints predating the egomimic -> egowam rename carry `_target_` strings (and pickled
    # class references) under the old package name; see egowam/compat.py.
    install_legacy_module_alias()
    return rewrite_legacy_targets(OmegaConf.load(hydra_cfg_path))


def _leaf_dataset_cfg_with_resolver(cfg_node: DictConfig) -> DictConfig:
    """
    Return the nested config node that defines `resolver` (Hydra target
    MultiDataset._from_resolver).

    Flat configs already have `resolver` at the top level. Nested MultiDataset
    configs use `datasets`; we descend into sorted sub-keys until we find a
    `_from_resolver` block (deterministic).
    """
    if OmegaConf.select(cfg_node, "resolver") is not None:
        return cfg_node
    datasets = OmegaConf.select(cfg_node, "datasets")
    if datasets is not None:
        for name in sorted(datasets.keys()):
            sub = datasets[name]
            try:
                return _leaf_dataset_cfg_with_resolver(sub)
            except ValueError:
                continue
    raise ValueError(
        "No resolver found in dataset config (expected MultiDataset._from_resolver "
        "or nested datasets containing one)."
    )


def _log_dataset_frame_counts(train_datasets: dict, valid_datasets: dict) -> None:
    rows = []
    for name, ds in train_datasets.items():
        rows.append(("train", name, len(ds)))
    if train_datasets:
        rows.append(
            ("TOTAL", "(train)", sum(len(ds) for ds in train_datasets.values()))
        )
    for name, ds in valid_datasets.items():
        rows.append(("valid", name, len(ds)))
    if valid_datasets:
        rows.append(
            ("TOTAL", "(valid)", sum(len(ds) for ds in valid_datasets.values()))
        )
    table = tabulate(
        rows,
        headers=["Split", "Dataset", "Frames"],
        tablefmt="rounded_outline",
        intfmt=",",
    )
    log.info("Dataset frame counts:\n" + table)


def _active_dataset_names_for_model(cfg: DictConfig) -> Optional[set[str]]:
    """Return the configured model domains as dataset names when available."""
    domains = OmegaConf.select(cfg, "model.robomimic_model.domains")
    if domains is None:
        return None
    resolved = OmegaConf.to_container(domains, resolve=True)
    if resolved is None:
        return None
    return {str(name) for name in resolved}


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    _set_cuda_device_from_local_rank()

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

        set_global_seed(cfg.seed)
    else:
        raise ValueError("Seed must be provided in cfg for reproducibility!")

    load_env()
    # log.info(f"Instantiating data schematic <{cfg.data_schematic._target_}>")

    data_schematic: DataSchematic = hydra.utils.instantiate(cfg.data_schematic)

    active_dataset_names = _active_dataset_names_for_model(cfg)
    configured_train_names = list(cfg.data.train_datasets.keys())
    configured_valid_names = list(cfg.data.valid_datasets.keys())
    if active_dataset_names is None:
        train_dataset_names = configured_train_names
        valid_dataset_names = configured_valid_names
    else:
        train_dataset_names = [
            name for name in configured_train_names if name in active_dataset_names
        ]
        valid_dataset_names = [
            name for name in configured_valid_names if name in active_dataset_names
        ]
        dropped_train = [
            name for name in configured_train_names if name not in active_dataset_names
        ]
        dropped_valid = [
            name for name in configured_valid_names if name not in active_dataset_names
        ]
        if not train_dataset_names:
            raise ValueError(
                "No training datasets remain after filtering by model domains "
                f"{sorted(active_dataset_names)}; configured train datasets were "
                f"{configured_train_names}"
            )
        if not valid_dataset_names:
            raise ValueError(
                "No validation datasets remain after filtering by model domains "
                f"{sorted(active_dataset_names)}; configured valid datasets were "
                f"{configured_valid_names}"
            )
        if dropped_train or dropped_valid:
            log.info(
                "Filtering data config to active model domains "
                f"{sorted(active_dataset_names)}; "
                f"dropped train={dropped_train or '[]'} "
                f"valid={dropped_valid or '[]'}"
            )

    # Modify dataset configs to include `data_schematic` dynamically at runtime
    train_datasets = {}
    for dataset_name in train_dataset_names:
        train_datasets[dataset_name] = hydra.utils.instantiate(
            cfg.data.train_datasets[dataset_name]
        )

    valid_datasets = {}
    for dataset_name in valid_dataset_names:
        valid_datasets[dataset_name] = hydra.utils.instantiate(
            cfg.data.valid_datasets[dataset_name]
        )

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    assert (
        "MultiDataModuleWrapper" in cfg.data._target_
    ), "cfg.data._target_ must be 'MultiDataModuleWrapper'"
    datamodule_cfg = copy.deepcopy(cfg.data)
    with open_dict(datamodule_cfg):
        datamodule_cfg.train_datasets = {
            name: datamodule_cfg.train_datasets[name] for name in train_dataset_names
        }
        datamodule_cfg.valid_datasets = {
            name: datamodule_cfg.valid_datasets[name] for name in valid_dataset_names
        }
        if OmegaConf.select(datamodule_cfg, "train_dataloader_params") is not None:
            datamodule_cfg.train_dataloader_params = {
                name: datamodule_cfg.train_dataloader_params[name]
                for name in train_dataset_names
                if name in datamodule_cfg.train_dataloader_params
            }
        if OmegaConf.select(datamodule_cfg, "valid_dataloader_params") is not None:
            datamodule_cfg.valid_dataloader_params = {
                name: datamodule_cfg.valid_dataloader_params[name]
                for name in valid_dataset_names
                if name in datamodule_cfg.valid_dataloader_params
            }
    datamodule: LightningDataModule = hydra.utils.instantiate(
        datamodule_cfg,
        train_datasets=train_datasets,
        valid_datasets=valid_datasets,
    )

    # TODO: deprecate shape inference in favor of LeRobotDatasetMetadata
    # NOTE: We assume that each dataset is of a unique embodiment. Multi-task datasets should be wrapped around TODO: MultiRLDBDataset

    # Point-flow preflight: compute/load normalization stats BEFORE dataset[0]
    # so NormalizeFlowDelta transforms don't hit an unregistered-stats error.
    # Single-user contract: on data regeneration, delete flow_stats__*.json.
    from egowam.rldb.zarr.point_flow_preflight import (
        inject_train_stats_into_val,
        run_flow_stats_preflight,
    )

    flow_cache_dir = cfg.paths.get("flow_stats_cache_dir", None) or os.path.join(
        cfg.trainer.default_root_dir, "flow_stats"
    )
    run_flow_stats_preflight(
        train_datasets=datamodule.train_datasets,
        cache_dir=flow_cache_dir,
        sample_frac=float(cfg.get("norm_stat_fraction", 1.0)),
    )
    inject_train_stats_into_val(
        train_datasets=datamodule.train_datasets,
        valid_datasets=datamodule.valid_datasets,
    )

    for dataset_name, dataset in datamodule.train_datasets.items():
        log.info(f"Inferring shapes for dataset <{dataset_name}>")
        data_schematic.infer_shapes_from_batch(dataset[0])
        # instantiate norm datasets which is same as dataset but with keymap
        # without the image keys (cameras) AND flow keys. Norm-stats only needs
        # proprio + action distributions; flow has its own dedicated preflight
        # (point_flow_preflight.ensure_flow_stats), so loading flow here just
        # forces mmap reads of point-flow arrays per sample with no payoff.
        instantiate_copy = copy.deepcopy(cfg.data.train_datasets[dataset_name])
        leaf_cfg = _leaf_dataset_cfg_with_resolver(instantiate_copy)
        keymap_cfg = leaf_cfg.resolver.key_map
        km = OmegaConf.to_container(keymap_cfg, resolve=False)  # plain dict

        _norm_drop_key_types = {"camera_keys", "flow_keys"}
        km = {
            k: v
            for k, v in km.items()
            if not (
                isinstance(v, Mapping) and v.get("key_type") in _norm_drop_key_types
            )
        }

        with open_dict(leaf_cfg.resolver):
            leaf_cfg.resolver.key_map = km
        norm_dataset = hydra.utils.instantiate(leaf_cfg)
        data_schematic.infer_norm_from_dataset(
            norm_dataset,
            dataset_name,
            sample_frac=cfg.norm_stat_fraction,
            benchmark_dir=os.path.join(
                cfg.trainer.default_root_dir, "benchmark_stats.json"
            ),
        )

    # NOTE: We also pass the data_schematic_dict into the robomimic model's instatiation now that we've initialzied the shapes and norm stats.  In theory, upon loading the PL checkpoint, it will remember this, but let's see.
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        cfg.model, robomimic_model={"data_schematic": data_schematic}
    )

    _log_dataset_frame_counts(train_datasets, valid_datasets)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    plugins = []
    if os.environ.get("SLURM_JOB_ID"):
        plugins.append(
            SLURMEnvironment(requeue_signal=[signal.SIGUSR1, signal.SIGUSR2])
        )
        print("SLURM REQUEUE ENABLED")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if (
        os.environ.get("SLURM_JOB_ID")
        and os.environ.get("SLURM_RESTART_COUNT", "0") != "0"
    ):
        last_ckpt_path = os.path.join(
            trainer.default_root_dir, "checkpoints", "last.ckpt"
        )
        if os.path.exists(last_ckpt_path):
            log.info("Detected SLURM requeue — resuming from 'last.ckpt'")
            cfg.ckpt_path = last_ckpt_path
        else:
            log.info(
                "Detected SLURM requeue but no 'last.ckpt' yet (preempted before first "
                "checkpoint) — starting fresh."
            )

    os.makedirs(os.path.join(trainer.default_root_dir, "videos"), exist_ok=True)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.get("ckpt_path"),
            weights_only=False,
        )

    if cfg.get("validate"):
        assert cfg.get("ckpt_path"), "ckpt_path must be provided for validation"
        ckpt_cfg = _load_ckpt_hydra_config(cfg.get("ckpt_path"))
        if ckpt_cfg is not None and ckpt_cfg.get("model") is not None:
            log.info("Re-instantiating model from checkpoint config for validation")
            model = hydra.utils.instantiate(
                ckpt_cfg.model, robomimic_model={"data_schematic": data_schematic}
            )
        else:
            log.info("Using current runtime model config for validation")
        log.info("Starting direct validation from checkpoint (no training loop)")
        trainer.validate(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.get("ckpt_path"),
            weights_only=False,
        )

    if cfg.get("eval"):
        eval: Eval = hydra.utils.instantiate(
            cfg.eval_class, config=cfg.model, ckpt_path=cfg.get("ckpt_path")
        )
        log.info("Starting evaluation!")
        eval.perfom_eval()

    train_metrics = trainer.callback_metrics

    # if cfg.get("test"):
    #     log.info("Starting testing!")
    #     ckpt_path = trainer.checkpoint_callback.best_model_path
    #     if ckpt_path == "":
    #         log.warning("Best ckpt not found! Using current weights for testing...")
    #         ckpt_path = None
    #     trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    #     log.info(f"Best ckpt path: {ckpt_path}")

    # test_metrics = trainer.callback_metrics

    # merge train and test metrics
    test_metrics = {}  # my stub
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(
    version_base="1.3", config_path="./hydra_configs", config_name="train_zarr.yaml"
)
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    print(OmegaConf.to_yaml(cfg))

    # cfg = OmegaConf.resolve(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # # safely retrieve metric value for hydra-based hyperparameter optimization
    # metric_value = get_metric_value(
    #     metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    # )

    # # return optimized metric
    # return metric_value


if __name__ == "__main__":
    main()
