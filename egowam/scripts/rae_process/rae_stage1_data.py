"""Build RAE stage-1 image datasets from an EgoWAM data config.

``external/RAE/src/train_stage1_egoverse.py`` imports
``load_egoverse_zarr_images_from_config`` from ``dataset.egoverse_zarr_dset``, but that
module only defines ``load_egoverse_zarr_images_train_val`` -- the caller was moved to a
config-driven API and the loader never followed. This supplies the missing function
without editing the vendored tree, so ``external/RAE`` stays byte-faithful to its base
commit for provenance.

The config it consumes is an EgoWAM data config: ``train_datasets`` / ``valid_datasets``
mapping an embodiment name to a block with ``resolver.folder_path``, ``resolver.key_map``,
``resolver.transform_list`` and ``filters``. Multiple embodiments are concatenated, which
is what makes an EVA+Aria co-trained decoder possible.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from torch.utils.data import ConcatDataset

_SPLIT_TO_KEY = {"train": "train_datasets", "valid": "valid_datasets"}


def _instantiate(spec):
    """Instantiate a `_target_` block, or return it unchanged if it is a plain value."""
    if isinstance(spec, dict) and "_target_" in spec:
        import hydra

        return hydra.utils.instantiate(spec, _convert_="all")
    return spec


def _build_one(
    entry: dict, *, split: str, transform, pixel_key, future_pixel_key, return_future
):
    from dataset.egoverse_zarr_dset import EgoVerseImageOnlyZarrDataset

    resolver = dict(entry.get("resolver") or {})
    folder_path = resolver.get("folder_path")
    if not folder_path:
        raise ValueError(f"dataset entry has no resolver.folder_path: {entry!r}")

    kwargs = dict(
        folder_path=folder_path,
        transform=transform,
        mode=split,
        filters=entry.get("filters") or {},
        pixel_key=pixel_key,
        future_pixel_key=future_pixel_key,
        return_future=return_future,
    )
    # The vendored class defaults key_map/transform_list to EVA's. That is wrong for an
    # Aria block, so pass the config's own when it names them.
    for name in ("key_map", "transform_list"):
        if name in resolver:
            kwargs[name] = _instantiate(resolver[name])
    if "valid_ratio" in entry:
        kwargs["valid_ratio"] = float(entry["valid_ratio"])
    return EgoVerseImageOnlyZarrDataset(**kwargs)


def load_egoverse_zarr_images_from_config(
    data_cfg: Dict,
    transform: Optional[Callable] = None,
    mode: str = "train",  # accepted for call-site compatibility; both splits are built
    return_future: bool = False,
    pixel_key: str = "observations.images.front_img_1",
    future_pixel_key: str = "future.observations.images.front_img_1",
) -> Dict[str, object]:
    """Return ``{"train": Dataset, "valid": Dataset}`` built from an EgoWAM data config."""
    out: Dict[str, object] = {}
    for split, cfg_key in _SPLIT_TO_KEY.items():
        parts = [
            _build_one(
                entry,
                split=split,
                transform=transform,
                pixel_key=pixel_key,
                future_pixel_key=future_pixel_key,
                return_future=return_future,
            )
            for entry in (data_cfg.get(cfg_key) or {}).values()
        ]
        # An empty list is returned rather than raising: the caller checks for it and
        # reports which split was empty, which is a clearer error than one from here.
        out[split] = (
            parts[0] if len(parts) == 1 else (ConcatDataset(parts) if parts else [])
        )
    return out
