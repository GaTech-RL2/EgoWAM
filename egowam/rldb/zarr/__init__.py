"""
Zarr-based dataset implementations for EgoVerse.
"""

from importlib import import_module

__all__ = [
    "EpisodeResolver",
    "MultiDataset",
    "ZarrDataset",
    "ZarrEpisode",
    "ZarrWriter",
    "LocalEpisodeResolver",
    "S3EpisodeResolver",
]


def __getattr__(name):
    if name == "ZarrWriter":
        return getattr(import_module("egowam.rldb.zarr.zarr_writer"), name)

    if name in {
        "EpisodeResolver",
        "LocalEpisodeResolver",
        "MultiDataset",
        "S3EpisodeResolver",
        "ZarrDataset",
        "ZarrEpisode",
    }:
        return getattr(import_module("egowam.rldb.zarr.zarr_dataset_multi"), name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
