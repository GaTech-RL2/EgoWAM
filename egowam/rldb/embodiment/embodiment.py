from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from enum import Enum
from typing import Any


class EMBODIMENT(Enum):
    EVE_RIGHT_ARM = 0
    EVE_LEFT_ARM = 1
    EVE_BIMANUAL = 2
    ARIA_RIGHT_ARM = 3
    ARIA_LEFT_ARM = 4
    ARIA_BIMANUAL = 5
    EVA_RIGHT_ARM = 6
    EVA_LEFT_ARM = 7
    EVA_BIMANUAL = 8
    MECKA_BIMANUAL = 9
    MECKA_RIGHT_ARM = 10
    MECKA_LEFT_ARM = 11
    SCALE_BIMANUAL = 12
    SCALE_RIGHT_ARM = 13
    SCALE_LEFT_ARM = 14
    ROBOTWIN_ALOHA_BIMANUAL = 15
    ROBOTWIN_ARX_BIMANUAL = 16
    ROBOTWIN_PIPER_BIMANUAL = 17
    ROBOTWIN_UR5_BIMANUAL = 18
    ROBOTWIN_FRANKA_BIMANUAL = 19


EMBODIMENT_ID_TO_KEY = {
    member.value: key for key, member in EMBODIMENT.__members__.items()
}


def get_embodiment(index):
    return EMBODIMENT_ID_TO_KEY.get(index, None)


def get_embodiment_id(embodiment_name):
    embodiment_name = embodiment_name.upper()
    return EMBODIMENT[embodiment_name].value


def future_image_key_specs(
    base_key: str,
    zarr_key: str,
    default_offset: int,
    offsets: Sequence[int] | None = None,
) -> dict[str, dict]:
    """Build the future-image entries of a ``key_map``.

    ``offsets=None`` reproduces the historical single-frame behaviour: one key
    named ``base_key``, read ``default_offset`` frames ahead of the anchor.

    Passing a list of offsets instead emits one key per offset, suffixed
    ``_s1 … _sN`` in ascending time order
    (``future.observations.images.front_img_1_s3``).  Sparse future-video
    supervision uses this form; no config in the release does today, but the
    mechanism is kept because it is embodiment-rate agnostic.  The suffix is a *step
    index*, not a frame count, deliberately: EVA at 30 Hz ([9,18,27,36,45]) and
    Aria at 20 Hz ([6,12,18,24,30]) then produce identical key names for the same
    wall-clock horizons (0.3 … 1.5 s), so one model config serves both domains
    and only the data config carries the per-embodiment frame rate.

    Each key is read independently by ZarrDataset, so N offsets cost N JPEG
    decodes per sample.
    """
    if offsets is None:
        return {
            base_key: {
                "key_type": "camera_keys",
                "zarr_key": zarr_key,
                "future_offset": int(default_offset),
            }
        }

    offsets = [int(o) for o in offsets]
    if len(offsets) == 0:
        raise ValueError(
            f"future image offsets for {base_key!r} must be non-empty; pass None "
            "for the single-frame default."
        )
    if any(o < 0 for o in offsets):
        raise ValueError(f"future image offsets must be non-negative, got {offsets}")
    if any(b <= a for a, b in zip(offsets, offsets[1:])):
        raise ValueError(
            f"future image offsets must be strictly increasing, got {offsets}. "
            "Order defines the sequence axis of the predicted latent stack."
        )

    return {
        f"{base_key}_s{step}": {
            "key_type": "camera_keys",
            "zarr_key": zarr_key,
            "future_offset": offset,
        }
        for step, offset in enumerate(offsets, start=1)
    }


class Embodiment(ABC):
    """Base embodiment class. An embodiment is responsible for defining the transform pipeline that converts between the raw data in the dataset and the canonical representation used by the model."""

    @staticmethod
    def get_transform_list() -> list[Any]:
        """Returns the list of transforms that convert between the raw data in the dataset and the canonical representation used by the model."""
        raise NotImplementedError

    @staticmethod
    def viz_transformed_batch(batch):
        """Visualizes a batch of transformed data."""
        raise NotImplementedError

    @staticmethod
    def get_keymap():
        """Returns a dictionary mapping from the raw keys in the dataset to the canonical keys used by the model."""
        raise NotImplementedError
