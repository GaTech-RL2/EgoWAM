"""Hydra helpers for Lightning FSDP strategy configs."""

from __future__ import annotations

from functools import partial
from importlib import import_module
from typing import Iterable, Optional, Type

import torch.nn as nn
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

DEFAULT_WAN_TRANSFORMER_BLOCKS = (
    "diffusers.models.transformers.transformer_wan.WanTransformerBlock",
    "diffusers.models.transformers.transformer_wan_vace.WanVACETransformerBlock",
)


def _resolve_module_class(class_path: str) -> Type[nn.Module]:
    module_name, class_name = class_path.rsplit(".", 1)
    module = import_module(module_name)
    cls = getattr(module, class_name)
    if not isinstance(cls, type) or not issubclass(cls, nn.Module):
        raise TypeError(f"{class_path} is not a torch.nn.Module class")
    return cls


def resolve_module_classes(
    class_paths: Optional[Iterable[str]] = None,
) -> list[Type[nn.Module]]:
    """Resolve import strings to module classes for Lightning FSDP options."""
    return [
        _resolve_module_class(class_path)
        for class_path in (class_paths or DEFAULT_WAN_TRANSFORMER_BLOCKS)
    ]


def build_transformer_auto_wrap_policy(
    class_paths: Optional[Iterable[str]] = None,
):
    """Return an FSDP auto-wrap policy for Wan transformer blocks."""
    transformer_layer_cls = set(resolve_module_classes(class_paths))
    return partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_layer_cls,
    )
