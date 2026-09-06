"""Backwards compatibility for artifacts produced before the ``egomimic`` -> ``egowam`` rename.

Checkpoints trained before the rename embed the old module name in two places:

* their ``.hydra/config.yaml`` carries ``_target_`` strings like
  ``egomimic.pl_utils.pl_model.ModelWrapper`` (42 of them in the RAE flagship run);
* the ``.ckpt`` itself pickles references to ``egomimic.*`` classes.

Without the alias below, loading one raises ``ModuleNotFoundError: No module named 'egomimic'``.

Call :func:`install_legacy_module_alias` before unpickling a checkpoint, and run any Hydra config
read off disk through :func:`rewrite_legacy_targets`. Both are idempotent.

The alias is installed as a meta-path finder rather than by pre-importing every submodule: the
eager form costs a full import of the package tree (torch, lightning, transformers) before the
first lookup, while this resolves each ``egomimic.x.y`` to ``egowam.x.y`` only when something
actually asks for it.
"""

from __future__ import annotations

import importlib
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec

LEGACY_NAME = "egomimic"
CURRENT_NAME = "egowam"

# Modules renamed WITHIN the package, beyond the top-level prefix swap. Rewriting only the prefix
# would yield `egowam.utils.egomimicUtils`, which does not exist -- the file itself was renamed.
MODULE_RENAMES = {
    "egomimicUtils": "egowamUtils",
}


def _current_name(fullname: str) -> str:
    name = CURRENT_NAME + fullname[len(LEGACY_NAME) :]
    for old, new in MODULE_RENAMES.items():
        name = name.replace(f".{old}", f".{new}")
    return name


class _LegacyAliasLoader(Loader):
    def create_module(self, spec: ModuleSpec):
        module = importlib.import_module(_current_name(spec.name))
        sys.modules[spec.name] = module
        return module

    def exec_module(self, module) -> None:  # already executed under its real name
        return None


class _LegacyAliasFinder(MetaPathFinder):
    """Resolves ``egomimic`` and ``egomimic.*`` onto the renamed package."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != LEGACY_NAME and not fullname.startswith(LEGACY_NAME + "."):
            return None
        return ModuleSpec(fullname, _LegacyAliasLoader())


def install_legacy_module_alias() -> None:
    """Make ``import egomimic.<x>`` resolve to ``egowam.<x>`` for the rest of the process."""
    if any(isinstance(f, _LegacyAliasFinder) for f in sys.meta_path):
        return
    sys.meta_path.insert(0, _LegacyAliasFinder())


def rewrite_legacy_targets(obj):
    """Rewrite ``egomimic.`` -> ``egowam.`` throughout a config tree, preserving its shape.

    Handles OmegaConf containers, plain dicts/lists and bare strings; anything else is returned
    untouched.
    """
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf
    except ImportError:
        OmegaConf = None
        DictConfig = ListConfig = ()

    if OmegaConf is not None and isinstance(obj, (DictConfig, ListConfig)):
        return OmegaConf.create(
            rewrite_legacy_targets(OmegaConf.to_container(obj, resolve=False))
        )
    if isinstance(obj, dict):
        return {k: rewrite_legacy_targets(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rewrite_legacy_targets(v) for v in obj]
    if isinstance(obj, str) and (
        f"{LEGACY_NAME}." in obj or any(m in obj for m in MODULE_RENAMES)
    ):
        out = obj.replace(f"{LEGACY_NAME}.", f"{CURRENT_NAME}.")
        for old, new in MODULE_RENAMES.items():
            out = out.replace(f".{old}", f".{new}")
        return out
    return obj
