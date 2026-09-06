"""Locating the third-party code under ``external/`` and failing usefully when it is not there.

``external/`` is vendored, so in a normal clone every dependency below is already present and
:meth:`ExternalDependency.require` is a no-op. It still exists for the cases where that is not true:
an incomplete clone, or a checkout redirected elsewhere with an environment variable. Describing
each dependency once here means a missing checkout produces the command that fixes it instead of a
bare ``ImportError`` from deep inside a stem.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class ExternalDependency:
    """One third-party checkout under external/, and what it is needed for."""

    def __init__(
        self,
        name: str,
        purpose: str,
        needed_by: str,
        env_var: str | None = None,
        marker: str = "",
        fallback: str = "",
        upstream: str = "",
    ) -> None:
        self.name = name
        self.purpose = purpose
        self.needed_by = needed_by
        self.env_var = env_var
        self.marker = marker
        self.fallback = fallback
        self.upstream = upstream

    @property
    def path(self) -> Path:
        if self.env_var:
            override = os.environ.get(self.env_var)
            if override:
                return Path(override)
        return REPO_ROOT / "external" / self.name

    def is_present(self) -> bool:
        root = self.path
        if not root.is_dir():
            return False
        # A directory that exists but is empty is a stale mount or a half-finished clone.
        if not any(root.iterdir()):
            return False
        return (root / self.marker).exists() if self.marker else True

    def install_hint(self) -> str:
        lines = [
            f"'{self.name}' is required for {self.needed_by} but was not found at {self.path}.",
            "",
            f"  What it is: {self.purpose}",
            f"  Upstream:   {self.upstream}",
        ]
        vendored_path = REPO_ROOT / "external" / self.name
        if self.env_var and os.environ.get(self.env_var):
            lines += [
                f"  {self.env_var} points there but it holds no checkout. Unset it to fall back",
                f"  on the vendored copy at {vendored_path}.",
            ]
        else:
            lines += [
                "  It is vendored in this repository, so an incomplete clone is the usual cause:",
                f"      git checkout -- external/{self.name}",
            ]
            if self.env_var:
                lines.append(f"  Or point {self.env_var} at an existing checkout.")
        if self.fallback:
            lines += ["", f"  {self.fallback}"]
        return "\n".join(lines)

    def require(self) -> Path:
        """Return the checkout path, or raise with an actionable message."""
        if not self.is_present():
            raise FileNotFoundError(self.install_hint())
        return self.path


COSMOS_POLICY = ExternalDependency(
    name="cosmos-policy",
    purpose="Wan 2.1 video VAE, used as the frozen target encoder for the Pixel world model",
    needed_by="the Pixel variant",
    marker="cosmos_policy",
    upstream="https://github.com/NVlabs/cosmos-policy.git",
    fallback="Not needed for BC, DINO or 3D-flow. The Wan tokenizer weights are a separate, "
    "gated download; see external/README.md.",
)

RAE = ExternalDependency(
    name="RAE",
    purpose="Representation Autoencoder: the frozen DINOv2 encoder and the DiT-DH denoiser",
    needed_by="the DINO world-model variant",
    env_var="EGOWAM_RAE_ROOT",
    marker="src",
    upstream="https://github.com/bytetriper/RAE.git",
    fallback="Not needed for BC, Pixel or 3D-flow. The RAE encoder checkpoint is a separate "
    "download; see docs/data.md.",
)

TRACK4WORLD = ExternalDependency(
    name="Track4World",
    purpose="feed-forward dense 3D point tracker that produces the 3D-flow targets",
    needed_by="regenerating 3D-flow targets from raw video",
    marker="track4world",
    upstream="https://github.com/TencentARC/Track4World.git",
    fallback="Only needed to PRODUCE flow sidecars. Training the 3D-flow variant on existing "
    "sidecars does not require it.",
)

ALL = (COSMOS_POLICY, RAE, TRACK4WORLD)


def status() -> str:
    """Human-readable presence table."""
    width = max(len(d.name) for d in ALL)
    rows = [
        f"  {d.name:<{width}}  {'present' if d.is_present() else 'MISSING':>7}   {d.needed_by}"
        for d in ALL
    ]
    return "\n".join(rows)
