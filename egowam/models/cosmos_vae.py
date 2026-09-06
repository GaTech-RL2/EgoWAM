from __future__ import annotations

import sys
from typing import Optional

import torch
import torch.nn as nn


def _maybe_stub_megatron() -> None:
    """If megatron is not installed, stub megatron.core.parallel_state so cosmos tokenizer can load (no context parallel)."""
    if "megatron.core" in sys.modules:
        return
    try:
        import megatron.core  # noqa: F401
    except ModuleNotFoundError:
        import types

        parallel_state = types.ModuleType("parallel_state")
        parallel_state.is_initialized = lambda: False
        megatron_core = types.ModuleType("megatron.core")
        megatron_core.parallel_state = parallel_state
        megatron = types.ModuleType("megatron")
        megatron.core = megatron_core
        sys.modules["megatron"] = megatron
        sys.modules["megatron.core"] = megatron_core
        sys.modules["megatron.core.parallel_state"] = parallel_state


class CosmosVAEWrapper(nn.Module):
    """Thin wrapper around Cosmos WAN VAE tokenizer interface."""

    def __init__(
        self,
        vae_pth: Optional[str] = None,
        s3_credential_path: str = "credentials/s3_training.secret",
        chunk_duration: int = 1,
        freeze: bool = True,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.freeze = freeze
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.interface = self._build_interface(
            vae_pth=vae_pth,
            s3_credential_path=s3_credential_path,
            chunk_duration=chunk_duration,
            device=device,
        )

        if self.freeze:
            model = getattr(self.interface.model, "model", self.interface.model)
            if hasattr(model, "parameters"):
                for p in model.parameters():
                    p.requires_grad = False

    def _apply(self, fn):
        """Move nested Wan VAE weights; ``self.interface`` is not an ``nn.Module`` child."""
        super()._apply(fn)
        iface = getattr(self, "interface", None)
        if iface is None:
            return self
        wan = getattr(iface, "model", None)
        if wan is None:
            return self
        inner = getattr(wan, "model", None)
        if isinstance(inner, nn.Module):
            wan.model = inner._apply(fn)
        for name in ("mean", "std", "img_mean", "img_std", "video_mean", "video_std"):
            t = getattr(wan, name, None)
            if torch.is_tensor(t):
                setattr(wan, name, fn(t))
        if hasattr(wan, "scale") and torch.is_tensor(getattr(wan, "mean", None)):
            m, s = wan.mean, wan.std
            if torch.is_tensor(s) and s.numel() > 0:
                wan.scale = [m, 1.0 / s]
        return self

    def _build_interface(
        self,
        vae_pth: Optional[str],
        s3_credential_path: str,
        chunk_duration: int,
        device: str = "cuda",
    ):
        from egowam.external_deps import COSMOS_POLICY

        # Fails naming the path it wanted rather than with a bare ImportError.
        cosmos_root = COSMOS_POLICY.require()
        if str(cosmos_root) not in sys.path:
            sys.path.append(str(cosmos_root))

        # Cosmos tokenizer imports megatron.core.parallel_state; stub it when absent
        # so we can run without installing megatron (single-GPU, no context parallel).
        _maybe_stub_megatron()

        try:
            from cosmos_policy._src.predict2.tokenizers.wan2pt1 import (
                Wan2pt1VAEInterface,
            )
        except Exception as exc:
            raise ImportError(
                "Failed to import Cosmos WAN VAE interface from external/cosmos-policy. "
                "If you see 'No module named megatron', install megatron-core or run in "
                "the cosmos-policy environment."
            ) from exc

        kwargs = {
            "chunk_duration": chunk_duration,
            "s3_credential_path": s3_credential_path,
            "device": device,
        }
        if vae_pth:
            kwargs["vae_pth"] = vae_pth
        return Wan2pt1VAEInterface(**kwargs)

    @torch.no_grad()
    def encode(self, videos_bcthw: torch.Tensor) -> torch.Tensor:
        return self.interface.encode(videos_bcthw)

    @torch.no_grad()
    def decode(self, latents_bcthw: torch.Tensor) -> torch.Tensor:
        return self.interface.decode(latents_bcthw)

    def decode_with_grad(self, latents_bcthw: torch.Tensor) -> torch.Tensor:
        """Decode allowing gradients to flow through the (frozen) VAE decoder.

        Used by FutureImageRegressionHead so that pixel-space loss can train the
        upstream latent-prediction MLP.
        """
        return self.interface.decode(latents_bcthw)
