"""
wan_dit.py

VACE-style Diffusion Transformer head for future image latent prediction.

Architecture overview (Wan2.1 VACE-inspired, adapted for HPT conditioning):

  cond_features [B, T_cond, cond_dim]   (HPT trunk future tokens)
      │
      ▼ vace_cond_proj
  context [B, T_cond, D]                D = hidden_dim
      │
      │  repeat per camera → [B·N, T_cond, D]
      │
      ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  For each block i = 0 … n_blocks-1:                         │
  │                                                             │
  │   VACE branch                                               │
  │     context_i = vace_blocks[i](context_{i-1})              │
  │     hint_i    = zero_proj[i](context_i)   ← zero-init      │
  │                                                             │
  │   Main branch                                               │
  │     latent = MainDiTBlock(latent, t_emb, hint=hint_i)       │
  │       ├─ AdaLN self-attention on latent patches             │
  │       ├─ Cross-attention latent → hint_i                    │
  │       └─ AdaLN MLP                                          │
  └─────────────────────────────────────────────────────────────┘
      │
      ▼ FinalLayer + unpatchify
  [B, N, C·H·W]   (velocity / clean-latent prediction)

Compatible with the existing FMPolicy / DenoisingPolicy interface:

    head.compute_loss(features, aux_data)   ← features [B, T_cond, cond_dim]
    head(global_cond)                        ← global_cond may be (features, domain) tuple
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint

OFFICIAL_WAN_REPO_ID = "Wan-AI/Wan2.1-VACE-1.3B"
OFFICIAL_WAN_TEXTCTX_REPO_ID = "Wan-AI/Wan2.1-T2V-1.3B"
OFFICIAL_WAN_FORBIDDEN_PATH_KEYWORDS = (
    "t5",
    "umt5",
    "text",
    "vae",
    "clip",
    "tokenizer",
    "scheduler",
)
OFFICIAL_WAN_MIN_CHECKPOINT_BYTES = 1_000_000_000

# ---------------------------------------------------------------------------
# Positional / time embeddings  (shared with dit_image_decoder)
# ---------------------------------------------------------------------------


def sinusoidal_embedding(
    timesteps: torch.Tensor, dim: int, max_period: float = 10_000.0
) -> torch.Tensor:
    """Scalar timesteps [B] → [B, dim]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def _get_1d_sincos(dim: int, positions: torch.Tensor) -> torch.Tensor:
    assert dim % 2 == 0
    half = dim // 2
    omega = torch.arange(half, dtype=torch.float32, device=positions.device)
    omega = 1.0 / (10000 ** (omega / half))
    out = positions.float().unsqueeze(1) * omega.unsqueeze(0)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=-1)


def build_2d_sincos_pos_embed(
    embed_dim: int, grid_h: int, grid_w: int, device, dtype
) -> torch.Tensor:
    """[grid_h * grid_w, embed_dim] 2-D sincos positional embedding."""
    assert embed_dim % 2 == 0
    half = embed_dim // 2
    emb_h = _get_1d_sincos(
        half, torch.arange(grid_h, device=device, dtype=torch.float32)
    )
    emb_w = _get_1d_sincos(
        half, torch.arange(grid_w, device=device, dtype=torch.float32)
    )
    grid = torch.cat(
        [
            emb_h[:, None, :].expand(grid_h, grid_w, -1),
            emb_w[None, :, :].expand(grid_h, grid_w, -1),
        ],
        dim=-1,
    )
    return grid.reshape(grid_h * grid_w, embed_dim).to(dtype=dtype)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PatchEmbed2D(nn.Module):
    """Conv2D patchify: [B, C, H, W] → [B, S, D]  with S = (H/ps)*(W/ps)."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # [B, D, H', W']
        return x.flatten(2).transpose(1, 2)  # [B, S, D]


def unpatchify2d(
    x: torch.Tensor, *, channels: int, patch_size: int, height: int, width: int
) -> torch.Tensor:
    """[B, S, patch_dim] → [B, C, H, W]  with patch_dim = C * ps^2."""
    gh, gw = height // patch_size, width // patch_size
    B, S, pd = x.shape
    assert S == gh * gw, f"S={S} != {gh}*{gw}"
    assert pd == channels * patch_size * patch_size
    p = patch_size
    x = x.reshape(B, gh, gw, channels, p, p)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    return x.reshape(B, channels, gh * p, gw * p)


class FFN(nn.Module):
    def __init__(self, dim: int, ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VACEContextBlock(nn.Module):
    """Lightweight transformer block for processing VACE context tokens.

    Applies self-attention + FFN to the context token sequence.  The output
    is then used as a "hint" for the corresponding main denoising block via a
    zero-initialised projection (see ``WanVACEDiT``).
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.ffn = FFN(hidden_dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B_N, T_cond, D]
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class MainDiTBlock(nn.Module):
    """Main denoising block: AdaLN self-attention + cross-attention to VACE hint + AdaLN FFN.

    AdaLN conditioning is derived from the time embedding; the VACE hint provides
    additional context via cross-attention.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )

        self.norm_cross = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )

        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.ffn = FFN(hidden_dim, mlp_ratio, dropout)

        # AdaLN-Zero: 6 * D → shift_sa, scale_sa, gate_sa, shift_ffn, scale_ffn, gate_ffn
        self.adaLN = nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)

    def forward(
        self, x: torch.Tensor, t_emb: torch.Tensor, hint: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x     : [B_N, S, D]  noisy latent patch tokens
        t_emb : [B_N, D]     AdaLN conditioning (time + camera)
        hint  : [B_N, T_cond, D] | None   VACE context (cross-attention key/value)
        """
        shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff = self.adaLN(
            t_emb
        ).chunk(6, dim=-1)

        # Self-attention with AdaLN
        h = modulate(self.norm1(x), shift_sa, scale_sa)
        sa_out, _ = self.self_attn(h, h, h)
        x = x + gate_sa.unsqueeze(1) * sa_out

        # Cross-attention to VACE hint (no AdaLN on this path)
        if hint is not None:
            q = self.norm_cross(x)
            ca_out, _ = self.cross_attn(q, hint, hint)
            x = x + ca_out

        # FFN with AdaLN
        h2 = modulate(self.norm2(x), shift_ff, scale_ff)
        x = x + gate_ff.unsqueeze(1) * self.ffn(h2)
        return x


class FinalLayer(nn.Module):
    """AdaLN + zero-init linear → patch output."""

    def __init__(self, hidden_dim: int, patch_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, patch_dim)
        self.adaLN = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(t_emb).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


# ---------------------------------------------------------------------------
# Full VACE-style DiT
# ---------------------------------------------------------------------------


class WanVACEDiT(nn.Module):
    """VACE-style Diffusion Transformer for per-camera future image latent denoising.

    Processes *all* cameras in a single flattened batch dimension so that camera
    embeddings capture inter-camera differences while the VACE context is shared.

    Forward:
        x_noisy       : [B, N_cam, C·H·W]   noisy latent
        t             : [B]                  flow-matching timestep in (0, 1]
        cond_features : [B, T_cond, cond_dim]  HPT future tokens
        N_cam         : int                  required when N_cam cannot be inferred

    Returns:
        velocity / clean-latent prediction   [B, N_cam, C·H·W]
    """

    def __init__(
        self,
        cond_dim: int = 256,
        future_horizon: int = 16,
        latent_channels: int = 16,
        latent_h: int = 16,
        latent_w: int = 16,
        patch_size: int = 2,
        hidden_dim: int = 768,
        n_blocks: int = 12,
        n_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        time_dim: int = 256,
        max_cameras: int = 4,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.n_blocks = n_blocks
        self.max_cameras = max_cameras

        if latent_h % patch_size != 0 or latent_w % patch_size != 0:
            raise ValueError(
                f"latent_h={latent_h}, latent_w={latent_w} must be divisible by patch_size={patch_size}"
            )
        self.grid_h = latent_h // patch_size
        self.grid_w = latent_w // patch_size
        self.num_patches = self.grid_h * self.grid_w
        self.patch_dim = latent_channels * patch_size * patch_size

        # --- Time embedding ---
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # --- Camera embedding ---
        self.camera_embed = nn.Embedding(max_cameras, hidden_dim)
        nn.init.zeros_(self.camera_embed.weight)

        # --- Latent patch embedding ---
        self.patch_embed = PatchEmbed2D(latent_channels, hidden_dim, patch_size)
        self.register_buffer("pos_embed", torch.zeros(self.num_patches, hidden_dim))
        self.register_buffer("_pos_init_done", torch.tensor(False))

        # --- VACE context projection (cond_dim → hidden_dim) ---
        self.vace_cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # --- Per-block VACE context and main denoising blocks ---
        self.vace_blocks = nn.ModuleList(
            [
                VACEContextBlock(hidden_dim, n_heads, mlp_ratio, dropout)
                for _ in range(n_blocks)
            ]
        )
        self.main_blocks = nn.ModuleList(
            [
                MainDiTBlock(hidden_dim, n_heads, mlp_ratio, dropout)
                for _ in range(n_blocks)
            ]
        )

        # Zero-init hint projection: vace_block output → hint added as cross-attention KV
        # Already handled implicitly; the cross-attention weights in MainDiTBlock learn to
        # attend to the VACE hint once the VACE blocks produce useful features.
        # Optionally a zero-init gate per block for training stability:
        self.vace_gates = nn.ParameterList(
            [nn.Parameter(torch.zeros(())) for _ in range(n_blocks)]
        )

        # --- Final output layer ---
        self.final_layer = FinalLayer(hidden_dim, self.patch_dim)

    def _init_pos_embed(self, device, dtype) -> None:
        pe = build_2d_sincos_pos_embed(
            self.hidden_dim, self.grid_h, self.grid_w, device, dtype
        )
        self.pos_embed.copy_(pe)

    def forward(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        cond_features: torch.Tensor,
        N_cam: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x_noisy       : [B, N_cam, C*H*W]
        t             : [B]
        cond_features : [B, T_cond, cond_dim]
        N_cam         : int (inferred from x_noisy if not provided)

        Returns
        -------
        velocity : [B, N_cam, C*H*W]
        """
        B, N, D_lat = x_noisy.shape
        if N_cam is None:
            N_cam = N
        assert N == N_cam, f"x_noisy N_cam={N} != expected N_cam={N_cam}"
        C, H, W = self.latent_channels, self.latent_h, self.latent_w
        assert D_lat == C * H * W, f"cam_lat_dim={D_lat} != {C}*{H}*{W}={C * H * W}"

        B_N = B * N_cam  # flattened batch × camera

        # ---- 1. Patchify noisy latent ----
        x_spatial = x_noisy.reshape(B_N, C, H, W)  # [B*N, C, H, W]
        if not self._pos_init_done:
            self._init_pos_embed(x_spatial.device, x_spatial.dtype)
            self._pos_init_done.fill_(True)
        tokens = self.patch_embed(x_spatial)  # [B*N, S, D]
        tokens = tokens + self.pos_embed.unsqueeze(0)  # 2D spatial positional embed

        # Camera embedding
        cam_ids = torch.arange(N_cam, device=x_noisy.device)  # [N]
        cam_ids = cam_ids.unsqueeze(0).expand(B, -1).reshape(B_N)  # [B*N]
        cam_emb = self.camera_embed(cam_ids)  # [B*N, D]
        tokens = tokens + cam_emb.unsqueeze(1)  # broadcast over S

        # ---- 2. Time embedding ----
        t_emb = sinusoidal_embedding(t, self.time_dim)  # [B, time_dim]
        t_emb = self.time_mlp(t_emb)  # [B, D]
        # Expand to [B*N, D] — same time for all cameras in a sample
        t_emb = t_emb.unsqueeze(1).expand(B, N_cam, -1).reshape(B_N, -1)  # [B*N, D]

        # ---- 3. Project VACE context ----
        ctx = self.vace_cond_proj(cond_features)  # [B, T_cond, D]
        T_cond = ctx.shape[1]
        # Expand to [B*N, T_cond, D]
        ctx = (
            ctx.unsqueeze(1)
            .expand(B, N_cam, T_cond, -1)
            .reshape(B_N, T_cond, self.hidden_dim)
        )

        assert tokens.shape == (
            B_N,
            self.num_patches,
            self.hidden_dim,
        ), f"tokens shape mismatch: {tokens.shape}"
        assert ctx.shape == (
            B_N,
            T_cond,
            self.hidden_dim,
        ), f"ctx shape mismatch: {ctx.shape}"

        # ---- 4. VACE + Main blocks ----
        for i, (vace_blk, main_blk) in enumerate(
            zip(self.vace_blocks, self.main_blocks)
        ):
            ctx = vace_blk(ctx)  # [B*N, T_cond, D]  — evolves context
            # Gate starts at 0 so early training is stable; grows as the VACE blocks learn.
            hint = torch.tanh(self.vace_gates[i]) * ctx  # [B*N, T_cond, D]
            tokens = main_blk(tokens, t_emb, hint=hint)

        # ---- 5. Final layer + unpatchify ----
        out_tokens = self.final_layer(tokens, t_emb)  # [B*N, S, patch_dim]
        out_spatial = unpatchify2d(
            out_tokens, channels=C, patch_size=self.patch_size, height=H, width=W
        )  # [B*N, C, H, W]

        # Residual connection in latent space (preserves signal at init)
        out_spatial = out_spatial + x_spatial

        return out_spatial.reshape(B, N_cam, C * H * W)  # [B, N_cam, C*H*W]


# ---------------------------------------------------------------------------
# Pretrained weight loading utilities
# ---------------------------------------------------------------------------


def _format_param_count(n_params: int) -> str:
    if n_params >= 1_000_000_000:
        return f"{n_params / 1_000_000_000:.2f}B"
    return f"{n_params / 1_000_000:.1f}M"


def _count_params(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(
        p.numel() for p in module.parameters() if not trainable_only or p.requires_grad
    )


def _format_bytes(n_bytes: Optional[int]) -> str:
    if n_bytes is None:
        return "unknown size"
    if n_bytes >= 1024**3:
        return f"{n_bytes / 1024**3:.2f} GiB"
    if n_bytes >= 1024**2:
        return f"{n_bytes / 1024**2:.1f} MiB"
    return f"{n_bytes} bytes"


def _torch_dtype_from_string(dtype: str) -> torch.dtype:
    normalized = (dtype or "bf16").lower()
    if normalized in ("bf16", "bfloat16"):
        return torch.bfloat16
    if normalized in ("fp16", "float16", "half"):
        return torch.float16
    if normalized in ("fp32", "float32", "float"):
        return torch.float32
    raise ValueError(
        f"Unsupported official_dtype={dtype!r}; expected bf16, fp16, or fp32."
    )


def _path_has_forbidden_official_keyword(path: str) -> bool:
    lower = path.lower()
    return any(kw in lower for kw in OFFICIAL_WAN_FORBIDDEN_PATH_KEYWORDS)


def _raise_if_forbidden_official_checkpoint(path: str) -> None:
    lower = path.lower()
    if "t5" in lower or "umt5" in lower or "text" in lower:
        raise RuntimeError(
            f"Selected checkpoint appears to be T5/text encoder, not DiT: {path}"
        )
    if "vae" in lower:
        raise RuntimeError(f"Selected checkpoint appears to be VAE, not DiT: {path}")
    for kw in ("clip", "tokenizer", "scheduler"):
        if kw in lower:
            raise RuntimeError(
                f"Selected checkpoint appears to be {kw}, not DiT: {path}"
            )


def _is_dit_weight_candidate(path: str) -> bool:
    rel = path.replace("\\", "/")
    name = Path(rel).name.lower()
    if _path_has_forbidden_official_keyword(rel):
        return False
    if name == "diffusion_pytorch_model.safetensors.index.json":
        return True
    return name.startswith("diffusion_pytorch_model") and name.endswith(".safetensors")


def _select_dit_checkpoint_candidate(
    candidates: List[Tuple[str, Optional[int]]],
    *,
    repo_id: str = OFFICIAL_WAN_REPO_ID,
    backend_label: str = "Wan DiT",
) -> Tuple[str, Optional[int]]:
    valid = [(rel, size) for rel, size in candidates if _is_dit_weight_candidate(rel)]
    if not valid:
        raise RuntimeError(
            f"No valid {backend_label} checkpoint found. Download the official DiT files with:\n"
            f"  huggingface-cli download {repo_id} --local-dir <dir> "
            '--include "diffusion_pytorch_model*" "model_index.json" "config.json" "**/*.json"'
        )

    def rank(item: Tuple[str, Optional[int]]) -> Tuple[int, str]:
        rel, _size = item
        name = Path(rel).name.lower()
        if name == "diffusion_pytorch_model.safetensors":
            return (0, rel)
        if name == "diffusion_pytorch_model.safetensors.index.json":
            return (1, rel)
        return (2, rel)

    selected_rel, selected_size = sorted(valid, key=rank)[0]
    _raise_if_forbidden_official_checkpoint(selected_rel)
    return selected_rel, selected_size


def _print_checkpoint_candidates(
    label: str,
    candidates: List[Tuple[str, Optional[int]]],
    selected_rel: str,
) -> None:
    cprint(f"[WanVACEFutureImageHead] {label} candidates:", "cyan")
    for rel, size in candidates:
        marker = "  *" if rel == selected_rel else "  -"
        cprint(f"{marker} {rel} ({_format_bytes(size)})", "cyan")
    cprint(f"[WanVACEFutureImageHead] Selected checkpoint path: {selected_rel}", "cyan")


def _local_checkpoint_candidates(root: Path) -> List[Tuple[str, Optional[int]]]:
    if root.is_file():
        _raise_if_forbidden_official_checkpoint(root.name)
        return [(root.name, root.stat().st_size)]

    candidates: List[Tuple[str, Optional[int]]] = []
    for pattern in ("*.safetensors", "*.pth", "*.pt", "*.bin", "*.json"):
        for path in sorted(root.rglob(pattern)):
            rel = path.relative_to(root).as_posix()
            candidates.append((rel, path.stat().st_size))
    return candidates


def _load_wan_pretrained(
    model: nn.Module,
    pretrained_path: str,
    load_main_blocks_only: bool,
    freeze_pretrained_main_blocks: bool,
    strict_pretrained_load: bool = True,
) -> None:
    """Load matching weights into the small custom fallback DiT.

    This path intentionally remains architecture-matching only. It may load a
    custom checkpoint for ``WanVACEDiT``, but it must not silently "succeed" with
    zero loaded tensors when handed an official 1.3B Wan checkpoint.
    """
    from egowam.utils.egowamUtils import download_from_huggingface

    if pretrained_path.startswith("hf://"):
        cprint(f"[WanVACE] Downloading Wan checkpoint from {pretrained_path} …", "cyan")
        local_dir = download_from_huggingface(pretrained_path[len("hf://") :])
        pretrained_path = local_dir

    ckpt_path = Path(pretrained_path)
    if ckpt_path.is_file():
        candidates = [ckpt_path]
    else:
        candidates = (
            list(ckpt_path.rglob("*.pth"))
            + list(ckpt_path.rglob("*.pt"))
            + list(ckpt_path.rglob("*.safetensors"))
        )

    skip_keywords = {
        "t5",
        "umt5",
        "text",
        "text_encoder",
        "tokenizer",
        "scheduler",
        "vae",
        "clip",
    }
    candidates = [
        p
        for p in candidates
        if not any(kw in p.as_posix().lower() for kw in skip_keywords)
    ]
    if not candidates:
        msg = f"[WanVACE] no compatible custom weight files found in {ckpt_path}."
        if strict_pretrained_load:
            raise RuntimeError(msg)
        cprint(f"{msg} Skipping pretrained loading.", "yellow")
        return

    def _custom_rank(path: Path) -> Tuple[int, str]:
        name = path.name.lower()
        if name.startswith("diffusion_pytorch_model") and name.endswith(".safetensors"):
            return (0, path.as_posix())
        if name.endswith(".safetensors"):
            return (1, path.as_posix())
        return (2, path.as_posix())

    ckpt_file = sorted(candidates, key=_custom_rank)[0]
    cprint("[WanVACE] Custom checkpoint candidates:", "cyan")
    for candidate in sorted(candidates, key=_custom_rank):
        cprint(f"  - {candidate} ({_format_bytes(candidate.stat().st_size)})", "cyan")
    cprint(f"[WanVACE] Loading pretrained weights from {ckpt_file}", "cyan")

    if ckpt_file.suffix == ".safetensors":
        from safetensors.torch import load_file

        ckpt_sd = load_file(str(ckpt_file), device="cpu")
    else:
        ckpt_sd = torch.load(str(ckpt_file), map_location="cpu")
        if "state_dict" in ckpt_sd:
            ckpt_sd = ckpt_sd["state_dict"]
        elif "model" in ckpt_sd:
            ckpt_sd = ckpt_sd["model"]

    # Optionally filter to main denoising block keys.
    if load_main_blocks_only:
        main_block_prefixes = (
            "blocks.",
            "main_blocks.",
            "patch_embed",
            "head.",
            "time_embedding.",
            "time_projection.",
            "vace_blocks.",
            "vace_patch_embedding.",
            "final_layer.",
        )
        ckpt_sd = {
            k: v for k, v in ckpt_sd.items() if k.startswith(main_block_prefixes)
        }

    model_sd = model.state_dict()
    n_loaded = 0
    n_skipped_missing = 0
    n_skipped_shape = 0
    n_unexpected = 0

    new_sd: Dict[str, torch.Tensor] = {}
    for k, v in ckpt_sd.items():
        if k not in model_sd:
            n_unexpected += 1
            continue
        if model_sd[k].shape != v.shape:
            cprint(
                f"  [WanVACE] shape mismatch {k}: ckpt={tuple(v.shape)} model={tuple(model_sd[k].shape)}",
                "yellow",
            )
            n_skipped_shape += 1
            continue
        new_sd[k] = v
        n_loaded += 1

    for k in model_sd:
        if k not in ckpt_sd:
            n_skipped_missing += 1

    model.load_state_dict(new_sd, strict=False)

    cprint(
        f"[WanVACE] Pretrained loading summary: "
        f"loaded={n_loaded}  shape_mismatch={n_skipped_shape}  "
        f"missing_in_ckpt={n_skipped_missing}  unexpected_in_ckpt={n_unexpected}",
        "green",
    )

    if n_loaded == 0:
        raise RuntimeError(
            f"[WanVACE] Pretrained loading matched zero tensors from {ckpt_file}. "
            "Refusing to continue with loaded=0. If you intended to use the official "
            "Wan-AI/Wan2.1-VACE-1.3B DiT, set wan_backend='official_vace_1_3b'."
        )

    if freeze_pretrained_main_blocks:
        frozen = 0
        for k in new_sd:
            parts = k.split(".")
            # Access parameter via getattr chain
            obj = model
            try:
                for p in parts[:-1]:
                    obj = getattr(obj, p) if not p.isdigit() else obj[int(p)]
                param = getattr(obj, parts[-1])
                if isinstance(param, nn.Parameter):
                    param.requires_grad = False
                    frozen += 1
            except (AttributeError, IndexError):
                pass
        cprint(f"[WanVACE] Froze {frozen} pretrained parameters.", "cyan")


# ---------------------------------------------------------------------------
# Official Diffusers Wan VACE backend
# ---------------------------------------------------------------------------


def _import_official_wan_vace():
    try:
        from diffusers import WanVACEPipeline
        from diffusers.models import WanVACETransformer3DModel
    except ImportError as exc:
        raise RuntimeError(
            "wan_backend='official_vace_1_3b' requires a recent diffusers install "
            "with Wan VACE support. Please upgrade/install recent diffusers, e.g. "
            "`pip install -U diffusers transformers accelerate safetensors`."
        ) from exc
    return WanVACEPipeline, WanVACETransformer3DModel


def _import_official_wan_textctx():
    try:
        from diffusers.models import WanTransformer3DModel
    except ImportError as exc:
        raise RuntimeError(
            "wan_backend='official_wan_1_3b_textctx' requires a recent diffusers "
            "install with WanTransformer3DModel support. Please upgrade/install "
            "recent diffusers, e.g. `pip install -U diffusers transformers "
            "accelerate safetensors`."
        ) from exc
    return WanTransformer3DModel


def _hf_checkpoint_candidates(repo_id: str) -> List[Tuple[str, Optional[int]]]:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "official Wan backends require huggingface_hub to list and download "
            "the selected Wan DiT checkpoint."
        ) from exc

    api = HfApi()
    try:
        info = api.model_info(repo_id, files_metadata=True)
    except TypeError:
        info = api.model_info(repo_id)

    candidates: List[Tuple[str, Optional[int]]] = []
    for sibling in getattr(info, "siblings", []):
        rel = getattr(sibling, "rfilename", "")
        if not rel:
            continue
        size = getattr(sibling, "size", None)
        if rel.endswith((".safetensors", ".pth", ".pt", ".bin", ".json")):
            candidates.append((rel, size))
    return candidates


def _download_official_snapshot(
    repo_id: str, backend_label: str = "official Wan DiT"
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "official Wan backends require huggingface_hub to download "
            "the official Wan DiT checkpoint."
        ) from exc

    cprint(
        f"[WanVACEFutureImageHead] Downloading {backend_label} from {repo_id}", "cyan"
    )
    local_dir = snapshot_download(
        repo_id=repo_id,
        allow_patterns=[
            "diffusion_pytorch_model*",
            "**/diffusion_pytorch_model*",
            "config.json",
            "**/config.json",
            "model_index.json",
            "**/*.json",
        ],
        ignore_patterns=[
            "*t5*",
            "*umt5*",
            "*text*",
            "*vae*",
            "*VAE*",
            "*clip*",
            "*tokenizer*",
            "*scheduler*",
            "*.pth",
            "*.pt",
            "*.bin",
        ],
    )
    return Path(local_dir)


def _resolve_official_checkpoint(
    pretrained_path: str,
    official_repo_id: str,
    *,
    backend_label: str = "official Wan DiT",
) -> Tuple[Path, Path]:
    source = pretrained_path or f"hf://{official_repo_id}"

    if source.startswith("hf://"):
        repo_id = source[len("hf://") :]
        remote_candidates = _hf_checkpoint_candidates(repo_id)
        selected_rel, _selected_size = _select_dit_checkpoint_candidate(
            remote_candidates,
            repo_id=repo_id,
            backend_label=backend_label,
        )
        _print_checkpoint_candidates(
            f"Remote {backend_label}", remote_candidates, selected_rel
        )
        local_root = _download_official_snapshot(repo_id, backend_label)
    else:
        maybe_path = Path(source)
        if maybe_path.exists():
            local_root = maybe_path if maybe_path.is_dir() else maybe_path.parent
        elif "/" in source and not source.startswith((".", "/")):
            remote_candidates = _hf_checkpoint_candidates(source)
            selected_rel, _selected_size = _select_dit_checkpoint_candidate(
                remote_candidates,
                repo_id=source,
                backend_label=backend_label,
            )
            _print_checkpoint_candidates(
                f"Remote {backend_label}", remote_candidates, selected_rel
            )
            local_root = _download_official_snapshot(source, backend_label)
        else:
            raise FileNotFoundError(
                f"{backend_label} checkpoint path does not exist: {source}"
            )

    local_candidates = _local_checkpoint_candidates(local_root)
    selected_rel, _selected_size = _select_dit_checkpoint_candidate(
        local_candidates,
        repo_id=official_repo_id,
        backend_label=backend_label,
    )
    _print_checkpoint_candidates(
        f"Local {backend_label}", local_candidates, selected_rel
    )
    selected_path = local_root / selected_rel
    _raise_if_forbidden_official_checkpoint(selected_rel)
    return local_root, selected_path


def _checkpoint_size_bytes(selected_path: Path) -> int:
    if selected_path.name == "diffusion_pytorch_model.safetensors.index.json":
        with selected_path.open("r") as f:
            index = json.load(f)
        shard_names = sorted(set(index.get("weight_map", {}).values()))
        total = 0
        for shard_name in shard_names:
            shard_path = selected_path.parent / shard_name
            if not shard_path.exists():
                raise RuntimeError(
                    f"Shard listed in {selected_path} is missing: {shard_path}"
                )
            total += shard_path.stat().st_size
        return total
    return selected_path.stat().st_size


def _check_official_checkpoint_size(
    selected_path: Path,
    allow_small_official_checkpoint: bool,
    backend_label: str = "Wan2.1 official DiT",
) -> None:
    size_bytes = _checkpoint_size_bytes(selected_path)
    if (
        size_bytes < OFFICIAL_WAN_MIN_CHECKPOINT_BYTES
        and not allow_small_official_checkpoint
    ):
        raise RuntimeError(
            f"Selected DiT checkpoint is too small for {backend_label}: "
            f"{selected_path} ({_format_bytes(size_bytes)}). Refusing to load a likely "
            "wrong file. Set allow_small_official_checkpoint=True only for explicit tests."
        )


def _diffusers_base_config_from_original_wan_config(config: Dict) -> Dict:
    dim = int(config["dim"])
    num_heads = int(config["num_heads"])
    if dim % num_heads != 0:
        raise ValueError(
            f"Wan config dim={dim} is not divisible by num_heads={num_heads}"
        )
    return {
        "patch_size": (1, 2, 2),
        "num_attention_heads": num_heads,
        "attention_head_dim": dim // num_heads,
        "in_channels": int(config.get("in_dim", 16)),
        "out_channels": int(config.get("out_dim", 16)),
        "text_dim": 4096,
        "freq_dim": int(config.get("freq_dim", 256)),
        "ffn_dim": int(config["ffn_dim"]),
        "num_layers": int(config["num_layers"]),
        "cross_attn_norm": True,
        "qk_norm": "rms_norm_across_heads",
        "eps": float(config.get("eps", 1e-6)),
        "image_dim": None,
        "added_kv_proj_dim": None,
        "rope_max_seq_len": 1024,
        "pos_embed_seq_len": None,
    }


def _diffusers_config_from_original_wan_config(config: Dict) -> Dict:
    diffusers_config = _diffusers_base_config_from_original_wan_config(config)
    diffusers_config.update(
        {
            "vace_layers": list(config.get("vace_layers", [0])),
            "vace_in_channels": int(config.get("vace_in_dim", 96)),
        }
    )
    return diffusers_config


def _convert_original_wan_transformer_key(key: str) -> Optional[str]:
    """Map original base Wan checkpoint keys to Diffusers WanTransformer3DModel keys."""
    exact_map = {
        "head.modulation": "scale_shift_table",
        "head.head.weight": "proj_out.weight",
        "head.head.bias": "proj_out.bias",
        "time_embedding.0.weight": "condition_embedder.time_embedder.linear_1.weight",
        "time_embedding.0.bias": "condition_embedder.time_embedder.linear_1.bias",
        "time_embedding.2.weight": "condition_embedder.time_embedder.linear_2.weight",
        "time_embedding.2.bias": "condition_embedder.time_embedder.linear_2.bias",
        "time_projection.1.weight": "condition_embedder.time_proj.weight",
        "time_projection.1.bias": "condition_embedder.time_proj.bias",
        "text_embedding.0.weight": "condition_embedder.text_embedder.linear_1.weight",
        "text_embedding.0.bias": "condition_embedder.text_embedder.linear_1.bias",
        "text_embedding.2.weight": "condition_embedder.text_embedder.linear_2.weight",
        "text_embedding.2.bias": "condition_embedder.text_embedder.linear_2.bias",
        "patch_embedding.weight": "patch_embedding.weight",
        "patch_embedding.bias": "patch_embedding.bias",
    }
    if key in exact_map:
        return exact_map[key]

    parts = key.split(".")
    if len(parts) < 3:
        return key
    if parts[0].startswith("vace"):
        return None
    if parts[0] != "blocks":
        return key

    block_prefix = ".".join(parts[:2])
    rest = ".".join(parts[2:])
    block_map = {
        "modulation": "scale_shift_table",
        "self_attn.q.weight": "attn1.to_q.weight",
        "self_attn.q.bias": "attn1.to_q.bias",
        "self_attn.k.weight": "attn1.to_k.weight",
        "self_attn.k.bias": "attn1.to_k.bias",
        "self_attn.v.weight": "attn1.to_v.weight",
        "self_attn.v.bias": "attn1.to_v.bias",
        "self_attn.o.weight": "attn1.to_out.0.weight",
        "self_attn.o.bias": "attn1.to_out.0.bias",
        "self_attn.norm_q.weight": "attn1.norm_q.weight",
        "self_attn.norm_k.weight": "attn1.norm_k.weight",
        "cross_attn.q.weight": "attn2.to_q.weight",
        "cross_attn.q.bias": "attn2.to_q.bias",
        "cross_attn.k.weight": "attn2.to_k.weight",
        "cross_attn.k.bias": "attn2.to_k.bias",
        "cross_attn.v.weight": "attn2.to_v.weight",
        "cross_attn.v.bias": "attn2.to_v.bias",
        "cross_attn.o.weight": "attn2.to_out.0.weight",
        "cross_attn.o.bias": "attn2.to_out.0.bias",
        "cross_attn.norm_q.weight": "attn2.norm_q.weight",
        "cross_attn.norm_k.weight": "attn2.norm_k.weight",
        "norm3.weight": "norm2.weight",
        "norm3.bias": "norm2.bias",
        "ffn.0.weight": "ffn.net.0.proj.weight",
        "ffn.0.bias": "ffn.net.0.proj.bias",
        "ffn.2.weight": "ffn.net.2.weight",
        "ffn.2.bias": "ffn.net.2.bias",
    }
    if rest in block_map:
        return f"{block_prefix}.{block_map[rest]}"

    return key


def _load_safetensors_checkpoint(selected_path: Path) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    if selected_path.name != "diffusion_pytorch_model.safetensors.index.json":
        return load_file(str(selected_path), device="cpu")

    with selected_path.open("r") as f:
        index = json.load(f)
    state_dict: Dict[str, torch.Tensor] = {}
    for shard_name in sorted(set(index.get("weight_map", {}).values())):
        shard_path = selected_path.parent / shard_name
        state_dict.update(load_file(str(shard_path), device="cpu"))
    return state_dict


def _convert_original_wan_vace_key(key: str) -> Optional[str]:
    """Map original Wan VACE checkpoint keys to Diffusers module names."""
    exact_map = {
        "head.modulation": "scale_shift_table",
        "head.head.weight": "proj_out.weight",
        "head.head.bias": "proj_out.bias",
        "time_embedding.0.weight": "condition_embedder.time_embedder.linear_1.weight",
        "time_embedding.0.bias": "condition_embedder.time_embedder.linear_1.bias",
        "time_embedding.2.weight": "condition_embedder.time_embedder.linear_2.weight",
        "time_embedding.2.bias": "condition_embedder.time_embedder.linear_2.bias",
        "time_projection.1.weight": "condition_embedder.time_proj.weight",
        "time_projection.1.bias": "condition_embedder.time_proj.bias",
        "text_embedding.0.weight": "condition_embedder.text_embedder.linear_1.weight",
        "text_embedding.0.bias": "condition_embedder.text_embedder.linear_1.bias",
        "text_embedding.2.weight": "condition_embedder.text_embedder.linear_2.weight",
        "text_embedding.2.bias": "condition_embedder.text_embedder.linear_2.bias",
        "patch_embedding.weight": "patch_embedding.weight",
        "patch_embedding.bias": "patch_embedding.bias",
        "vace_patch_embedding.weight": "vace_patch_embedding.weight",
        "vace_patch_embedding.bias": "vace_patch_embedding.bias",
    }
    if key in exact_map:
        return exact_map[key]

    parts = key.split(".")
    if len(parts) < 3 or parts[0] not in ("blocks", "vace_blocks"):
        return key

    block_prefix = ".".join(parts[:2])
    rest = ".".join(parts[2:])
    block_map = {
        "modulation": "scale_shift_table",
        "self_attn.q.weight": "attn1.to_q.weight",
        "self_attn.q.bias": "attn1.to_q.bias",
        "self_attn.k.weight": "attn1.to_k.weight",
        "self_attn.k.bias": "attn1.to_k.bias",
        "self_attn.v.weight": "attn1.to_v.weight",
        "self_attn.v.bias": "attn1.to_v.bias",
        "self_attn.o.weight": "attn1.to_out.0.weight",
        "self_attn.o.bias": "attn1.to_out.0.bias",
        "self_attn.norm_q.weight": "attn1.norm_q.weight",
        "self_attn.norm_k.weight": "attn1.norm_k.weight",
        "cross_attn.q.weight": "attn2.to_q.weight",
        "cross_attn.q.bias": "attn2.to_q.bias",
        "cross_attn.k.weight": "attn2.to_k.weight",
        "cross_attn.k.bias": "attn2.to_k.bias",
        "cross_attn.v.weight": "attn2.to_v.weight",
        "cross_attn.v.bias": "attn2.to_v.bias",
        "cross_attn.o.weight": "attn2.to_out.0.weight",
        "cross_attn.o.bias": "attn2.to_out.0.bias",
        "cross_attn.norm_q.weight": "attn2.norm_q.weight",
        "cross_attn.norm_k.weight": "attn2.norm_k.weight",
        "norm3.weight": "norm2.weight",
        "norm3.bias": "norm2.bias",
        "ffn.0.weight": "ffn.net.0.proj.weight",
        "ffn.0.bias": "ffn.net.0.proj.bias",
        "ffn.2.weight": "ffn.net.2.weight",
        "ffn.2.bias": "ffn.net.2.bias",
    }
    if rest in block_map:
        return f"{block_prefix}.{block_map[rest]}"

    if parts[0] == "vace_blocks":
        if rest == "before_proj.weight":
            return f"{block_prefix}.proj_in.weight"
        if rest == "before_proj.bias":
            return f"{block_prefix}.proj_in.bias"
        if rest == "after_proj.weight":
            return f"{block_prefix}.proj_out.weight"
        if rest == "after_proj.bias":
            return f"{block_prefix}.proj_out.bias"

    return key


def _load_manual_official_state_dict(
    transformer: nn.Module,
    selected_path: Path,
    *,
    key_converter,
    checkpoint_label: str,
    min_loaded_params: int,
) -> None:
    ckpt_sd = _load_safetensors_checkpoint(selected_path)
    model_sd = transformer.state_dict()
    compatible: Dict[str, torch.Tensor] = {}
    n_shape_mismatch = 0
    n_unexpected = 0

    for key, value in ckpt_sd.items():
        converted_key = key_converter(key)
        if converted_key is None or converted_key not in model_sd:
            n_unexpected += 1
            continue
        if model_sd[converted_key].shape != value.shape:
            n_shape_mismatch += 1
            continue
        compatible[converted_key] = value

    loaded_params = sum(v.numel() for v in compatible.values())
    if loaded_params == 0:
        raise RuntimeError(
            f"{checkpoint_label} checkpoint loading matched zero tensors from {selected_path}. "
            "Refusing to continue with loaded=0; upgrade diffusers or use a converted "
            "Diffusers-format checkpoint."
        )
    if loaded_params < min_loaded_params:
        raise RuntimeError(
            f"{checkpoint_label} checkpoint loading matched only {loaded_params} params "
            f"from {selected_path}; this is not a valid 1.3B Wan DiT load."
        )

    transformer.load_state_dict(compatible, strict=False)
    cprint(
        f"[WanVACEFutureImageHead] {checkpoint_label} checkpoint loading summary: "
        f"loaded_params={_format_param_count(loaded_params)}  "
        f"shape_mismatch={n_shape_mismatch}  unexpected={n_unexpected}",
        "green",
    )


def _load_official_wan_vace_transformer(
    *,
    pretrained_path: str,
    official_repo_id: str,
    official_dtype: str,
    allow_small_official_checkpoint: bool,
) -> Tuple[nn.Module, Path]:
    _WanVACEPipeline, WanVACETransformer3DModel = _import_official_wan_vace()
    torch_dtype = _torch_dtype_from_string(official_dtype)
    local_root, selected_path = _resolve_official_checkpoint(
        pretrained_path,
        official_repo_id,
        backend_label="official Wan VACE DiT",
    )
    _check_official_checkpoint_size(
        selected_path,
        allow_small_official_checkpoint,
        backend_label="Wan2.1-VACE-1.3B",
    )

    selected_parent = selected_path.parent
    config_path = selected_parent / "config.json"
    config: Dict = {}
    if config_path.exists():
        with config_path.open("r") as f:
            config = json.load(f)

    if config.get("_class_name") == "WanVACETransformer3DModel":
        transformer = WanVACETransformer3DModel.from_pretrained(
            str(selected_parent),
            torch_dtype=torch_dtype,
        )
        return transformer, selected_path

    transformer_dir = local_root / "transformer"
    transformer_config = transformer_dir / "config.json"
    if transformer_config.exists():
        with transformer_config.open("r") as f:
            transformer_cfg = json.load(f)
        if transformer_cfg.get("_class_name") == "WanVACETransformer3DModel":
            transformer = WanVACETransformer3DModel.from_pretrained(
                str(local_root),
                subfolder="transformer",
                torch_dtype=torch_dtype,
            )
            return transformer, selected_path

    if config.get("_class_name") == "VaceWanModel":
        cprint(
            "[WanVACEFutureImageHead] Loading original-format Wan VACE checkpoint "
            "with explicit 1.3B Diffusers config mapping.",
            "cyan",
        )
        transformer = WanVACETransformer3DModel(
            **_diffusers_config_from_original_wan_config(config)
        )
        _load_manual_official_state_dict(
            transformer,
            selected_path,
            key_converter=_convert_original_wan_vace_key,
            checkpoint_label="Official Wan VACE",
            min_loaded_params=1_000_000_000,
        )
        transformer.to(dtype=torch_dtype)
        return transformer, selected_path

    try:
        transformer = WanVACETransformer3DModel.from_pretrained(
            str(selected_parent),
            torch_dtype=torch_dtype,
        )
        return transformer, selected_path
    except Exception as exc:
        raise RuntimeError(
            f"Could not instantiate official Wan VACE transformer from {selected_parent}. "
            "Install a recent diffusers version with Wan VACE support or provide a "
            "Diffusers-format checkpoint directory containing transformer/config.json "
            "and diffusion_pytorch_model*.safetensors."
        ) from exc


def _looks_like_vace_config(config: Dict) -> bool:
    class_name = str(config.get("_class_name", "")).lower()
    model_type = str(config.get("model_type", "")).lower()
    return (
        "vace" in class_name
        or model_type == "vace"
        or "vace_in_dim" in config
        or "vace_layers" in config
    )


def _load_official_wan_textctx_transformer(
    *,
    pretrained_path: str,
    official_repo_id_textctx: str,
    official_dtype: str,
    allow_small_official_checkpoint: bool,
    load_weights: bool = True,
) -> Tuple[nn.Module, Path]:
    WanTransformer3DModel = _import_official_wan_textctx()
    torch_dtype = _torch_dtype_from_string(official_dtype)
    local_root, selected_path = _resolve_official_checkpoint(
        pretrained_path,
        official_repo_id_textctx,
        backend_label="official base Wan DiT",
    )
    if load_weights:
        _check_official_checkpoint_size(
            selected_path,
            allow_small_official_checkpoint,
            backend_label="Wan2.1 base 1.3B",
        )

    selected_parent = selected_path.parent
    config_path = selected_parent / "config.json"
    config: Dict = {}
    if config_path.exists():
        with config_path.open("r") as f:
            config = json.load(f)

    if _looks_like_vace_config(config):
        raise RuntimeError(
            "wan_backend='official_wan_1_3b_textctx' requested a base Wan "
            f"transformer, but the selected checkpoint config is VACE/control-style: "
            f"{config_path}. Use wan_backend='official_vace_1_3b' for that checkpoint "
            f"or set official_repo_id_textctx/pretrained_path to {OFFICIAL_WAN_TEXTCTX_REPO_ID}."
        )

    if config.get("_class_name") == "WanTransformer3DModel":
        if load_weights:
            transformer = WanTransformer3DModel.from_pretrained(
                str(selected_parent),
                torch_dtype=torch_dtype,
            )
        else:
            transformer = WanTransformer3DModel.from_config(config)
            transformer = transformer.to(torch_dtype)
        return transformer, selected_path

    transformer_dir = local_root / "transformer"
    transformer_config = transformer_dir / "config.json"
    if transformer_config.exists():
        with transformer_config.open("r") as f:
            transformer_cfg = json.load(f)
        if _looks_like_vace_config(transformer_cfg):
            raise RuntimeError(
                "wan_backend='official_wan_1_3b_textctx' must not instantiate VACE "
                f"transformer components, but found a VACE config at {transformer_config}."
            )
        if transformer_cfg.get("_class_name") == "WanTransformer3DModel":
            if load_weights:
                transformer = WanTransformer3DModel.from_pretrained(
                    str(local_root),
                    subfolder="transformer",
                    torch_dtype=torch_dtype,
                )
            else:
                transformer = WanTransformer3DModel.from_config(transformer_cfg)
                transformer = transformer.to(torch_dtype)
            return transformer, selected_path

    if config.get("_class_name") == "WanModel":
        cprint(
            "[WanVACEFutureImageHead] Loading original-format base Wan checkpoint "
            "with explicit 1.3B Diffusers config mapping.",
            "cyan",
        )
        transformer = WanTransformer3DModel(
            **_diffusers_base_config_from_original_wan_config(config)
        )
        if load_weights:
            _load_manual_official_state_dict(
                transformer,
                selected_path,
                key_converter=_convert_original_wan_transformer_key,
                checkpoint_label="Official base Wan",
                min_loaded_params=1_000_000_000,
            )
        transformer.to(dtype=torch_dtype)
        return transformer, selected_path

    if load_weights:
        try:
            transformer = WanTransformer3DModel.from_pretrained(
                str(selected_parent),
                torch_dtype=torch_dtype,
            )
            return transformer, selected_path
        except Exception as exc:
            raise RuntimeError(
                f"Could not instantiate official base Wan transformer from {selected_parent}. "
                "Install a recent diffusers version with WanTransformer3DModel support or "
                "provide a Diffusers-format checkpoint directory containing "
                "transformer/config.json and diffusion_pytorch_model*.safetensors."
            ) from exc
    else:
        raise RuntimeError(
            f"Could not find a valid config.json to instantiate official_wan_1_3b_textctx "
            f"with random weights from {selected_parent}. Ensure the checkpoint directory "
            "contains a transformer/config.json or top-level config.json."
        )


class OfficialWanVACEAdapter(nn.Module):
    """Adapter from the repo's FMPolicy head API to Diffusers Wan VACE.

    The current future-image target stores each camera latent as one row:
    ``[B, N_cam, C*H*W]``. Wan VACE is a video transformer over
    ``[B, C, F, H, W]`` latents, so this adapter treats cameras as a tiny frame
    axis: ``N_cam -> F``. The reverse reshape restores the public output shape.
    """

    def __init__(
        self,
        transformer: nn.Module,
        *,
        cond_dim: int,
        latent_channels: int,
        latent_h: int,
        latent_w: int,
        wan_num_train_timesteps: int,
        no_text_conditioning: bool,
    ):
        super().__init__()
        self.transformer = transformer
        self.latent_channels = latent_channels
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.wan_num_train_timesteps = wan_num_train_timesteps
        self.no_text_conditioning = no_text_conditioning
        cfg = getattr(transformer, "config", {})
        cfg_get = cfg.get if hasattr(cfg, "get") else lambda key, default=None: default
        self.text_dim = int(getattr(cfg, "text_dim", cfg_get("text_dim", 4096)))
        self.in_channels = int(
            getattr(cfg, "in_channels", cfg_get("in_channels", latent_channels))
        )
        self.vace_in_channels = int(
            getattr(cfg, "vace_in_channels", cfg_get("vace_in_channels", 96))
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, self.text_dim),
            nn.SiLU(),
            nn.Linear(self.text_dim, self.text_dim),
        )
        self._debug_logged = False

        if self.in_channels != latent_channels:
            raise ValueError(
                f"Official Wan VACE transformer in_channels={self.in_channels} does not "
                f"match latent_channels={latent_channels}."
            )

    def freeze_transformer(self) -> None:
        for p in self.transformer.parameters():
            p.requires_grad = False

    def _wan_timesteps(self, t: torch.Tensor) -> torch.Tensor:
        # Repo flow matching samples normalized t in (0, 1]. Diffusers Wan VACE
        # expects scheduler timesteps on the training scale, so map 1.0 -> 1000
        # and near-clean values toward 0.
        return (t.float().clamp(0.0, 1.0) * self.wan_num_train_timesteps).long()

    def forward(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        cond_features: torch.Tensor,
        N_cam: Optional[int] = None,
    ) -> torch.Tensor:
        B, N, D_lat = x_noisy.shape
        if N_cam is None:
            N_cam = N
        assert N == N_cam, f"x_noisy N_cam={N} != expected N_cam={N_cam}"
        C, H, W = self.latent_channels, self.latent_h, self.latent_w
        assert D_lat == C * H * W, f"cam_lat_dim={D_lat} != {C}*{H}*{W}={C * H * W}"

        transformer_dtype = next(self.transformer.parameters()).dtype
        latents = x_noisy.reshape(B, N_cam, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
        latents = latents.to(dtype=transformer_dtype)

        # HPT future tokens replace text encoder output. no_text_conditioning=True
        # means no T5 is used; the projected HPT context still conditions Wan.
        context = self.cond_proj(cond_features).to(
            device=latents.device,
            dtype=transformer_dtype,
        )

        # VACE's control branch is mandatory in the official transformer. This
        # head has no image/mask control input, so provide a zero control volume
        # with the official vace_in_channels while HPT context drives generation.
        control = torch.zeros(
            B,
            self.vace_in_channels,
            N_cam,
            H,
            W,
            device=latents.device,
            dtype=transformer_dtype,
        )
        wan_timestep = self._wan_timesteps(t).to(latents.device)

        output = self.transformer(
            hidden_states=latents,
            timestep=wan_timestep,
            encoder_hidden_states=context,
            control_hidden_states=control,
            return_dict=False,
        )
        if isinstance(output, tuple):
            output_latents = output[0]
        else:
            output_latents = output.sample

        out = output_latents.to(dtype=x_noisy.dtype).permute(0, 2, 1, 3, 4).contiguous()
        out = out.reshape(B, N_cam, C * H * W)

        if os.environ.get("WAN_DEBUG") == "1" and not self._debug_logged:
            cprint(
                "[WanVACEFutureImageHead] WAN_DEBUG shapes: "
                f"features={tuple(cond_features.shape)}  "
                f"target/noisy={tuple(x_noisy.shape)}  "
                f"official_latents={tuple(latents.shape)}  "
                f"context={tuple(context.shape)}  "
                f"transformer_output={tuple(output_latents.shape)}  "
                f"final={tuple(out.shape)}",
                "blue",
            )
            self._debug_logged = True

        assert out.shape == (B, N_cam, C * H * W)
        return out


class OfficialWanTextContextAdapter(nn.Module):
    """Adapter from FMPolicy tensors to the base Diffusers Wan transformer.

    This backend intentionally avoids VACE/control modules. HPT future tokens
    are projected into Wan's text/context dimension and passed directly as
    ``encoder_hidden_states`` in place of T5 encoder outputs.

    The future-image target stores one flattened latent per camera:
    ``[B, N_cam, C*H*W]``. Base Wan is a video transformer over
    ``[B, C, F, H, W]`` latents, so cameras are temporarily treated as the
    frame axis: ``N_cam -> F``. The reverse reshape restores the public output
    shape expected by FMPolicy.
    """

    def __init__(
        self,
        transformer: nn.Module,
        *,
        cond_dim: int,
        latent_channels: int,
        latent_h: int,
        latent_w: int,
        wan_num_train_timesteps: int,
    ):
        super().__init__()
        self.transformer = transformer
        self.latent_channels = latent_channels
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.wan_num_train_timesteps = wan_num_train_timesteps
        cfg = getattr(transformer, "config", {})
        cfg_get = cfg.get if hasattr(cfg, "get") else lambda key, default=None: default
        self.text_dim = int(getattr(cfg, "text_dim", cfg_get("text_dim", 4096)))
        self.in_channels = int(
            getattr(cfg, "in_channels", cfg_get("in_channels", latent_channels))
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, self.text_dim),
            nn.SiLU(),
            nn.Linear(self.text_dim, self.text_dim),
        )
        self._debug_logged = False

        if self.in_channels != latent_channels:
            raise ValueError(
                f"Official base Wan transformer in_channels={self.in_channels} does not "
                f"match latent_channels={latent_channels}."
            )

    def freeze_transformer(self) -> None:
        for p in self.transformer.parameters():
            p.requires_grad = False

    def _wan_timesteps(self, t: torch.Tensor) -> torch.Tensor:
        # Repo flow matching samples normalized t in (0, 1]. Diffusers Wan
        # expects scheduler timesteps on the training scale, so map 1.0 -> 1000
        # and near-clean values toward 0. The scale is configurable from Hydra.
        return (t.float().clamp(0.0, 1.0) * self.wan_num_train_timesteps).long()

    def forward(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        cond_features: torch.Tensor,
        N_cam: Optional[int] = None,
    ) -> torch.Tensor:
        B, N, D_lat = x_noisy.shape
        if N_cam is None:
            N_cam = N
        assert N == N_cam, f"x_noisy N_cam={N} != expected N_cam={N_cam}"
        C, H, W = self.latent_channels, self.latent_h, self.latent_w
        assert D_lat == C * H * W, f"cam_lat_dim={D_lat} != {C}*{H}*{W}={C * H * W}"

        transformer_dtype = next(self.transformer.parameters()).dtype
        # Cameras are represented as the video frame axis for the base Wan DiT.
        latents = x_noisy.reshape(B, N_cam, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
        latents = latents.to(dtype=transformer_dtype)

        # HPT future tokens replace T5 encoder hidden states. No tokenizer,
        # text encoder, VACE branch, or zero control volume is constructed.
        encoder_hidden_states = self.cond_proj(cond_features).to(
            device=latents.device,
            dtype=transformer_dtype,
        )
        wan_timestep = self._wan_timesteps(t).to(latents.device)

        output = self.transformer(
            hidden_states=latents,
            timestep=wan_timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )
        if isinstance(output, tuple):
            output_latents = output[0]
        else:
            output_latents = output.sample

        out = output_latents.to(dtype=x_noisy.dtype).permute(0, 2, 1, 3, 4).contiguous()
        out = out.reshape(B, N_cam, C * H * W)

        if os.environ.get("WAN_DEBUG") == "1" and not self._debug_logged:
            cprint(
                "[WanVACEFutureImageHead] WAN_DEBUG shapes: "
                f"features={tuple(cond_features.shape)}  "
                f"target/noisy={tuple(x_noisy.shape)}  "
                f"wan_latents={tuple(latents.shape)}  "
                f"encoder_hidden_states={tuple(encoder_hidden_states.shape)}  "
                f"transformer_output={tuple(output_latents.shape)}  "
                f"final={tuple(out.shape)}",
                "blue",
            )
            self._debug_logged = True

        assert out.shape == (B, N_cam, C * H * W)
        return out


# ---------------------------------------------------------------------------
# Main head (FMPolicy-compatible interface)
# ---------------------------------------------------------------------------


class WanVACEFutureImageHead(nn.Module):
    """Flow-matching head that predicts future image latents using a VACE-style DiT.

    Compatible with the existing FMPolicy / DenoisingPolicy interface used by
    ``HPTModel._compute_joint_state_loss``:

        head.compute_loss(features, aux_data)
            features  : [B, T_cond, cond_dim]   HPT future tokens (no pooling)
            aux_data  : dict  with key "action" holding target latents [B, N_cam, C*H*W]

        head(global_cond)
            global_cond may be a (features, domain_str) tuple (when HPT diffusion=True)
            or a raw tensor  [B, T_cond, cond_dim].
            Returns predicted latents  [B, N_cam, C*H*W].

    Shape assertions are placed at key points to catch mis-wired configs early.
    """

    def __init__(
        self,
        # --- conditioning ---
        cond_dim: int = 256,
        future_horizon: int = 16,
        # --- latent geometry ---
        latent_channels: int = 16,
        latent_h: int = 16,
        latent_w: int = 16,
        image_size: int = 128,
        # --- DiT architecture ---
        patch_size: int = 2,
        hidden_dim: int = 768,
        n_blocks: int = 12,
        n_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        time_dim: int = 256,
        # --- sampling ---
        num_inference_steps: int = 50,
        prediction_type: str = "v_prediction",
        time_dist: str = "beta",
        # --- cameras ---
        future_image_keys: Optional[List[str]] = None,
        max_cameras: int = 4,
        no_text_conditioning: bool = True,
        # --- pretrained loading ---
        wan_backend: str = "custom",
        official_repo_id: str = OFFICIAL_WAN_REPO_ID,
        official_repo_id_textctx: str = OFFICIAL_WAN_TEXTCTX_REPO_ID,
        official_dtype: str = "bf16",
        wan_num_train_timesteps: int = 1000,
        strict_pretrained_load: bool = True,
        load_pretrained: bool = False,
        pretrained_path: str = "",
        load_main_blocks_only: bool = True,
        freeze_pretrained_main_blocks: bool = False,
        allow_small_official_checkpoint: bool = False,
        gradient_checkpointing: bool = False,
        # --- compatibility shim: DenoisingPolicy uses infer_ac_dims ---
        infer_ac_dims: Optional[Dict[str, int]] = None,
        # catch-all for extra hydra keys (e.g. _target_)
        **kwargs,
    ):
        super().__init__()

        if prediction_type not in ("v_prediction", "x_prediction"):
            raise ValueError(
                f"prediction_type must be 'v_prediction' or 'x_prediction', got {prediction_type!r}"
            )

        self.cond_dim = cond_dim
        self.future_horizon = future_horizon
        self.latent_channels = latent_channels
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.cam_lat_dim = latent_channels * latent_h * latent_w
        self.patch_size = patch_size
        self.num_inference_steps = num_inference_steps
        self.prediction_type = prediction_type
        self.time_dist = time_dist
        self.future_image_keys = list(future_image_keys or ["future_front_img_1"])
        self.max_cameras = max_cameras
        self.wan_backend = wan_backend
        self.official_repo_id = official_repo_id
        self.official_repo_id_textctx = official_repo_id_textctx
        self.official_dtype = official_dtype
        self.wan_num_train_timesteps = wan_num_train_timesteps
        self.strict_pretrained_load = strict_pretrained_load

        if wan_backend not in (
            "custom",
            "official_vace_1_3b",
            "official_wan_1_3b_textctx",
        ):
            raise ValueError(
                "wan_backend must be 'custom', 'official_vace_1_3b', or "
                f"'official_wan_1_3b_textctx', got {wan_backend!r}"
            )
        cprint(f"[WanVACEFutureImageHead] Backend: {wan_backend}", "cyan")

        # Compatibility shim so _default_build_aux_cond / DenoisingPolicy dispatch works.
        self.infer_ac_dims = infer_ac_dims or {}
        self.action_horizon = 1
        self.pooling = None  # no pooling — we use all future tokens
        self._debug_loss_logged = False

        if wan_backend == "custom":
            self.model = WanVACEDiT(
                cond_dim=cond_dim,
                future_horizon=future_horizon,
                latent_channels=latent_channels,
                latent_h=latent_h,
                latent_w=latent_w,
                patch_size=patch_size,
                hidden_dim=hidden_dim,
                n_blocks=n_blocks,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                time_dim=time_dim,
                max_cameras=max_cameras,
            )
            cprint(
                f"[WanVACEFutureImageHead] Using custom WanVACEDiT backend "
                f"(hidden={hidden_dim}, n_blocks={n_blocks}, heads={n_heads}, "
                f"latent={latent_channels}x{latent_h}x{latent_w}, patch_size={patch_size})",
                "cyan",
            )

            if load_pretrained and pretrained_path:
                _load_wan_pretrained(
                    self.model,
                    pretrained_path,
                    load_main_blocks_only,
                    freeze_pretrained_main_blocks,
                    strict_pretrained_load=strict_pretrained_load,
                )
        elif wan_backend == "official_vace_1_3b":
            if not load_pretrained:
                raise ValueError(
                    "wan_backend='official_vace_1_3b' requires load_pretrained=True "
                    "so the real Wan2.1-VACE-1.3B DiT weights are loaded."
                )
            official_source = pretrained_path or f"hf://{official_repo_id}"
            transformer, selected_checkpoint = _load_official_wan_vace_transformer(
                pretrained_path=official_source,
                official_repo_id=official_repo_id,
                official_dtype=official_dtype,
                allow_small_official_checkpoint=allow_small_official_checkpoint,
            )
            transformer_params = _count_params(transformer)
            cprint(
                "[WanVACEFutureImageHead] Using official Wan2.1-VACE-1.3B backend",
                "cyan",
            )
            cprint(
                f"[WanVACEFutureImageHead] Selected checkpoint path: {selected_checkpoint}",
                "cyan",
            )
            cprint(
                f"[WanVACEFutureImageHead] Transformer parameters: "
                f"{_format_param_count(transformer_params)}",
                "cyan",
            )
            if transformer_params < 1_000_000_000:
                raise RuntimeError(
                    "official_vace_1_3b requested but instantiated model has only "
                    f"{transformer_params} params; this is not the 1.3B Wan VACE transformer."
                )

            self.model = OfficialWanVACEAdapter(
                transformer,
                cond_dim=cond_dim,
                latent_channels=latent_channels,
                latent_h=latent_h,
                latent_w=latent_w,
                wan_num_train_timesteps=wan_num_train_timesteps,
                no_text_conditioning=no_text_conditioning,
            )
            if freeze_pretrained_main_blocks:
                self.model.freeze_transformer()
        else:
            if not load_pretrained:
                cprint(
                    "[WanVACEFutureImageHead] load_pretrained=False with official_wan_1_3b_textctx: "
                    "instantiating Wan2.1-1.3B architecture with RANDOM weights (ablation mode).",
                    "yellow",
                )
            official_source = pretrained_path or f"hf://{official_repo_id_textctx}"
            transformer, selected_checkpoint = _load_official_wan_textctx_transformer(
                pretrained_path=official_source,
                official_repo_id_textctx=official_repo_id_textctx,
                official_dtype=official_dtype,
                allow_small_official_checkpoint=allow_small_official_checkpoint,
                load_weights=load_pretrained,
            )
            transformer_params = _count_params(transformer)
            cprint(
                "[WanVACEFutureImageHead] Using official base Wan2.1-1.3B text-context backend",
                "cyan",
            )
            cprint(
                f"[WanVACEFutureImageHead] Selected checkpoint path: {selected_checkpoint}",
                "cyan",
            )
            cprint(
                f"[WanVACEFutureImageHead] Transformer parameters: "
                f"{_format_param_count(transformer_params)}",
                "cyan",
            )
            if transformer_params <= 1_000_000_000:
                raise RuntimeError(
                    "official_wan_1_3b_textctx requested but instantiated model has only "
                    f"{transformer_params} params; this is not the base 1.3B Wan transformer."
                )
            if transformer_params >= 1_800_000_000:
                raise RuntimeError(
                    "official_wan_1_3b_textctx requested but instantiated model has "
                    f"{transformer_params} params; this is likely the VACE/control "
                    "transformer, not the base Wan 1.3B transformer."
                )

            self.model = OfficialWanTextContextAdapter(
                transformer,
                cond_dim=cond_dim,
                latent_channels=latent_channels,
                latent_h=latent_h,
                latent_w=latent_w,
                wan_num_train_timesteps=wan_num_train_timesteps,
            )
            if freeze_pretrained_main_blocks:
                self.model.freeze_transformer()

        if gradient_checkpointing:
            self._enable_gradient_checkpointing()

        total_params = _count_params(self.model)
        trainable_params = _count_params(self.model, trainable_only=True)
        cprint(
            f"[WanVACEFutureImageHead] Total parameters: {_format_param_count(total_params)}  "
            f"Trainable parameters: {_format_param_count(trainable_params)}",
            "cyan",
        )

    def _enable_gradient_checkpointing(self) -> None:
        """Recompute Wan block activations in the backward pass instead of storing them.

        Memory, not maths: the objective is unchanged, each block is just re-run once
        during backward (~30% slower). Needed because the fp32 weights, gradients and
        Adam moments of the 1.3B transformer already occupy ~21 GiB of a 44 GiB A40
        before a single activation is stored -- and a cotrain step pushes TWO domains
        through this one head before backward, so activations are held twice over.
        """
        transformer = getattr(self.model, "transformer", None)
        if transformer is not None and hasattr(
            transformer, "enable_gradient_checkpointing"
        ):
            transformer.enable_gradient_checkpointing()
            cprint(
                "[WanVACEFutureImageHead] Gradient checkpointing ENABLED on the Wan "
                "transformer blocks.",
                "cyan",
            )
            return
        cprint(
            "[WanVACEFutureImageHead] gradient_checkpointing=true requested but the "
            f"backend ({self.wan_backend}) exposes no enable_gradient_checkpointing(); "
            "continuing without it.",
            "yellow",
        )

    # ------------------------------------------------------------------
    # Flow-matching helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _expand_t(t: torch.Tensor, ndim: int) -> torch.Tensor:
        """Broadcast [B] scalar time to ndim-dimensional tensor."""
        te = t
        while te.ndim < ndim:
            te = te.unsqueeze(-1)
        return te

    def _sample_time(self, B: int, device: torch.device) -> torch.Tensor:
        if self.time_dist == "beta":
            a, b = 1.5, 1.0
            t = torch.distributions.Beta(a, b).sample((B,)).to(device)
        else:
            t = torch.distributions.Uniform(0, 1).sample((B,)).to(device)
        return (t * 0.999 + 0.001).float()

    def _forward_model(
        self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, N_cam: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run model and convert to v_prediction if needed.

        Returns (pred_v, t_expanded) where pred_v is the flow velocity.
        """
        t_exp = self._expand_t(t, x_t.ndim)
        if self.prediction_type == "v_prediction":
            pred_v = self.model(x_t, t, cond, N_cam)
        else:
            x_theta = self.model(x_t, t, cond, N_cam)
            pred_v = (x_t - x_theta) / t_exp.clamp(min=1e-3)
        return pred_v, t_exp

    # ------------------------------------------------------------------
    # Training: compute_loss(features, aux_data)
    # ------------------------------------------------------------------

    def compute_loss(self, features: torch.Tensor, aux_data: dict) -> torch.Tensor:
        """Flow-matching loss.

        Parameters
        ----------
        features  : [B, T_cond, cond_dim]   HPT future tokens
        aux_data  : dict  —  aux_data["action"] = target latents [B, N_cam, C*H*W]

        Returns
        -------
        scalar loss tensor
        """
        assert (
            features.ndim == 3
        ), f"WanVACEFutureImageHead.compute_loss expects features [B, T_cond, D], got {features.shape}"
        target = aux_data["action"]  # [B, N_cam, C*H*W]
        assert target.ndim == 3, f"target must be [B, N_cam, C*H*W], got {target.shape}"

        B, N_cam, _ = target.shape
        assert features.shape[0] == B, "batch size mismatch between features and target"

        if os.environ.get("WAN_DEBUG") == "1" and not self._debug_loss_logged:
            cprint(
                f"  [WanVACE loss] features={tuple(features.shape)}  "
                f"target={tuple(target.shape)}  N_cam={N_cam}",
                "blue",
            )
            self._debug_loss_logged = True

        noise = torch.randn_like(target)
        t = self._sample_time(B, features.device)  # [B]
        t_exp = self._expand_t(t, target.ndim)  # [B, 1, 1]

        x_t = t_exp * noise + (1 - t_exp) * target  # noisy latent
        u_t = noise - target  # ground-truth flow velocity

        pred_v, _ = self._forward_model(x_t, t, features, N_cam)

        return F.mse_loss(pred_v, u_t)

    # ------------------------------------------------------------------
    # Inference: denoising ODE integration
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, N_cam: Optional[int] = None) -> torch.Tensor:
        """Sample future image latents by integrating the flow ODE from t=1→0.

        Parameters
        ----------
        cond  : [B, T_cond, cond_dim]
        N_cam : int  (defaults to len(self.future_image_keys))

        Returns
        -------
        pred_latents : [B, N_cam, C*H*W]
        """
        if N_cam is None:
            N_cam = len(self.future_image_keys)

        B = cond.shape[0]
        x = torch.randn(
            B,
            N_cam,
            self.cam_lat_dim,
            device=cond.device,
            dtype=cond.dtype,
        )

        dt = -1.0 / self.num_inference_steps
        t = torch.ones(B, device=cond.device, dtype=cond.dtype)

        while t[0] >= -dt / 2:
            pred_v, _ = self._forward_model(x, t, cond, N_cam)
            x = x + dt * pred_v
            t = t + dt

        cprint(
            f"  [WanVACE sample] cond={tuple(cond.shape)}  out={tuple(x.shape)}",
            "blue",
        ) if os.environ.get("WAN_DEBUG") else None
        return x

    # ------------------------------------------------------------------
    # Single-step denoising diagnostic (NOT teacher forcing / rollout)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def single_step_denoise_eval(
        self,
        cond: torch.Tensor,
        target_latents: torch.Tensor,
        t_noise_level: float = 0.5,
    ) -> torch.Tensor:
        """One-step denoising from a noisy GT latent — a diagnostic tool only.

        This is NOT the sequential rollout evaluation.  It takes the ground-truth
        latent, adds noise at ``t_noise_level``, runs the model once, and
        estimates the clean latent.  It is useful for checking whether the
        denoising network has learned to invert flow noise, but it is NOT a
        simulation of how the model is used at inference time (which always starts
        from pure Gaussian noise via ``sample()``).

        The predicted clean latent:
            x0_pred = x_t - t * v_pred   (one-step flow reversal)

        Parameters
        ----------
        cond           : [B, T_cond, cond_dim]
        target_latents : [B, N_cam, C*H*W]   ground-truth latents
        t_noise_level  : float in (0, 1]     noise level to condition on

        Returns
        -------
        pred_x0 : [B, N_cam, C*H*W]
        """
        B, N_cam, _ = target_latents.shape
        noise = torch.randn_like(target_latents)
        t = torch.full((B,), t_noise_level, device=cond.device, dtype=cond.dtype)
        t_exp = self._expand_t(t, target_latents.ndim)

        x_t = t_exp * noise + (1 - t_exp) * target_latents

        pred_v, _ = self._forward_model(x_t, t, cond, N_cam)

        # Predicted clean latent: x0 = x_t - t * v
        pred_x0 = x_t - t_exp * pred_v
        return pred_x0

    # ------------------------------------------------------------------
    # forward() — FMPolicy / DenoisingPolicy-compatible eval entry point
    # ------------------------------------------------------------------

    def forward(self, global_cond) -> torch.Tensor:
        """Sample future image latents.

        Parameters
        ----------
        global_cond : (Tensor, str) | Tensor
            When HPT is configured with ``diffusion=True``, the caller wraps the
            conditioning as ``(future_feats, domain_name)``.  This method unpacks
            both forms.

        Returns
        -------
        [B, N_cam, C*H*W]
        """
        if isinstance(global_cond, tuple):
            cond, _domain = global_cond
        else:
            cond = global_cond

        assert cond.ndim == 3, (
            f"WanVACEFutureImageHead.forward expects features [B, T_cond, D], got {cond.shape}. "
            "Make sure pooling is NOT set for this head."
        )
        N_cam = len(self.future_image_keys)
        return self.sample(cond, N_cam=N_cam)


def _smoke_test_official_vace() -> None:
    """Fail-fast check that the official backend really loads the 1.3B DiT."""
    head = WanVACEFutureImageHead(
        wan_backend="official_vace_1_3b",
        load_pretrained=True,
        pretrained_path="hf://Wan-AI/Wan2.1-VACE-1.3B",
        cond_dim=256,
        latent_channels=16,
        latent_h=16,
        latent_w=16,
    )
    total_params = _count_params(head)
    print(f"Total params: {_format_param_count(total_params)}")
    features = torch.randn(1, 16, 256)
    target = torch.randn(1, 1, 4096)
    loss = head.compute_loss(features, {"action": target})
    print(f"Smoke loss: {loss.item():.6f}")


if __name__ == "__main__":
    _smoke_test_official_vace()
