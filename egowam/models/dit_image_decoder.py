"""
dit_image_decoder.py

Diffusion Transformer (DiT) backbone for image latent flow matching — drop-in
replacement for :class:`fm_image_decoder.FMImageDecoder`.

Patchifies each camera's VAE latent [C, H, W], runs AdaLN-Zero transformer blocks
(Peebles & Xie), then unpatchifies. Same ``forward(x, timesteps, cond)`` contract
as the conv decoder.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: float = 10_000.0,
) -> torch.Tensor:
    """Scalar timesteps [B] -> [B, dim] sinusoidal embedding."""
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


def _get_1d_sincos(
    dim: int,
    positions: torch.Tensor,
) -> torch.Tensor:
    """positions: [L] integer positions -> [L, dim] with sin/cos bands."""
    assert dim % 2 == 0
    half = dim // 2
    omega = torch.arange(half, dtype=torch.float32, device=positions.device)
    omega = 1.0 / (10000 ** (omega / half))
    out = positions.float().unsqueeze(1) * omega.unsqueeze(0)
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)
    return emb


def build_2d_sincos_pos_embed(
    embed_dim: int,
    grid_h: int,
    grid_w: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Fixed 2D sin-cos positional embedding for a patch grid [grid_h, grid_w].

    Returns
    -------
    Tensor of shape [grid_h * grid_w, embed_dim]
    """
    assert embed_dim % 2 == 0, "embed_dim must be even for 2D sincos"
    half = embed_dim // 2
    emb_h = _get_1d_sincos(
        half, torch.arange(grid_h, device=device, dtype=torch.float32)
    )
    emb_w = _get_1d_sincos(
        half, torch.arange(grid_w, device=device, dtype=torch.float32)
    )
    # [grid_h, grid_w, embed_dim] — concat h- and w-axes (DiT / ViT style)
    grid = torch.cat(
        [
            emb_h[:, None, :].expand(grid_h, grid_w, -1),
            emb_w[None, :, :].expand(grid_h, grid_w, -1),
        ],
        dim=-1,
    )
    return grid.reshape(grid_h * grid_w, embed_dim).to(dtype=dtype)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive shift/scale after LayerNorm (DiT-style)."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PatchEmbed(nn.Module):
    """Conv patchify: [B, C, H, W] -> [B, S, D] with S = (H/ps)*(W/ps)."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, D, H', W']
        B, D, _, _ = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, S, D]
        return x


def unpatchify(
    x: torch.Tensor,
    *,
    channels: int,
    patch_size: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    [B, S, patch_dim] -> [B, C, H, W] with patch_dim = C * patch_size^2.

    Patches are laid out in row-major order over the latent grid.
    """
    gh, gw = height // patch_size, width // patch_size
    B, S, pd = x.shape
    assert S == gh * gw
    assert pd == channels * patch_size * patch_size
    p = patch_size
    x = x.reshape(B, gh, gw, channels, p, p)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    return x.reshape(B, channels, gh * p, gw * p)


class DiTBlock(nn.Module):
    """Transformer block with AdaLN-Zero (Peebles & Xie)."""

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_dim),
        )
        self.adaLN_modulation = nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # c: [B_sub, hidden_dim] conditioning per (batch * camera)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(h, h, h)
        x = x + gate_msa.unsqueeze(1) * attn_out
        h2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h2)
        return x


class FinalLayer(nn.Module):
    """Last AdaLN + linear to patch pixels (zero-init linear for identity start)."""

    def __init__(self, hidden_dim: int, patch_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, patch_dim)
        self.adaLN_modulation = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiTImageDecoder(nn.Module):
    """
    DiT-style velocity network for per-camera VAE latents.

    Same interface as :class:`egowam.models.fm_image_decoder.FMImageDecoder`::

        v_t = model(x_t, timesteps, cond)
        # x_t       : [B, N_cams, cam_lat_dim]
        # timesteps : [B]
        # cond      : [B, N_cams, obs_dim] | [B, obs_dim] | [B, seq, obs_dim]
        # returns   : [B, N_cams, cam_lat_dim]
    """

    def __init__(
        self,
        lat_channels: int = 16,
        lat_h: int = 16,
        lat_w: int = 16,
        patch_size: int = 2,
        hidden_dim: int = 384,
        n_blocks: int = 6,
        n_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        obs_dim: int = 256,
        time_dim: int = 256,
        max_cameras: int = 3,
    ):
        super().__init__()
        self.lat_channels = lat_channels
        self.lat_h = lat_h
        self.lat_w = lat_w
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim

        gh = lat_h // patch_size
        gw = lat_w // patch_size
        if lat_h % patch_size != 0 or lat_w % patch_size != 0:
            raise ValueError(
                f"lat_h={lat_h}, lat_w={lat_w} must be divisible by patch_size={patch_size}"
            )
        self.grid_h = gh
        self.grid_w = gw
        self.num_patches = gh * gw
        self.patch_dim = lat_channels * patch_size * patch_size

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.max_cameras = max_cameras
        self.camera_embed = nn.Embedding(max_cameras, time_dim)
        nn.init.zeros_(self.camera_embed.weight)

        self.cond_proj = nn.Sequential(
            nn.Linear(obs_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        cond_in_dim = 2 * time_dim
        self.cond_vec_mlp = nn.Sequential(
            nn.Linear(cond_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.patch_embed = PatchEmbed(lat_channels, hidden_dim, patch_size)
        self.register_buffer("pos_embed", torch.zeros(self.num_patches, hidden_dim))
        self.register_buffer("_pos_init_done", torch.tensor(False))

        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_dim, n_heads, mlp_ratio, dropout) for _ in range(n_blocks)]
        )
        self.final_layer = FinalLayer(hidden_dim, self.patch_dim)

    def _init_pos_embed(self, device: torch.device, dtype: torch.dtype) -> None:
        pe = build_2d_sincos_pos_embed(
            self.hidden_dim, self.grid_h, self.grid_w, device, dtype
        )
        self.pos_embed.copy_(pe)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        cond: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        B, N, D = x.shape
        C, H, W = self.lat_channels, self.lat_h, self.lat_w
        assert D == C * H * W, f"Expected cam_lat_dim {C*H*W}, got {D}"

        if cond.ndim == 2:
            cond = cond.unsqueeze(1).expand(B, N, -1)
        elif cond.ndim == 3 and cond.shape[1] != N:
            cond = cond.mean(dim=1, keepdim=True).expand(B, N, -1)

        x_spatial = x.reshape(B * N, C, H, W)

        t_emb = sinusoidal_embedding(timesteps, self.time_dim)
        t_emb = self.time_mlp(t_emb)
        t_emb = t_emb[:, None, :].expand(B, N, -1).reshape(B * N, -1)

        cam_ids = (
            torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1).reshape(B * N)
        )
        cam_emb = self.camera_embed(cam_ids)
        t_emb = t_emb + cam_emb

        c_emb = self.cond_proj(cond).reshape(B * N, -1)
        combined = torch.cat([t_emb, c_emb], dim=-1)
        c_vec = self.cond_vec_mlp(combined)

        if not self._pos_init_done:
            self._init_pos_embed(x.device, x.dtype)
            self._pos_init_done.fill_(True)

        h = self.patch_embed(x_spatial)
        h = h + self.pos_embed.unsqueeze(0)

        for block in self.blocks:
            h = block(h, c_vec)

        out_tokens = self.final_layer(h, c_vec)
        out_spatial = unpatchify(
            out_tokens,
            channels=C,
            patch_size=self.patch_size,
            height=H,
            width=W,
        )
        out_spatial = out_spatial + x_spatial
        return out_spatial.reshape(B, N, D)
