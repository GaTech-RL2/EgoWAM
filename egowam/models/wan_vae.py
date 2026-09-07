"""
wan_vae.py

Self-contained Wan image-VAE wrapper adapted from the official Wan2.1 VAE
implementation. Unlike the old wrapper, this version does not depend on the
empty / optional ``external/cosmos-policy`` checkout or the ``megatron``
runtime stack.

Public contract:
- ``encode(videos_bcthw) -> latents_bcthw``
- ``decode(latents_bcthw) -> videos_bcthw``
- ``encode_image(images_bchw) -> latents_bchw``
- ``decode_image(latents_bchw) -> images_bchw``

Input images / videos are accepted in ``[0, 1]`` (or ``[0, 255]``) and are
internally mapped to the official Wan VAE input range ``[-1, 1]``.
Decoded outputs are mapped back to ``[0, 1]``.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from huggingface_hub import hf_hub_download

LOG = logging.getLogger(__name__)

CACHE_T = 2
DEFAULT_WAN_VAE_FILE = "Wan2.1_VAE.pth"
WAN_LATENT_MEAN = [
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
]
WAN_LATENT_STD = [
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.9160,
]


def _resolve_dtype(dtype: str | torch.dtype | None) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype is None or dtype == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported Wan VAE compute dtype: {dtype!r}")
    return mapping[dtype]


def _resolve_pretrained_path(pretrained_path: Optional[str]) -> Optional[str]:
    if not pretrained_path:
        return None

    if pretrained_path.startswith("hf://"):
        spec = pretrained_path[len("hf://") :].strip("/")
        if spec.endswith(".pth") or spec.endswith(".pt") or spec.endswith(".bin"):
            repo_id, filename = spec.rsplit("/", 1)
        else:
            repo_id, filename = spec, DEFAULT_WAN_VAE_FILE
        local_path = hf_hub_download(repo_id=repo_id, filename=filename)
        LOG.info("Resolved Wan VAE checkpoint %s -> %s", pretrained_path, local_path)
        return local_path

    path = Path(pretrained_path).expanduser()
    if path.is_dir():
        direct = path / DEFAULT_WAN_VAE_FILE
        if direct.exists():
            return str(direct)
        matches = list(path.rglob(DEFAULT_WAN_VAE_FILE))
        if matches:
            return str(matches[0])
        raise FileNotFoundError(
            f"Could not find {DEFAULT_WAN_VAE_FILE} under directory: {path}"
        )

    if path.exists():
        return str(path)

    raise FileNotFoundError(f"Wan VAE checkpoint not found: {pretrained_path}")


def _maybe_normalize_to_minus1_1(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    if x.max() > 1.5:
        x = x / 255.0
    if x.min() < -0.05:
        return x.clamp(-1.0, 1.0)
    return x.clamp(0.0, 1.0).mul(2.0).sub(1.0)


def _maybe_denormalize_from_minus1_1(x: torch.Tensor) -> torch.Tensor:
    return x.float().clamp(-1.0, 1.0).add(1.0).div(2.0)


def _autocast_ctx(device_type: str, dtype: torch.dtype):
    if device_type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


class CausalConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0,
        )
        self.padding = (0, 0, 0)

    def forward(self, x: torch.Tensor, cache_x: Optional[torch.Tensor] = None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return super().forward(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, channel_first: bool = True, images: bool = True):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = 1 if self.channel_first else -1
        return F.normalize(x, dim=dim) * self.scale * self.gamma


class Upsample(nn.Upsample):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):
    def __init__(self, dim: int, mode: str):
        super().__init__()
        assert mode in (
            "none",
            "upsample2d",
            "upsample3d",
            "downsample2d",
            "downsample3d",
        )
        self.mode = mode
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)),
            )
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)),
            )
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
            )
        else:
            self.resample = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: Optional[list] = None,
        feat_idx: Optional[list[int]] = None,
    ) -> torch.Tensor:
        if feat_idx is None:
            feat_idx = [0]
        b, c, _t, _h, _w = x.size()
        if self.mode == "upsample3d" and feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = "Rep"
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] not in (None, "Rep"):
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :]
                            .unsqueeze(2)
                            .to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                if cache_x.shape[2] < 2 and feat_cache[idx] == "Rep":
                    cache_x = torch.cat(
                        [torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2
                    )
                x = self.time_conv(
                    x if feat_cache[idx] == "Rep" else x,
                    None if feat_cache[idx] == "Rep" else feat_cache[idx],
                )
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
                x = x.reshape(b, 2, c, x.shape[2], x.shape[3], x.shape[4])
                x = torch.stack((x[:, 0], x[:, 1]), dim=3)
                x = x.reshape(b, c, -1, x.shape[4], x.shape[5])

        t = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resample(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)

        if self.mode == "downsample3d" and feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = x.clone()
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -1:, :, :].clone()
                x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.residual = nn.Sequential(
            RMSNorm(in_dim, images=False),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMSNorm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = (
            CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: Optional[list] = None,
        feat_idx: Optional[list[int]] = None,
    ) -> torch.Tensor:
        if feat_idx is None:
            feat_idx = [0]
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :]
                            .unsqueeze(2)
                            .to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.norm(x)
        q, k, v = (
            self.to_qkv(x)
            .reshape(b * t, 1, c * 3, -1)
            .permute(0, 1, 3, 2)
            .contiguous()
            .chunk(3, dim=-1)
        )
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)
        x = self.proj(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
        return x + identity


class Encoder3d(nn.Module):
    def __init__(
        self,
        dim: int = 96,
        z_dim: int = 32,
        dim_mult: list[int] | tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales: list[float] | tuple[float, ...] = (),
        temperal_downsample: list[bool] | tuple[bool, ...] = (False, True, True),
        dropout: float = 0.0,
    ):
        super().__init__()
        dims = [dim * u for u in [1] + list(dim_mult)]
        scale = 1.0
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout),
        )
        self.head = nn.Sequential(
            RMSNorm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: Optional[list] = None,
        feat_idx: Optional[list[int]] = None,
    ) -> torch.Tensor:
        if feat_idx is None:
            feat_idx = [0]
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.downsamples:
            x = layer(x, feat_cache, feat_idx) if feat_cache is not None else layer(x)

        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :]
                            .unsqueeze(2)
                            .to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class Decoder3d(nn.Module):
    def __init__(
        self,
        dim: int = 96,
        z_dim: int = 16,
        dim_mult: list[int] | tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales: list[float] | tuple[float, ...] = (),
        temperal_upsample: list[bool] | tuple[bool, ...] = (True, True, False),
        dropout: float = 0.0,
    ):
        super().__init__()
        dims = [dim * u for u in [dim_mult[-1]] + list(dim_mult)[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout),
        )

        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i in (1, 2, 3):
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                mode = "upsample3d" if temperal_upsample[i] else "upsample2d"
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)
        self.head = nn.Sequential(
            RMSNorm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, 3, 3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: Optional[list] = None,
        feat_idx: Optional[list[int]] = None,
    ) -> torch.Tensor:
        if feat_idx is None:
            feat_idx = [0]
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.upsamples:
            x = layer(x, feat_cache, feat_idx) if feat_cache is not None else layer(x)

        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :]
                            .unsqueeze(2)
                            .to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


def _count_causal_conv3d(model: nn.Module) -> int:
    return sum(isinstance(module, CausalConv3d) for module in model.modules())


class _WanVAECore(nn.Module):
    def __init__(
        self,
        dim: int = 96,
        z_dim: int = 16,
        dim_mult: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales: tuple[float, ...] = (),
        temperal_downsample: tuple[bool, ...] = (False, True, True),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = Encoder3d(
            dim=dim,
            z_dim=z_dim * 2,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temperal_downsample=temperal_downsample,
            dropout=dropout,
        )
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(
            dim=dim,
            z_dim=z_dim,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temperal_upsample=tuple(reversed(temperal_downsample)),
            dropout=dropout,
        )
        self.clear_cache()

    def clear_cache(self) -> None:
        self._conv_num = _count_causal_conv3d(self.decoder)
        self._feat_map = [None] * self._conv_num
        self._enc_conv_num = _count_causal_conv3d(self.encoder)
        self._enc_feat_map = [None] * self._enc_conv_num

    def encode(
        self, x: torch.Tensor, scale: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        self.clear_cache()
        steps = 1 + (x.shape[2] - 1) // 4
        out = None
        for i in range(steps):
            feat_idx = [0]
            chunk = x[:, :, :1] if i == 0 else x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i]
            out_i = self.encoder(
                chunk, feat_cache=self._enc_feat_map, feat_idx=feat_idx
            )
            out = out_i if out is None else torch.cat([out, out_i], dim=2)
        mu, _ = self.conv1(out).chunk(2, dim=1)
        shift, inv_std = scale
        mu = (mu - shift.view(1, self.z_dim, 1, 1, 1)) * inv_std.view(
            1, self.z_dim, 1, 1, 1
        )
        self.clear_cache()
        return mu

    def decode(
        self, z: torch.Tensor, scale: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        self.clear_cache()
        shift, inv_std = scale
        z = z / inv_std.view(1, self.z_dim, 1, 1, 1) + shift.view(
            1, self.z_dim, 1, 1, 1
        )
        x = self.conv2(z)
        out = None
        for i in range(x.shape[2]):
            feat_idx = [0]
            out_i = self.decoder(
                x[:, :, i : i + 1], feat_cache=self._feat_map, feat_idx=feat_idx
            )
            out = out_i if out is None else torch.cat([out, out_i], dim=2)
        self.clear_cache()
        return out


def _build_wan_vae_model(
    *,
    checkpoint_path: Optional[str],
    z_dim: int,
) -> _WanVAECore:
    model = _WanVAECore(z_dim=z_dim)
    if checkpoint_path is not None:
        LOG.info("Loading Wan VAE weights from %s", checkpoint_path)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Wan VAE checkpoint mismatch: "
                f"missing={missing[:10]} unexpected={unexpected[:10]}"
            )
    return model


class WanImageVAEWrapper(nn.Module):
    def __init__(
        self,
        load_pretrained: bool = True,
        pretrained_path: Optional[str] = None,
        vae_pth: Optional[str] = None,
        freeze: bool = True,
        image_size: int = 128,
        z_dim: int = 16,
        compute_dtype: str | torch.dtype | None = "auto",
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.image_size = image_size
        self.freeze = freeze
        self.compute_dtype = _resolve_dtype(compute_dtype)

        resolved_path = pretrained_path or vae_pth
        checkpoint_path = (
            _resolve_pretrained_path(resolved_path) if load_pretrained else None
        )
        self.model = _build_wan_vae_model(checkpoint_path=checkpoint_path, z_dim=z_dim)
        self.register_buffer(
            "latent_mean", torch.tensor(WAN_LATENT_MEAN, dtype=torch.float32)
        )
        self.register_buffer(
            "latent_std", torch.tensor(WAN_LATENT_STD, dtype=torch.float32)
        )

        if freeze:
            self.model.requires_grad_(False)
            self.eval()

    def _scale_tensors(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.latent_mean.to(device=device, dtype=torch.float32)
        inv_std = (1.0 / self.latent_std).to(device=device, dtype=torch.float32)
        return mean, inv_std

    @torch.no_grad()
    def encode(self, videos_bcthw: torch.Tensor) -> torch.Tensor:
        videos_bcthw = _maybe_normalize_to_minus1_1(videos_bcthw)
        with _autocast_ctx(videos_bcthw.device.type, self.compute_dtype):
            return self.model.encode(
                videos_bcthw, self._scale_tensors(videos_bcthw.device)
            ).float()

    @torch.no_grad()
    def decode(self, latents_bcthw: torch.Tensor) -> torch.Tensor:
        with _autocast_ctx(latents_bcthw.device.type, self.compute_dtype):
            decoded = self.model.decode(
                latents_bcthw.float(), self._scale_tensors(latents_bcthw.device)
            )
        return _maybe_denormalize_from_minus1_1(decoded)

    def decode_with_grad(self, latents_bcthw: torch.Tensor) -> torch.Tensor:
        with _autocast_ctx(latents_bcthw.device.type, self.compute_dtype):
            decoded = self.model.decode(
                latents_bcthw.float(), self._scale_tensors(latents_bcthw.device)
            )
        return _maybe_denormalize_from_minus1_1(decoded)

    @torch.no_grad()
    def encode_image(self, images_bchw: torch.Tensor) -> torch.Tensor:
        assert (
            images_bchw.ndim == 4
        ), f"Expected [B,3,H,W], got {tuple(images_bchw.shape)}"
        return self.encode(images_bchw.unsqueeze(2))[:, :, 0]

    @torch.no_grad()
    def decode_image(self, latents_bchw: torch.Tensor) -> torch.Tensor:
        assert (
            latents_bchw.ndim == 4
        ), f"Expected [B,C,H,W], got {tuple(latents_bchw.shape)}"
        return self.decode(latents_bchw.unsqueeze(2))[:, :, 0]
