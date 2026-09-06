"""
Point-conditioned flow decoder used by PointFMPolicy for both 2D (point_dim=2)
and 3D (point_dim=3) flow heads.

Input:  x_t         Tensor [B, H, N, D]  (noisy normalized delta trajectory)
        time        Tensor [B]           (flow-matching time in [0,1])
        global_cond dict
            trunk_feats    Tensor [B, L_trunk, cond_dim]  (cross-attn context)
            current_points Tensor [B, N, D]               (anchor query)
Output: Tensor [B, H, N, D]  (predicted normalized delta trajectory)

See plans/3d_point_flow_supervision.md for the design.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from egowam.models.denoising_nets import CrossBlock, posemb_sincos


class PointFlowDecoder(nn.Module):
    def __init__(
        self,
        num_points: int = 512,
        point_dim: int = 3,
        action_horizon: int = 100,
        cond_dim: int = 256,
        hidden_dim: int = 256,
        n_blocks: int = 4,
        n_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        time_dim: int = 256,
    ):
        super().__init__()
        self.num_points = num_points
        self.point_dim = point_dim
        self.action_horizon = action_horizon
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim

        # Anchor projection: current point position -> per-point query embedding.
        self.anchor_proj = nn.Linear(point_dim, hidden_dim)

        # Noise projection: flatten the H-step noisy delta per point into the query.
        # Each point carries action_horizon * point_dim scalars.
        self.noise_proj = nn.Linear(action_horizon * point_dim, hidden_dim)

        # Time MLP over a sinusoidal positional embedding.
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Stacked cross-attention blocks with trunk_feats as the context.
        self.blocks = nn.ModuleList(
            [
                CrossBlock(
                    cond_dim=cond_dim,
                    hidden_dim=hidden_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                    mlp_layers=2,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(n_blocks)
            ]
        )

        # Output projection: per-point hidden -> H * D scalars. Zero-init so the
        # decoder starts as identity-like (matches DiT FinalLayer idiom).
        self.out_proj = nn.Linear(hidden_dim, action_horizon * point_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        time: torch.Tensor,
        global_cond: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if x_t.ndim != 4:
            raise ValueError(
                f"PointFlowDecoder expects x_t [B, H, N, D], got {tuple(x_t.shape)}"
            )
        B, H, N, D = x_t.shape
        if H != self.action_horizon or D != self.point_dim:
            raise ValueError(
                f"PointFlowDecoder built for H={self.action_horizon}, D={self.point_dim}"
                f" but got H={H}, D={D}"
            )

        current_points = global_cond["current_points"]
        trunk_feats = global_cond["trunk_feats"]
        if current_points.shape[:2] != (B, N) or current_points.shape[-1] != D:
            raise ValueError(
                f"current_points shape {tuple(current_points.shape)} incompatible with "
                f"x_t [B={B}, H, N={N}, D={D}]"
            )

        anchor_tok = self.anchor_proj(current_points)  # [B, N, hidden_dim]

        # Flatten per-point noisy trajectory: [B, H, N, D] -> [B, N, H*D]
        noise_flat = x_t.permute(0, 2, 1, 3).reshape(B, N, H * D)
        noise_tok = self.noise_proj(noise_flat)  # [B, N, hidden_dim]

        if time.ndim == 0:
            time = time.expand(B)
        time_emb = posemb_sincos(
            time.to(dtype=anchor_tok.dtype),
            self.time_dim,
            min_period=1.0,
            max_period=100.0,
        )
        time_tok = self.time_mlp(time_emb).unsqueeze(1)  # [B, 1, hidden_dim]

        tokens = anchor_tok + noise_tok + time_tok  # [B, N, hidden_dim]

        for blk in self.blocks:
            tokens = blk(tokens, trunk_feats)

        out_flat = self.out_proj(tokens)  # [B, N, H*D]
        return out_flat.reshape(B, N, H, D).permute(0, 2, 1, 3).contiguous()
