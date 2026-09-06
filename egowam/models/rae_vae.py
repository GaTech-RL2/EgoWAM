from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _import_rae_instantiate_from_config(rae_src: Path):
    """Load RAE instantiate helper, even when another top-level `utils` shadows it."""
    rae_src_str = str(rae_src)
    if rae_src_str in sys.path:
        sys.path.remove(rae_src_str)
    sys.path.insert(0, rae_src_str)

    try:
        module = importlib.import_module("utils.model_utils")
        instantiate_from_config = getattr(module, "instantiate_from_config")
        return instantiate_from_config
    except Exception as first_exc:
        model_utils_path = rae_src / "utils" / "model_utils.py"
        if not model_utils_path.exists():
            raise ImportError(
                "Failed to import RAE Stage-2 instantiate helper; "
                f"missing file: {model_utils_path}"
            ) from first_exc

        spec = importlib.util.spec_from_file_location(
            "rae_stage2_model_utils",
            str(model_utils_path),
        )
        if spec is None or spec.loader is None:
            raise ImportError(
                "Failed to import RAE Stage-2 instantiate helper; "
                "could not create import spec for model_utils.py"
            ) from first_exc

        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as second_exc:
            raise ImportError(
                "Failed to import RAE Stage-2 instantiate helper "
                "from external/RAE/src/utils/model_utils.py"
            ) from second_exc

        instantiate_from_config = getattr(module, "instantiate_from_config", None)
        if instantiate_from_config is None:
            raise ImportError(
                "Failed to import RAE Stage-2 instantiate helper; "
                "`instantiate_from_config` not found in model_utils.py"
            )
        return instantiate_from_config


def _resolve_rae_roots(
    repo_root: Path,
    explicit_root: Optional[Union[str, Path]] = None,
) -> tuple[Path, Path]:
    """Return (rae_repo_root, rae_src). ``rae_src`` must be on ``sys.path`` for ``import stage1``.

    Priority: ``explicit_root`` (Hydra) > ``EGOVERSE_RAE_ROOT`` > ``<repo>/external/RAE``.
    """
    if explicit_root is not None and str(explicit_root).strip():
        rae_root = Path(explicit_root).expanduser().resolve()
    else:
        # EGOWAM_RAE_ROOT is the documented name (it matches paths.rae_root in the Hydra
        # configs); EGOVERSE_RAE_ROOT is accepted for continuity with older setups.
        env_root = (
            os.environ.get("EGOWAM_RAE_ROOT", "").strip()
            or os.environ.get("EGOVERSE_RAE_ROOT", "").strip()
        )
        if env_root:
            rae_root = Path(env_root).expanduser().resolve()
        else:
            rae_root = (repo_root / "external" / "RAE").resolve()
    rae_src = rae_root / "src"
    if not rae_src.is_dir():
        from egowam.external_deps import RAE

        # Actionable failure: name the command that fixes it, not just the missing path.
        raise FileNotFoundError(RAE.install_hint())
    return rae_root, rae_src


def _import_stage1_rae(rae_src: Path):
    """Load ``RAE`` from upstream layout: ``src/stage1/`` (package) or legacy ``src/stage1.py``."""
    stage1_py = rae_src / "stage1.py"
    stage1_pkg_init = rae_src / "stage1" / "__init__.py"
    if not rae_src.is_dir():
        raise ImportError(
            f"RAE source directory missing: {rae_src}. "
            "external/RAE is vendored in this repo; restore it with "
            "`git checkout -- external/RAE`, or set EGOWAM_RAE_ROOT to another "
            "RAE repo root (a directory containing `src/stage1/`)."
        )
    if not stage1_py.is_file() and not stage1_pkg_init.is_file():
        raise ImportError(
            f"RAE Stage-1 not found under {rae_src} (expected stage1.py or stage1/__init__.py)."
        )

    if str(rae_src) not in sys.path:
        sys.path.insert(0, str(rae_src))

    try:
        from stage1 import RAE  # noqa: PLC0415

        return RAE
    except Exception as first_exc:
        if not stage1_py.is_file():
            raise ImportError(
                f"Failed to import RAE Stage-1 from {rae_src} (package stage1/). "
                "Install RAE dependencies (see external/RAE/environment.yml)."
            ) from first_exc
        spec = importlib.util.spec_from_file_location(
            "egoverse_rae_stage1", str(stage1_py)
        )
        if spec is None or spec.loader is None:
            raise ImportError(
                f"Could not load RAE stage1 module from {stage1_py}"
            ) from first_exc
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as load_exc:
            raise ImportError(
                f"Failed to exec_module RAE stage1 from {stage1_py}"
            ) from load_exc
        RAE = getattr(module, "RAE", None)
        if RAE is None:
            raise ImportError(f"No class `RAE` in {stage1_py}") from None
        return RAE


class RAEVAEWrapper(nn.Module):
    """Thin wrapper around external RAE stage-1 model with Cosmos-like API.

    Expected I/O (to match `CosmosVAEWrapper` usage in HPT):
    - `encode`: `[B, C, T, H, W] -> [B, C_lat, T, H_lat, W_lat]`
    - `decode`: `[B, C_lat, T, H_lat, W_lat] -> [B, C, T, H, W]`
    """

    def __init__(
        self,
        config_path: str,
        ckpt_path: Optional[str] = None,
        use_stage2_features: bool = True,
        stage2_t: float = 1.0,
        stage2_label: int = 0,
        freeze: bool = True,
        device: Optional[str] = None,
        rae_root: Optional[str] = None,
    ):
        super().__init__()
        self.freeze = freeze
        self._rae_root_override: Optional[Path] = (
            Path(rae_root).expanduser().resolve() if rae_root else None
        )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = self._build_model(
            config_path=config_path,
            ckpt_path=ckpt_path,
            device=device,
        )
        self.use_stage2_features = bool(use_stage2_features)
        self.stage2_t = float(stage2_t)
        self.stage2_label = int(stage2_label)
        self.stage2_model = None
        if self.use_stage2_features:
            self.stage2_model = self._build_stage2_model(
                config_path=config_path, device=device
            )
            self.stage2_model.eval()
        self._configure_latent_adapter(
            config_path=config_path,
        )

        if self.freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def _configure_latent_adapter(
        self,
        config_path: str,
    ) -> None:
        # Native latent shape from Stage-1 model settings.
        native_c = int(getattr(self.model, "latent_dim", 0))
        base_patches = int(getattr(self.model, "base_patches", 0))
        native_hw = int(round(math.sqrt(base_patches))) if base_patches > 0 else 0
        if native_c <= 0 or native_hw <= 0:
            raise ValueError(
                "Could not infer native latent shape from Stage-1 RAE model."
            )
        self.native_latent_channels = native_c
        self.native_latent_h = native_hw
        self.native_latent_w = native_hw

    def _load_omegaconf(self, config_path: str):
        repo_root = Path(__file__).resolve().parents[2]
        rae_root, _rae_src = _resolve_rae_roots(repo_root, self._rae_root_override)

        def _resolve_local_path(path_like: Optional[str]) -> Optional[str]:
            if (
                path_like is None
                or not isinstance(path_like, str)
                or path_like.strip() == ""
            ):
                return path_like
            p = Path(path_like)
            if p.is_absolute():
                return str(p)
            if "://" in path_like:
                return path_like
            candidates = [Path.cwd() / p, repo_root / p, rae_root / p]
            for cand in candidates:
                if cand.exists():
                    return str(cand.resolve())
            return path_like

        try:
            from omegaconf import OmegaConf
        except Exception as exc:
            raise ImportError("omegaconf is required to parse RAE configs.") from exc

        resolved_path = _resolve_local_path(config_path)
        if resolved_path is None or not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        return OmegaConf.load(resolved_path)

    def _build_model(
        self,
        config_path: str,
        ckpt_path: Optional[str],
        device: str,
    ) -> nn.Module:
        repo_root = Path(__file__).resolve().parents[2]
        rae_root, rae_src = _resolve_rae_roots(repo_root, self._rae_root_override)

        def _resolve_local_path(path_like: Optional[str]) -> Optional[str]:
            if (
                path_like is None
                or not isinstance(path_like, str)
                or path_like.strip() == ""
            ):
                return path_like
            p = Path(path_like)
            if p.is_absolute():
                return str(p)

            # Keep URI-style inputs untouched (e.g., hf://)
            if "://" in path_like:
                return path_like

            candidates = [Path.cwd() / p, repo_root / p, rae_root / p]
            for cand in candidates:
                if cand.exists():
                    return str(cand.resolve())
            return path_like

        try:
            from omegaconf import OmegaConf
        except Exception as exc:
            raise ImportError("omegaconf is required to load RAE configs.") from exc

        RAE = _import_stage1_rae(rae_src)

        resolved_config_path = _resolve_local_path(config_path)
        if resolved_config_path is None or not os.path.exists(resolved_config_path):
            raise FileNotFoundError(
                f"RAE config not found: {config_path}. "
                "Expected an existing local YAML file."
            )
        cfg = OmegaConf.load(resolved_config_path)
        stage1_cfg = cfg.get("stage_1", None)
        if stage1_cfg is None:
            raise ValueError(
                f"No 'stage_1' section found in config: {resolved_config_path}"
            )

        params = dict(stage1_cfg.get("params", {}))
        for key in (
            "decoder_config_path",
            "pretrained_decoder_path",
            "normalization_stat_path",
            "encoder_config_path",
        ):
            if key in params:
                params[key] = _resolve_local_path(params[key])

        model = RAE(**params)

        resolved_ckpt = _resolve_local_path(ckpt_path or stage1_cfg.get("ckpt", None))
        if resolved_ckpt is not None:
            state = torch.load(resolved_ckpt, map_location="cpu")
            if "ema" in state:
                state = state["ema"]
            elif "model" in state:
                state = state["model"]
            model.load_state_dict(state, strict=True)

        model = model.to(device)
        model.eval()
        return model

    def _build_stage2_model(self, config_path: str, device: str) -> nn.Module:
        repo_root = Path(__file__).resolve().parents[2]
        rae_root, rae_src = _resolve_rae_roots(repo_root, self._rae_root_override)

        def _resolve_local_path(path_like: Optional[str]) -> Optional[str]:
            if (
                path_like is None
                or not isinstance(path_like, str)
                or path_like.strip() == ""
            ):
                return path_like
            p = Path(path_like)
            if p.is_absolute():
                return str(p)
            if "://" in path_like:
                return path_like
            candidates = [Path.cwd() / p, repo_root / p, rae_root / p]
            for cand in candidates:
                if cand.exists():
                    return str(cand.resolve())
            return path_like

        if str(rae_src) not in sys.path:
            sys.path.insert(0, str(rae_src))

        instantiate_from_config = _import_rae_instantiate_from_config(rae_src)

        cfg = self._load_omegaconf(config_path)
        stage2_cfg = cfg.get("stage_2", None)
        if stage2_cfg is None:
            raise ValueError(
                "use_stage2_features=True requires config_path to include a stage_2 block."
            )

        stage2_ckpt = stage2_cfg.get("ckpt", None)
        if stage2_ckpt is not None:
            stage2_cfg["ckpt"] = _resolve_local_path(stage2_ckpt)

        model = instantiate_from_config(stage2_cfg)
        model = model.to(device)
        model.eval()
        return model

    def forward(self, x, t, y, s=None, mask=None):
        m = self.stage2_model
        t = m.t_embedder(t)
        y = m.y_embedder(y, m.training)
        c = nn.functional.silu(t + y)
        if s is None:
            s = m.s_embedder(x)
            if m.use_pos_embed:
                s = s + m.pos_embed
            # print(f"t shape: {t.shape}, y shape: {y.shape}, c shape: {c.shape}, s shape: {s.shape}, pos_embed shape: {m.pos_embed.shape}")
            for i in range(m.num_encoder_blocks):
                s = m.blocks[i](s, c, feat_rope=m.enc_feat_rope)
            # broadcast t to s
            t = t.unsqueeze(1).repeat(1, s.shape[1], 1)
            s = nn.functional.silu(t + s)
        else:
            s = m.s_embedder(s)
            if m.use_pos_embed:
                s = s + m.pos_embed
            # print(f"t shape: {t.shape}, y shape: {y.shape}, c shape: {c.shape}, s shape: {s.shape}, pos_embed shape: {m.pos_embed.shape}")
            for i in range(m.num_encoder_blocks):
                s = m.blocks[i](s, c, feat_rope=m.enc_feat_rope)
            # broadcast t to s
            t = t.unsqueeze(1).repeat(1, s.shape[1], 1)
            s = nn.functional.silu(t + s)
        s = m.s_projector(s)
        x = m.x_embedder(x)
        if m.use_pos_embed and m.x_pos_embed is not None:
            x = x + m.x_pos_embed
        for i in range(m.num_encoder_blocks, m.num_blocks):
            x = m.blocks[i](x, s, feat_rope=m.dec_feat_rope)
        x = m.final_layer(x, s)
        x = m.unpatchify(x)
        return x

    def _stage2_forward_features(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        replicate DiTwDDTHead.forward up to (but excluding) final_layer.
        Returns:
            x_tokens: [B, N_tokens, hidden]
            s_tokens: [B, N_tokens, hidden]
        """
        if self.stage2_model is None:
            raise RuntimeError("Stage-2 model is not initialized.")

        m = self.stage2_model
        required = [
            "t_embedder",
            "y_embedder",
            "s_embedder",
            "x_embedder",
            "s_projector",
            "blocks",
            "num_encoder_blocks",
            "num_blocks",
            "final_layer",
            "unpatchify",
        ]
        missing = [name for name in required if not hasattr(m, name)]
        if missing:
            raise AttributeError(
                "Stage-2 model missing required attributes for local feature extraction: "
                + ", ".join(missing)
            )

        t_emb = m.t_embedder(t)
        y_emb = m.y_embedder(y, m.training)
        c = F.silu(t_emb + y_emb)

        s = m.s_embedder(x)
        if getattr(m, "use_pos_embed", False):
            s = s + m.pos_embed
        enc_rope = getattr(m, "enc_feat_rope", None)
        for i in range(m.num_encoder_blocks):
            s = m.blocks[i](s, c, feat_rope=enc_rope)

        t_seq = t_emb.unsqueeze(1).repeat(1, s.shape[1], 1)
        s = F.silu(t_seq + s)
        s = m.s_projector(s)
        # print("Stage-2 forward features:", f"s shape {s.shape}")

        x_tok = m.x_embedder(x)
        if (
            getattr(m, "use_pos_embed", False)
            and getattr(m, "x_pos_embed", None) is not None
        ):
            x_tok = x_tok + m.x_pos_embed
        dec_rope = getattr(m, "dec_feat_rope", None)
        # print("Stage-2 forward features:", f"x_tok initial shape {x_tok.shape}, s shape {s.shape}")
        for i in range(m.num_encoder_blocks, m.num_blocks):
            x_tok = m.blocks[i](x_tok, s, feat_rope=dec_rope)
            # print(f"After block {i}: x_tok shape {x_tok.shape}, s shape {s.shape}")
        x_reconst = m.final_layer(x_tok, s)
        # print(f"After final_layer: x_reconst shape {x_reconst.shape}")
        x_reconst = m.unpatchify(x_reconst)
        # print(f"After unpatchify: x_reconst shape {x_reconst.shape}")
        return x_tok, x_reconst

    def _encode_with_stage2_features(self, latents_btchw: torch.Tensor) -> torch.Tensor:
        if self.stage2_model is None:
            return latents_btchw
        bt = latents_btchw.shape[0]
        t = torch.full(
            (bt,),
            self.stage2_t,
            device=latents_btchw.device,
            dtype=latents_btchw.dtype,
        )
        y = torch.full(
            (bt,),
            self.stage2_label,
            device=latents_btchw.device,
            dtype=torch.long,
        )
        feats, reconst = self._stage2_forward_features(latents_btchw, t, y)
        # Use reconstructed features (output of final_layer + unpatchify) for encode
        # This is already in native latent shape [BT, C, H, W]
        return reconst

    @staticmethod
    def _flatten_video(x_bcthw: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        if x_bcthw.ndim != 5:
            raise ValueError(
                f"Expected input shape [B,C,T,H,W], got {tuple(x_bcthw.shape)}"
            )
        b, c, t, h, w = x_bcthw.shape
        x_btchw = x_bcthw.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        return x_btchw, b, t

    @staticmethod
    def _unflatten_video(z_btchw: torch.Tensor, b: int, t: int) -> torch.Tensor:
        if z_btchw.ndim != 4:
            raise ValueError(
                f"Expected latent shape [B*T,C,H,W], got {tuple(z_btchw.shape)}"
            )
        bt, c, h, w = z_btchw.shape
        if bt != b * t:
            raise ValueError(
                f"Mismatched batch/time dims: expected B*T={b * t}, got {bt}"
            )
        return z_btchw.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()

    @torch.no_grad()
    def encode(self, videos_bcthw: torch.Tensor) -> torch.Tensor:
        videos_btchw, b, t = self._flatten_video(videos_bcthw)
        latents_btchw = self.model.encode(videos_btchw)  # return DINO features
        # if self.use_stage2_features:
        # 	latents_btchw = self._encode_with_stage2_features(latents_btchw)
        return self._unflatten_video(latents_btchw, b=b, t=t)

    @torch.no_grad()
    def decode(self, latents_bcthw: torch.Tensor) -> torch.Tensor:
        latents_btchw, b, t = self._flatten_video(latents_bcthw)
        videos_btchw = self.model.decode(latents_btchw)
        return self._unflatten_video(videos_btchw, b=b, t=t)

    def decode_with_grad(self, latents_bcthw: torch.Tensor) -> torch.Tensor:
        """Decode allowing gradients to flow through the (optionally frozen) decoder."""
        latents_btchw, b, t = self._flatten_video(latents_bcthw)
        videos_btchw = self.model.decode(latents_btchw)
        return self._unflatten_video(videos_btchw, b=b, t=t)
