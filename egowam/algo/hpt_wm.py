import os
from collections import OrderedDict
from functools import partial
from typing import Dict, List, Optional

import cv2
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from overrides import override
from termcolor import cprint
from torchmetrics import MeanSquaredError

from egowam.algo.algo import Algo
from egowam.models.fm_policy import PointFMPolicy
from egowam.models.hpt_nets import (
    CrossAttention,
    MultiheadAttention,
    SimpleTransformer,
)
from egowam.rldb.embodiment.embodiment import get_embodiment, get_embodiment_id
from egowam.rldb.zarr.point_flow_loader import POINT_FLOW_2D_IMAGE_SHAPE
from egowam.utils.egowamUtils import (
    STD_SCALE,
    EinOpsRearrange,
    download_from_huggingface,
    draw_actions,
    draw_rotation_text,
    frechet_gaussian_over_time,
    get_sinusoid_encoding_table,
    reverse_kl_from_samples,
)


class HPTModel(nn.Module):
    """
    Heterogenous Pretrained Transformer (HPT) implementation based on the HPT paper, with additional modifications.
    This model integrates modality-specific stems, a transformer trunk, and domain-specific heads to process
    multi-modal data.
    """

    def __init__(
        self,
        embed_dim=1024,
        num_blocks=24,
        num_heads=16,
        token_postprocessing="action_token",
        observation_horizon=4,
        action_horizon=1,
        future_horizon=0,
        no_trunk=False,
        shared_modality_trunk=None,
        use_domain_embedding=False,
        drop_path=0.0,
        weight_init_style="pytorch",
        **kwargs,
    ):
        """
        Initialize the HPTModel.

        Parameters
        ----------
        embed_dim : int, optional
            Dimension of the token embeddings (default is 1024).
        num_blocks : int, optional
            Number of transformer blocks (default is 24).
        num_heads : int, optional
            Number of attention heads in each transformer block (default is 16).
        token_postprocessing : str, optional
            Strategy for postprocessing tokens. Options include "action_token", "joint_token" (action + future
            query tokens for factored p(s'|z)p(a|s',z)), "mean", "max", "last", and "no-op"
            (default is "action_token").
        observation_horizon : int, optional
            Number of past observations to consider (default is 4).
        action_horizon : int, optional
            Number of action tokens to predict (default is 1).
        future_horizon : int, optional
            Number of future-state query tokens when token_postprocessing is "joint_token" (default is 0).
        no_trunk : bool, optional
            If True, the transformer trunk is skipped (default is False).
        shared_modality_trunk : optional
            Shared trunk module for modality-specific processing if provided.
        use_domain_embedding : bool, optional
            Whether to use domain-specific embeddings (default is False).
        drop_path : float, optional
            Drop path rate for regularization (default is 0.0).
        weight_init_style : str, optional
            Weight initialization style (default is "pytorch").
        **kwargs : dict
            Additional keyword arguments.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.shared_modality_trunk = shared_modality_trunk
        self.no_trunk = no_trunk

        self.encoders = nn.ModuleDict()

        self.trunk = self._create_policy_trunk(
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            drop_path=drop_path,
            weight_init_style=weight_init_style,
        )

        self.stems = {}
        self.heads = {}
        # self.normalizer = {}
        self.domains = []
        self.use_modality_embedding = use_domain_embedding
        self.observation_horizon = observation_horizon
        self.action_horizon = action_horizon
        self.future_horizon = future_horizon
        self.token_postprocessing = token_postprocessing
        # self.modalities_tokens = {}
        self.action_tokens = None
        self.future_tokens = None
        self.stem_spec = {}
        self.head_spec = {}

        self.modalities = {}

        self.shared_keys = []

        self.auxiliary_ac_keys = None
        self.auxiliary_loss_weights = {}
        # If None, legacy: use shared_{key} when that head exists in head_specs / self.heads.
        # If a list (possibly empty), only keys in the list use shared auxiliary heads.
        self.shared_auxiliary_ac_keys = None
        self.shared_action = False
        # Domains that receive action (BC) supervision. None => all domains.
        # Domains absent from this set still contribute auxiliary/world-model
        # losses (e.g. flow) but no per-domain or shared action-head loss.
        self.action_domains = None
        self.device = None

        self.ot_6dof = False
        self.use_dtw = False
        self.depth = None
        self.lambd = None

        self.diffusion = None
        self.wm_prediction_mode = "autoregressive"
        self.autoregressive_condition_scale = 0.1
        self.teacher_forcing_ratio = 1.0
        self._last_loss_breakdown = {}

        # 1=aux-only, 2=action-only, 3=joint (default: joint, backward compatible)
        self.training_stage = 3

    @staticmethod
    def _is_point_flow_head(head) -> bool:
        return isinstance(head, PointFMPolicy)

    def _build_point_flow_aux_data(
        self, domain: str, features, batch, flow_key: str
    ) -> dict:
        """Assemble dict aux_data for PointFMPolicy from the batch sidecars the
        loader publishes (`{k}__current_points`, `{k}__validity_mask`).

        Used by both the loss path (compute_loss) and the sampling path
        (sample_aux) so dispatch sites share one assembly site.
        """
        cur_key = f"{flow_key}__current_points"
        mask_key = f"{flow_key}__validity_mask"
        if cur_key not in batch or mask_key not in batch:
            raise KeyError(
                f"Point-flow head '{flow_key}' requires batch['{cur_key}'] and "
                f"batch['{mask_key}'] (published by the loader). Present keys: "
                f"{sorted(batch.keys())[:20]}"
            )
        return {
            "trunk_feats": features,
            "current_points": batch[cur_key],
            "validity_mask": batch[mask_key].to(dtype=torch.bool),
            "target_delta_normed": batch[flow_key],
            "embodiment": domain,
        }

    def _predict_state_targets(
        self, domain, future_features, action_features=None, batch=None
    ):
        """
        Sample WM state predictions from auxiliary heads for conditioning.

        Uses future token features when ``joint_token`` mode provides them; otherwise falls back
        to ``action_features`` (e.g. ``action_token`` mode). For PointFMPolicy heads this
        routes through sample_aux(dict_cond) using the same _build_point_flow_aux_data
        helper as the loss path — single dispatch contract across all three aux call
        sites (this one, _compute_joint_state_loss, and the eval aux loop at L~880).
        """
        pred_states = {}
        if domain not in self.auxiliary_ac_keys:
            return pred_states

        features = future_features if future_features is not None else action_features
        if features is None:
            return pred_states

        for key in self.auxiliary_ac_keys[domain]:
            head_key = self._resolve_aux_head_key(domain, key)
            if head_key not in self.heads:
                continue
            head = self.heads[head_key]
            if self._is_point_flow_head(head):
                if batch is None:
                    # No batch available in this path (e.g. prediction-only).
                    # Skip point-flow aux prediction here; action loss doesn't
                    # need the predicted flow trajectory as conditioning.
                    continue
                aux_cond = self._build_point_flow_aux_data(domain, features, batch, key)
                pred_states[key] = head.sample_aux(aux_cond)
            else:
                global_cond = (features, domain) if self.diffusion else features
                pred_states[key] = head(global_cond)
        return pred_states

    def _resolve_aux_head_key(self, domain, key):
        """Resolve auxiliary head name: explicit shared_auxiliary_ac_keys or legacy head lookup."""
        shared_key = f"shared_{key}"
        sal = getattr(self, "shared_auxiliary_ac_keys", None)
        if sal is not None:
            if key in sal:
                return shared_key
            return f"{domain}_{key}"
        if shared_key in self.heads:
            return shared_key
        return f"{domain}_{key}"

    def _get_gt_state_targets(self, domain, data):
        """
        Return ground-truth future state targets from the batch for teacher forcing.
        """
        gt_states = {}
        if domain not in self.auxiliary_ac_keys:
            return gt_states
        for key in self.auxiliary_ac_keys[domain]:
            if key in data:
                gt_states[key] = data[key]
        return gt_states

    def _fuse_features_with_state(self, features, pred_states):
        """
        Deterministically fuse predicted s' into trunk features.
        """
        if len(pred_states) == 0:
            return features

        summaries = []
        for state_pred in pred_states.values():
            if state_pred.ndim == 3:
                summaries.append(state_pred[:, 0].float())
            else:
                summaries.append(state_pred.float().reshape(state_pred.shape[0], -1))

        state_vec = torch.cat(summaries, dim=-1)
        if state_vec.shape[-1] >= self.embed_dim:
            state_vec = state_vec[..., : self.embed_dim]
        else:
            pad = torch.zeros(
                state_vec.shape[0],
                self.embed_dim - state_vec.shape[-1],
                device=state_vec.device,
                dtype=state_vec.dtype,
            )
            state_vec = torch.cat([state_vec, pad], dim=-1)
        fused = features + self.autoregressive_condition_scale * state_vec.unsqueeze(1)
        return fused

    def _compute_joint_state_loss(
        self, domain, future_features, data, action_features=None, return_per_key=False
    ):
        """
        Predict future states with auxiliary heads from future-token features (or action tokens if none).
        When return_per_key is True, also returns a dict {aux_key: detached scalar} of unweighted per-head losses.
        """
        auxiliary_action_loss = torch.tensor(0.0, device=self.device)
        per_key: Dict[str, torch.Tensor] = {}
        features = future_features if future_features is not None else action_features
        if features is None or domain not in self.auxiliary_ac_keys:
            if return_per_key:
                return auxiliary_action_loss, per_key
            return auxiliary_action_loss
        for key in self.auxiliary_ac_keys[domain]:
            head_key = self._resolve_aux_head_key(domain, key)
            if head_key in self.heads:
                head = self.heads[head_key]
                if self._is_point_flow_head(head):
                    aux_data = self._build_point_flow_aux_data(
                        domain, features, data, key
                    )
                    aux_loss = head.compute_loss(features, aux_data)
                else:
                    aux_data = {**data, "action": data[key]}
                    aux_loss = head.compute_loss(features, aux_data)
                weight = self.auxiliary_loss_weights.get(key, 1.0)
                auxiliary_action_loss += weight * aux_loss
                if return_per_key:
                    per_key[key] = aux_loss.detach()
        if return_per_key:
            return auxiliary_action_loss, per_key
        return auxiliary_action_loss

    def _apply_action_cross_attention(self, domain, action_feats, future_feats):
        """
        Cross-attend action tokens to the trunk's future tokens (cross mode).
        Returns action_feats unchanged when cross attn is unavailable for the domain
        or future_feats is missing.
        """
        if (
            future_feats is None
            or not hasattr(self, "action_cross_attn")
            or domain not in self.action_cross_attn
        ):
            return action_feats
        cross_out = self.action_cross_attn[domain](action_feats, future_feats)
        gate = self.action_cross_gate[domain]
        return action_feats + gate * cross_out

    def _domain_has_action_supervision(self, domain):
        """Whether `domain` contributes to the action (BC) loss. When
        `action_domains` is None all domains do (legacy behaviour); otherwise
        only the listed domains do. Domains excluded here still keep their
        auxiliary/world-model (e.g. flow) supervision."""
        if self.action_domains is None:
            return True
        return domain in self.action_domains

    def _compute_action_head_loss(self, domain, data, action_feats, future_feats):
        """
        Main policy / flow-matching loss on action tokens, optionally conditioned on predicted or GT s'.
        """
        if domain not in self.heads:
            return torch.tensor(0.0, device=self.device)
        if self.wm_prediction_mode == "cross":
            # action_feats was already cross-attended in the caller (compute_loss).
            return self.heads[domain].compute_loss(action_feats, data)
        if self.wm_prediction_mode == "joint":
            return self.heads[domain].compute_loss(action_feats, data)
        if (
            self.wm_prediction_mode == "autoregressive"
            and domain in self.auxiliary_ac_keys
        ):
            pred_states = self._predict_state_targets(
                domain, future_feats, action_feats, batch=data
            )
            conditioned = self._fuse_features_with_state(action_feats, pred_states)
            return self.heads[domain].compute_loss(conditioned, data)
        if self.wm_prediction_mode == "teacher_forcing":
            state_targets = self._get_gt_state_targets(domain, data)
            conditioned = self._fuse_features_with_state(action_feats, state_targets)
            return self.heads[domain].compute_loss(conditioned, data)
        return self.heads[domain].compute_loss(action_feats, data)

    def init_encoder(self, modality, encoder_spec):
        """
        Initialize an encoder for the specified modality.

        Parameters
        ----------
        modality : str
            The name of the modality.
        encoder_spec : dict or object
            The specification or configuration for the encoder.
        """
        self.encoders[modality] = encoder_spec

    def init_domain_stem(self, domain_name, stem_spec):
        """
        Initialize the stem (feature extractor) for a given domain along with its modalities.

        Parameters
        ----------
        domain_name : str
            The name of the domain.
        stem_spec : dict-like
            A specification containing configurations for each modality's stem.
        """

        self.stem_spec[domain_name] = stem_spec
        self.modalities[domain_name] = list(stem_spec.keys())

        for modality in self.modalities[domain_name]:
            stem_name = f"{domain_name}_{modality}"
            self.stems[stem_name] = stem_spec[modality]
            if hasattr(self.stems[stem_name], "init_cross_attn"):
                self.stems[stem_name].init_cross_attn(
                    stem_spec[modality].specs.cross_attn
                )

    def init_domain_head(self, domain_name, head_spec):
        """
        Initialize the head (prediction module) for a given domain.

        Parameters
        ----------
        domain_name : str
            The name of the domain.
        head_spec : dict or object
            The specification or configuration for the head, used with hydra.utils.instantiate.
        """
        self.head_spec[domain_name] = head_spec
        self.domains.append(domain_name)
        self.heads[domain_name] = head_spec

    def finalize_modules(self):
        """
        Finalize the module initialization by converting stems, heads, and modality tokens into
        nn.ModuleDict/nn.ParameterDict objects, applying weight initialization, and creating shared
        action tokens if required.
        """
        self.stems = nn.ModuleDict(self.stems)
        self.heads = nn.ModuleDict(self.heads)
        self.apply(self._init_weights)

        # Learnable query tokens for trunk
        if self.token_postprocessing == "action_token":
            self.action_tokens = nn.Parameter(
                torch.randn(1, self.action_horizon, self.embed_dim) * STD_SCALE
            )
        elif self.token_postprocessing == "joint_token":
            if self.future_horizon <= 0:
                raise ValueError(
                    "token_postprocessing='joint_token' requires future_horizon > 0"
                )
            self.action_tokens = nn.Parameter(
                torch.randn(1, self.action_horizon, self.embed_dim) * STD_SCALE
            )
            self.future_tokens = nn.Parameter(
                torch.randn(1, self.future_horizon, self.embed_dim) * STD_SCALE
            )

    def _create_policy_trunk(
        self, embed_dim, num_blocks, num_heads, drop_path, weight_init_style
    ):
        """
        Create the transformer trunk module for policy processing.

        Parameters
        ----------
        embed_dim : int
            Dimension of token embeddings.
        num_blocks : int
            Number of transformer blocks.
        num_heads : int
            Number of attention heads in each block.
        drop_path : float
            Drop path rate for regularization.
        weight_init_style : str
            Weight initialization style.

        Returns
        -------
        nn.ModuleDict
            A module dictionary containing the main trunk transformer and, if provided, shared modality trunks.
        """
        trunk = {}

        trunk["trunk"] = SimpleTransformer(
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            ffn_dropout_rate=0.0,
            drop_path_rate=drop_path,
            attn_target=partial(
                MultiheadAttention,
                embed_dim=embed_dim,
                num_heads=num_heads,
                bias=True,
                add_bias_kv=True,
            ),
            pre_transformer_layer=nn.Sequential(
                nn.Identity(),
                EinOpsRearrange("b l d -> l b d"),
            ),
            post_transformer_layer=EinOpsRearrange("l b d -> b l d"),
            weight_init_style=weight_init_style,
        )
        if (
            hasattr(self, "shared_modality_trunk")
            and self.shared_modality_trunk is not None
        ):
            for modality in self.shared_modality_trunk.modalities:
                trunk[modality] = self.shared_modality_trunk[modality]

        return nn.ModuleDict(trunk)

    def get_position_embedding(self, feature, embed_dim):
        """
        Generate sinusoidal positional embeddings for a given feature tensor.

        Parameters
        ----------
        feature : torch.Tensor
            The input tensor for which positional embeddings are computed.
        embed_dim : int
            The embedding dimension.

        Returns
        -------
        torch.Tensor
            The positional embedding tensor with the same device as the input.
        """
        tokensize = int(feature.shape[1])
        tokens = get_sinusoid_encoding_table(0, tokensize, self.embed_dim)
        return tokens.repeat((1, 1, 1)).to(feature.device)

    def preprocess_tokens(self, domain, features):
        """
        Preprocess and combine stem tokens with optional action tokens and add positional embeddings.

        Parameters
        ----------
        domain : str
            The domain for which tokens are being processed.
        features : list of torch.Tensor
            List of feature tokens from different modalities.

        Returns
        -------
        torch.Tensor
            The combined token tensor after adding positional embeddings.
        """
        tokens = torch.cat(features, dim=-2)  # [B, cross_atten_dim, modality_embed_dim]

        if self.token_postprocessing == "action_token":
            action_tokens = self.action_tokens.repeat(len(tokens), 1, 1)
            tokens = torch.cat([action_tokens, tokens], dim=-2)
        elif self.token_postprocessing == "joint_token":
            action_tokens = self.action_tokens.repeat(len(tokens), 1, 1)
            future_tokens = self.future_tokens.repeat(len(tokens), 1, 1)
            tokens = torch.cat([action_tokens, future_tokens, tokens], dim=-2)

        position_tokens = self.get_position_embedding(tokens, self.embed_dim)
        return tokens + position_tokens

    def postprocess_tokens(self, trunk_tokens):
        """
        Postprocess the tokens output from the transformer trunk based on the token_postprocessing strategy.

        Parameters
        ----------
        trunk_tokens : torch.Tensor
            The token tensor output from the transformer trunk.

        Returns
        -------
        torch.Tensor
            The processed token tensor (e.g., averaged, max pooled, or selected action tokens).
        """
        if self.token_postprocessing == "mean":
            return trunk_tokens.mean(dim=1), None
        elif self.token_postprocessing == "action_token":
            return trunk_tokens[:, : self.action_horizon], None
        elif self.token_postprocessing == "joint_token":
            ah = self.action_horizon
            fh = self.future_horizon
            return trunk_tokens[:, :ah], trunk_tokens[:, ah : ah + fh]
        elif self.token_postprocessing == "max":
            return trunk_tokens.max(dim=1)[0], None
        elif self.token_postprocessing == "last":
            return trunk_tokens[:, -1], None
        elif self.token_postprocessing == "no-op":
            return trunk_tokens, None
        else:
            raise ValueError(
                f"Invalid token_postprocessing: {self.token_postprocessing}"
            )

    def preprocess_states(self, domain, data):
        """
        Preprocess state information in the input data by adding a new dimension if necessary.

        Parameters
        ----------
        domain : str
            The domain name.
        data : dict
            Dictionary containing input data with potential "state" keys.

        Returns
        -------
        dict
            Updated data dictionary with preprocessed state information.
        """
        for key in data:
            if "state" in key:
                data[key] = data[key][:, :, None]
        return data

    def stem_process(self, domain, data):
        """
        Process input data through modality-specific stems to compute latent feature tokens.

        Parameters
        ----------
        domain : str
            The domain corresponding to the input data.
        data : dict
            Dictionary containing input data for various modalities.

        Returns
        -------
        tuple
            A tuple containing:
                - A list of tokens from each modality.
                - A dictionary mapping each modality to its computed token.
        """
        feats = []
        feat_dict = {}
        for modality in self.modalities.get(domain, []) + self.shared_keys:
            if modality not in data:
                continue
            if modality in self.shared_keys:
                domain = "shared"

            stem = self.stems[f"{domain}_{modality}"]
            if modality in self.encoders:
                data[modality] = self.encoders[modality](data[modality])

            data_shape = data[modality].shape
            data_horizon = data_shape[1]
            horizon = data_horizon

            if (
                getattr(self, "train_mode", False)
                and self.stem_spec[domain][modality].specs.random_horizon_masking
                and data_horizon > 1
            ):
                horizon = np.random.randint(1, data_horizon + 1)
                data[modality] = data[modality][:, data_horizon - horizon :]

            positional_embedding = get_sinusoid_encoding_table(
                0, horizon * int(np.prod(data_shape[2:-1])), data_shape[-1]
            ).to(data[modality])
            positional_embedding = einops.repeat(
                positional_embedding, "b h w -> (repeat b) h w", repeat=data_shape[0]
            )

            data[modality] = data[modality] + positional_embedding.view(
                data[modality].shape
            )
            stem_token = stem.compute_latent(data[modality])
            feats.append(stem_token)
            feat_dict[modality] = stem_token

        return feats, feat_dict

    def resume_from_depth(self, block_outputs, depth):
        """
        Detach at trunk depth and resume trunk forward pass.
        Gradients will only flow from depth upward.
        """
        cut_tokens = block_outputs[depth - 1].detach()

        blocks = self.trunk["trunk"].blocks
        for blk in list(blocks)[depth:]:
            cut_tokens = blk(cut_tokens, attn_mask=None)

        if self.trunk["trunk"].post_transformer_layer is not None:
            cut_tokens = self.trunk["trunk"].post_transformer_layer(cut_tokens)

        return self.postprocess_tokens(cut_tokens)

    def get_visual_embeds(self, domain, data, modality):
        """
        Compute visual embeddings for a given modality from the input data.

        Parameters
        ----------
        domain : str
            The domain corresponding to the input data.
        data : dict
            Dictionary containing input data.
        modality : str
            The modality for which visual embeddings are to be computed.

        Returns
        -------
        list
            A list containing:
                - The encoded features from the encoder.
                - The latent tokens computed by the modality stem.
        """
        if modality in self.shared_keys:
            domain = "shared"

        stem = self.stems[f"{domain}_{modality}"]

        encoder_feats = None

        if modality in self.encoders:
            encoder_feats = self.encoders[modality](data[modality])
        data_shape = encoder_feats.shape
        data_horizon = data_shape[1]
        horizon = data_horizon

        positional_embedding = get_sinusoid_encoding_table(
            0, horizon * int(np.prod(data_shape[2:-1])), data_shape[-1]
        ).to(encoder_feats)
        positional_embedding = einops.repeat(
            positional_embedding, "b h w -> (repeat b) h w", repeat=data_shape[0]
        )
        stem_feats = encoder_feats + positional_embedding.view(encoder_feats.shape)
        stem_token = stem.compute_latent(stem_feats)
        return [encoder_feats, stem_token]

    def forward_features(self, domain, data):
        """
        Compute feature tokens by processing the input data through stems and the transformer trunk.

        Parameters
        ----------
        domain : str
            The domain name for which features are computed.
        data : dict
            Dictionary containing input data for various modalities.

        Returns
        -------
        tuple
            ``(action_features, future_features, block_outputs)``. ``future_features`` is ``None``
            unless ``token_postprocessing == "joint_token"``.
        """
        data = self.preprocess_states(domain, data)

        # stem_tokens: num_features [B, cross_atten_dim, modality_embed_dim]
        stem_tokens, token_dict = self.stem_process(domain, data)

        trunk_tokens = self.preprocess_tokens(domain, stem_tokens)

        if not self.no_trunk:
            trunk_tokens, block_outputs = self.trunk["trunk"](trunk_tokens)

        action_feats, future_feats = self.postprocess_tokens(trunk_tokens)
        return action_feats, future_feats, block_outputs

    def compute_loss_depth(self, batch, depth):
        """
        Compute BC loss but restrict gradient flow to trunk blocks from `depth` upward.
        """
        self.train_mode = self.training
        domain, data = batch["domain"], batch["data"]

        # with amp.autocast(device_type=self.device.type):
        _, _, block_outputs = self.forward_features(domain, data)
        action_feats, future_feats = self.resume_from_depth(block_outputs, depth)
        action_loss = torch.tensor(0.0, device=self.device)
        shared_action_loss = torch.tensor(0.0, device=self.device)
        auxiliary_action_loss = torch.tensor(0.0, device=self.device)

        if self.training_stage in (2, 3) and self._domain_has_action_supervision(
            domain
        ):
            action_loss += self._compute_action_head_loss(
                domain, data, action_feats, future_feats
            )
            if self.shared_action:
                shared_action_loss += self.heads["shared"].compute_loss(
                    action_feats, data
                )

        aux_per_key: Dict[str, torch.Tensor] = {}
        if self.training_stage in (1, 3):
            auxiliary_action_loss, aux_per_key = self._compute_joint_state_loss(
                domain,
                future_feats,
                data,
                action_features=action_feats,
                return_per_key=True,
            )

        total_loss = action_loss + shared_action_loss + auxiliary_action_loss
        self._last_loss_breakdown = {
            "action": action_loss.detach(),
            "shared": shared_action_loss.detach(),
            "aux": auxiliary_action_loss.detach(),
            "aux_per_key": aux_per_key,
            "mode": self.wm_prediction_mode,
            "stage": self.training_stage,
        }

        return total_loss

    def compute_loss(self, batch):
        """
        Compute the loss for a given batch of training data.

        Parameters
        ----------
        batch : dict
            Dictionary containing the keys "domain" and "data" for the input batch.

        Returns
        -------
        torch.Tensor
            The computed loss value.
        """
        self.train_mode = self.training
        domain, data = batch["domain"], batch["data"]

        # scaler = amp.GradScaler()
        # with amp.autocast(device_type=self.device.type):
        action_feats, future_feats, _ = self.forward_features(domain, data)

        # Cross mode pre-conditions action_feats once so both per-domain and shared
        # heads receive the same cross-attended tensor.
        conditioned_feats = action_feats
        if self.wm_prediction_mode == "cross":
            conditioned_feats = self._apply_action_cross_attention(
                domain, action_feats, future_feats
            )

        action_loss = torch.tensor(0.0, device=self.device)
        shared_action_loss = torch.tensor(0.0, device=self.device)
        auxiliary_action_loss = torch.tensor(0.0, device=self.device)

        if self.training_stage in (2, 3) and self._domain_has_action_supervision(
            domain
        ):
            action_loss += self._compute_action_head_loss(
                domain, data, conditioned_feats, future_feats
            )
            if self.shared_action:
                shared_action_loss = self.heads["shared"].compute_loss(
                    conditioned_feats, data
                )

        aux_per_key: Dict[str, torch.Tensor] = {}
        if self.training_stage in (1, 3):
            auxiliary_action_loss, aux_per_key = self._compute_joint_state_loss(
                domain,
                future_feats,
                data,
                action_features=action_feats,
                return_per_key=True,
            )

        total_loss = action_loss + shared_action_loss + auxiliary_action_loss
        self._last_loss_breakdown = {
            "action": action_loss.detach(),
            "shared": shared_action_loss.detach(),
            "aux": auxiliary_action_loss.detach(),
            "aux_per_key": aux_per_key,
            "mode": self.wm_prediction_mode,
            "stage": self.training_stage,
        }
        return total_loss

    def forward(self, domain, data, include_aux_predictions=True):
        """
        Forward pass of the HPTModel to compute actions.

        Parameters
        ----------
        domain : str
            The domain corresponding to the input data.
        data : dict
            Dictionary containing input data for various modalities.

        Returns
        -------
        torch.Tensor
            The predicted action output.
        """
        action_feats, future_feats, block_outputs = self.forward_features(domain, data)
        action = {}

        # Cross mode: pre-condition action_feats once. This must run independent
        # of the include_aux_predictions / emit_aux_at_eval gates below, since
        # future_feats is a trunk output, not an aux-head output.
        if self.wm_prediction_mode == "cross":
            conditioned_feats = self._apply_action_cross_attention(
                domain, action_feats, future_feats
            )
        else:
            conditioned_feats = action_feats

        pred_states = {}
        inference_features = conditioned_feats
        if self.wm_prediction_mode == "cross":
            # When emit_aux_at_eval is set, populate pred_states so the aux
            # emission loop can render flow / image-latent panels. Action head
            # input (inference_features) is unaffected — it still uses the
            # cross-attended `conditioned_feats`.
            if (
                include_aux_predictions
                and domain in self.auxiliary_ac_keys
                and getattr(self, "emit_aux_at_eval", False)
            ):
                pred_states = self._predict_state_targets(
                    domain, future_feats, action_feats, batch=data
                )
        elif self.wm_prediction_mode != "cross":
            if include_aux_predictions and domain in self.auxiliary_ac_keys:
                if self.wm_prediction_mode in ("autoregressive", "teacher_forcing"):
                    # Teacher forcing at inference falls back to autoregressive (no GT available).
                    pred_states = self._predict_state_targets(
                        domain, future_feats, action_feats, batch=data
                    )
                    inference_features = self._fuse_features_with_state(
                        action_feats, pred_states
                    )
                elif self.wm_prediction_mode == "joint":
                    pred_states = self._predict_state_targets(
                        domain, future_feats, action_feats, batch=data
                    )

        raw_features = future_feats if future_feats is not None else action_feats
        if self.diffusion:
            inference_features = (inference_features, domain)
            raw_features = (raw_features, domain)

        if domain in self.heads:
            action[domain] = self.heads[domain](inference_features)

        if self.shared_action:
            # Original code unconditionally wrapped as 2-tuple; preserve that contract
            # but feed the shared head the cross-conditioned features in cross mode.
            action["shared"] = self.heads["shared"]((conditioned_feats, domain))

        emit_aux = (
            include_aux_predictions
            and domain in self.auxiliary_ac_keys
            and (
                self.wm_prediction_mode != "cross"
                or getattr(self, "emit_aux_at_eval", False)
            )
        )
        if emit_aux:
            for key in self.auxiliary_ac_keys[domain]:
                head_key = self._resolve_aux_head_key(domain, key)
                if head_key in self.heads:
                    if key in pred_states:
                        action[key] = pred_states[key]
                    else:
                        head = self.heads[head_key]
                        if self._is_point_flow_head(head):
                            # Point heads need dict cond with per-sample current_points +
                            # validity_mask. _predict_state_targets was called with batch,
                            # so pred_states should hold the prediction; if we got here
                            # without it, skip rather than crash.
                            continue
                        # raw_features is (tensor, domain) when self.diffusion
                        action[key] = head(raw_features)

        return action

    def save(self, checkpoint_path="./.checkpoints/hpt/full/"):
        """
        Save the state of the HPTModel to a specified checkpoint path.

        Parameters
        ----------
        checkpoint_path : str, optional
            The path to save the checkpoint (default is "./.checkpoints/hpt/full/").
        """
        try:
            torch.save(self.state_dict(), checkpoint_path)
        except FileNotFoundError:
            print(f"Could not save module parameters for trunk to {checkpoint_path}.")

    def _init_weights(self, m):
        """
        Initialize weights of a module using Xavier uniform initialization for Linear layers and constant
        initialization for LayerNorm layers.

        Parameters
        ----------
        m : nn.Module
            The module to initialize.
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def freeze_action_head(self):
        """Freeze the primary action head and all shared action heads."""
        for domain in list(self.heads.keys()):
            if not any(
                domain.endswith(f"_{k}")
                for keys in (self.auxiliary_ac_keys or {}).values()
                for k in keys
            ):
                for param in self.heads[domain].parameters():
                    param.requires_grad = False

    def unfreeze_action_head(self):
        """Unfreeze the primary action head and all shared action heads."""
        for domain in list(self.heads.keys()):
            if not any(
                domain.endswith(f"_{k}")
                for keys in (self.auxiliary_ac_keys or {}).values()
                for k in keys
            ):
                for param in self.heads[domain].parameters():
                    param.requires_grad = True

    def _iter_aux_head_names(self):
        """Yield every unique head name actually wired for aux loss.

        Resolves shared heads via _resolve_aux_head_key so freeze/unfreeze and
        other bookkeeping covers the cotrain case where the registered head is
        named f"shared_{key}" rather than f"{domain}_{key}".
        """
        seen = set()
        for domain, key_list in (self.auxiliary_ac_keys or {}).items():
            for key in key_list:
                head_key = self._resolve_aux_head_key(domain, key)
                if head_key in self.heads and head_key not in seen:
                    seen.add(head_key)
                    yield head_key

    def freeze_auxiliary_heads(self):
        """Freeze all auxiliary (future-prediction) heads (domain-specific + shared)."""
        for head_key in self._iter_aux_head_names():
            for param in self.heads[head_key].parameters():
                param.requires_grad = False

    def unfreeze_auxiliary_heads(self):
        """Unfreeze all auxiliary (future-prediction) heads (domain-specific + shared)."""
        for head_key in self._iter_aux_head_names():
            for param in self.heads[head_key].parameters():
                param.requires_grad = True

    def set_training_stage(self, stage: int):
        """
        Configure which heads are trained.

        Parameters
        ----------
        stage : int
            1 = auxiliary heads only (trunk + stems + aux heads; action head frozen)
            2 = action head only (trunk + stems + action head; aux heads frozen)
            3 = all heads jointly (default)
        """
        if stage == 1:
            self.unfreeze_auxiliary_heads()
            self.freeze_action_head()
        elif stage == 2:
            self.unfreeze_action_head()
            self.freeze_auxiliary_heads()
        elif stage == 3:
            self.unfreeze_action_head()
            self.unfreeze_auxiliary_heads()
        else:
            raise ValueError(f"Invalid training_stage={stage}. Expected 1, 2, or 3.")
        self.training_stage = stage
        from termcolor import cprint

        cprint(
            f"[HPTModel] training_stage set to {stage} "
            f"({'aux-only' if stage == 1 else 'action-only' if stage == 2 else 'joint'})",
            color="cyan",
        )

    def freeze_trunk(self, num_layers=0):
        """
        Freeze a specified number of layers in the transformer trunk to prevent them from updating during training.

        Parameters
        ----------
        num_layers : int, optional
            The number of layers to freeze from the end of the trunk (default is 0).
        """
        layers = list(self.trunk["trunk"].children())
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = False

    def unfreeze_trunk(self, num_layers=0):
        """
        Unfreeze a specified number of layers in the transformer trunk to allow them to update during training.

        Parameters
        ----------
        num_layers : int, optional
            The number of layers to unfreeze from the end of the trunk (default is 0).
        """
        layers = list(self.trunk["trunk"].children())
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

    def load_trunk(self, path):
        """
        Load the transformer trunk state from a given file path or a HuggingFace URL.

        Parameters
        ----------
        path : str
            The file path or HuggingFace identifier (prefixed with "hf://") from which to load the trunk state.
        """
        if "hf://" in path:
            if "output" in path:
                path = path.replace("output/", "")
            path = download_from_huggingface(path[len("hf://") :])
        self.trunk.load_state_dict(torch.load(path), strict=True)

    def load_pretrained(self, checkpoint_path):
        """
        Load pretrained trunk weights from a specified checkpoint directory or HuggingFace URL.

        Parameters
        ----------
        checkpoint_path : str
            The path or HuggingFace identifier (prefixed with "hf://") for the pretrained checkpoint.
        """
        if not os.path.exists(checkpoint_path):
            checkpoint_path = download_from_huggingface(checkpoint_path[len("hf://") :])

        self.load_trunk(os.path.join(checkpoint_path, "trunk.pth"))


class HPT(Algo):
    """ """

    _DEFAULT_WM_VISUAL_OBS_KEYS = {
        "front_img_1",
        "left_wrist_img",
        "right_wrist_img",
    }

    def _apply_checkpoint_backward_compatibility(self):
        # Older HPT-WM checkpoints were saved before vae_stem_projection existed.
        # Those checkpoints should keep the pre-projector behavior, which was
        # equivalent to truncating/padding the flattened VAE latent.
        projection = getattr(self, "vae_stem_projection", None)
        if projection not in ("truncate", "linear"):
            self.vae_stem_projection = "truncate"

        if not hasattr(self, "_vae_latent_dim"):
            self._vae_latent_dim = None

        if hasattr(self, "_vae_stem_camera_keys"):
            self._vae_stem_camera_keys = set(self._vae_stem_camera_keys)
        else:
            self._vae_stem_camera_keys = set()

        if hasattr(self, "_wm_visual_obs_keys"):
            self._wm_visual_obs_keys = set(self._wm_visual_obs_keys)
        else:
            self._wm_visual_obs_keys = set(self._DEFAULT_WM_VISUAL_OBS_KEYS)

        if not hasattr(self, "_future_latent_layout"):
            self._future_latent_layout = {}

        if not hasattr(self, "_has_future_image_head"):
            self._has_future_image_head = any(
                getattr(self, "future_image_latent_key", "future_image_latent") in keys
                for _, keys in getattr(self, "auxiliary_ac_keys", {}).items()
            )

        if not hasattr(self, "_has_future_flow_head"):
            self._has_future_flow_head = any(
                getattr(self, "future_flow_key", "future_flow") in keys
                for _, keys in getattr(self, "auxiliary_ac_keys", {}).items()
            )

        if not hasattr(self, "_has_point_flow_head"):
            self._has_point_flow_head = any(
                k in keys
                for _, keys in getattr(self, "auxiliary_ac_keys", {}).items()
                for k in ("future_flow_2d_points", "future_flow_3d_points")
            )

        if not hasattr(self, "shared_auxiliary_ac_keys"):
            self.shared_auxiliary_ac_keys = None

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._apply_checkpoint_backward_compatibility()

    def __init__(
        self,
        data_schematic,
        camera_transforms,
        # ---------------------------
        # Image augmentations
        # ---------------------------
        train_image_augs,
        eval_image_augs,
        # ---------------------------
        # Trunk params
        # ---------------------------
        trunk: dict = None,
        # ---------------------------
        # Other model params
        # ---------------------------
        stem_specs: dict = None,
        head_specs: dict = None,
        shared_stem_specs: dict = None,
        shared_obs_keys: list = None,
        encoder_specs: dict = None,
        domains: list = None,
        action_domains: Optional[List[str]] = None,
        auxiliary_ac_keys: dict = {},
        auxiliary_loss_weights: dict = {},
        shared_auxiliary_ac_keys: Optional[List[str]] = None,
        future_proprio_key: str = "future_ee_pose",
        future_image_keys: Optional[List[str]] = None,
        future_image_latent_key: str = "future_image_latent",
        future_flow_key: str = "future_flow",
        future_image_resize: int = 128,
        cosmos_vae=None,
        wm_prediction_mode: str = "autoregressive",
        teacher_forcing_ratio: float = 1.0,
        use_shared_vae_stem: bool = False,
        vae_stem_projection: str = "truncate",
        training_stage: int = 3,
        # ---------------------------
        # Cross-attention mode (action_feats <-> trunk future_feats)
        # ---------------------------
        cross_attn: Optional[dict] = None,
        emit_aux_at_eval: bool = False,
        # ---------------------------
        # Pretrained
        # ---------------------------
        pretrained: bool = False,
        pretrained_checkpoint: str = "",
        # ---------------------------
        # Catch-all kwargs
        # ---------------------------
        **kwargs,
    ):
        self.nets = nn.ModuleDict()
        self.data_schematic = data_schematic

        self.camera_transforms = camera_transforms
        self.train_image_augs = train_image_augs
        self.eval_image_augs = eval_image_augs
        self.stem_specs = stem_specs
        self.head_specs = head_specs
        self.encoders = encoder_specs

        self.shared_stem_specs = shared_stem_specs
        self.shared_obs_keys = shared_obs_keys

        self.pretrained = pretrained
        self.pretrained_checkpoint = pretrained_checkpoint

        self.domains = domains.copy()
        # None => every domain gets action supervision (legacy). A list restricts
        # the action (BC) loss to those domains; others stay flow/aux-only.
        self.action_domains = None if action_domains is None else list(action_domains)
        self.auxiliary_ac_keys = auxiliary_ac_keys.copy()
        self.auxiliary_loss_weights = auxiliary_loss_weights.copy()
        self.shared_auxiliary_ac_keys = (
            None if shared_auxiliary_ac_keys is None else list(shared_auxiliary_ac_keys)
        )
        self.shared_ac_key = kwargs.get("shared_ac_key", None)
        self.is_6dof = kwargs.get("6dof", False)
        self.kinematics_solver = kwargs.get("kinematics_solver", None)
        self.future_proprio_key = future_proprio_key
        self.future_image_keys = future_image_keys or []
        self.future_image_latent_key = future_image_latent_key
        self.future_flow_key = future_flow_key
        self.future_image_resize = future_image_resize
        self.cosmos_vae = cosmos_vae
        self.wm_prediction_mode = wm_prediction_mode
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.use_shared_vae_stem = use_shared_vae_stem
        self.vae_stem_projection = vae_stem_projection
        self.training_stage = training_stage
        if self.vae_stem_projection not in ("truncate", "linear"):
            raise ValueError(
                f"Invalid vae_stem_projection={self.vae_stem_projection!r}. "
                "Expected 'truncate' or 'linear'."
            )
        if self.wm_prediction_mode not in (
            "autoregressive",
            "joint",
            "teacher_forcing",
            "cross",
        ):
            raise ValueError(
                f"Invalid wm_prediction_mode={self.wm_prediction_mode}. "
                "Expected one of: autoregressive, joint, teacher_forcing, cross."
            )
        if self.wm_prediction_mode == "cross":
            trunk_token_post = trunk.get("token_postprocessing", "action_token")
            if trunk_token_post != "joint_token":
                raise ValueError(
                    "wm_prediction_mode='cross' requires trunk.token_postprocessing="
                    "'joint_token' so that future_tokens are produced. Got "
                    f"token_postprocessing={trunk_token_post!r}."
                )
        if self.training_stage not in (1, 2, 3):
            raise ValueError(
                f"Invalid training_stage={self.training_stage}. Expected 1, 2, or 3."
            )
        self._vae_stem_camera_keys = set()
        self._wm_visual_obs_keys = set(
            ["front_img_1", "left_wrist_img", "right_wrist_img"]
        )
        self._future_latent_layout = {}
        self._has_future_image_head = any(
            self.future_image_latent_key in keys
            for _, keys in self.auxiliary_ac_keys.items()
        )
        self._has_future_flow_head = any(
            self.future_flow_key in keys for _, keys in self.auxiliary_ac_keys.items()
        )
        self._has_point_flow_head = any(
            k in keys
            for _, keys in self.auxiliary_ac_keys.items()
            for k in ("future_flow_2d_points", "future_flow_3d_points")
        )

        model = HPTModel(**trunk)
        model.action_domains = self.action_domains
        model.auxiliary_ac_keys = self.auxiliary_ac_keys
        model.auxiliary_loss_weights = self.auxiliary_loss_weights
        model.shared_auxiliary_ac_keys = self.shared_auxiliary_ac_keys
        model.wm_prediction_mode = self.wm_prediction_mode
        model.teacher_forcing_ratio = self.teacher_forcing_ratio

        self.multitask = kwargs.get("multitask", False)
        self.device = kwargs.get(
            "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.cosmos_vae is not None:
            self.cosmos_vae = self.cosmos_vae.to(self.device)
            self.cosmos_vae.eval()
        model.device = self.device

        self.diffusion = kwargs.get("diffusion", False)
        model.diffusion = self.diffusion

        if self.diffusion:
            if self.data_schematic.norm_mode == "zscore":
                cprint(
                    "WARNING: HPTModel with diffusion / flow matching is using 'zscore' normalization. "
                    "Consider switching to 'minmax' or 'quantile' norm_mode in train.yaml for better stability",
                    color="yellow",
                    attrs=["bold"],
                )

        if self.pretrained:
            model.load_pretrained(self.pretrained_checkpoint)

        if self.shared_obs_keys is not None:
            model.init_domain_stem("shared", self.shared_stem_specs)
            model.shared_keys = self.shared_obs_keys

        for domain in self.domains:
            if self.stem_specs[domain]:
                model.init_domain_stem(domain, self.stem_specs[domain])
            if self.head_specs[domain]:
                model.init_domain_head(domain, self.head_specs[domain])

        if self.shared_ac_key is not None:
            domain = "shared"
            model.shared_action = True
            model.init_domain_head(domain, self.head_specs[domain])

        initialized_shared_aux_heads = set()
        sal = self.shared_auxiliary_ac_keys
        for domain, key_list in self.auxiliary_ac_keys.items():
            for key in key_list:
                shared_domain_key = f"shared_{key}"
                if sal is not None:
                    use_shared = key in sal and shared_domain_key in self.head_specs
                else:
                    use_shared = shared_domain_key in self.head_specs
                if use_shared:
                    if shared_domain_key not in initialized_shared_aux_heads:
                        model.init_domain_head(
                            shared_domain_key, self.head_specs[shared_domain_key]
                        )
                        initialized_shared_aux_heads.add(shared_domain_key)
                    continue

                domain_key = f"{domain}_{key}"
                model.init_domain_head(domain_key, self.head_specs[domain_key])

        for modality, encoder_cfg in self.encoders.items():
            if self.use_shared_vae_stem and modality in self._wm_visual_obs_keys:
                self._vae_stem_camera_keys.add(modality)
                continue
            model.init_encoder(modality, encoder_cfg)

        model.finalize_modules()

        # VAE-stem projectors need a dummy encode to know latent_dim; deferred to after
        # the model is placed on device (see _build_vae_stem_projectors below).
        self._vae_latent_dim = None  # populated lazily on first encode

        self.ac_keys = {}
        self.camera_keys = {}
        self.proprio_keys = {}
        self.lang_keys = {}

        self.freeze_repr = kwargs.get("freeze_repr", False)
        self.depth = kwargs.get("depth", 8)
        self.freeze_depth = kwargs.get("freeze_depth", 8)
        model.depth = self.depth

        self.rkl_samples = kwargs.get("reverse_kl_samples", 4)

        self.ac_keys = kwargs.get("ac_keys", {})

        for embodiment in self.domains:
            embodiment_id = get_embodiment_id(embodiment)
            self.camera_keys[embodiment_id] = []
            self.proprio_keys[embodiment_id] = []
            self.lang_keys[embodiment_id] = []
            for key in data_schematic.keys_of_type("action_keys", embodiment_id):
                if (
                    data_schematic.is_key_with_embodiment(key, embodiment_id)
                    and key == self.ac_keys[embodiment]
                ):
                    self.ac_keys[embodiment_id] = key
            for key in data_schematic.keys_of_type("camera_keys", embodiment_id):
                if data_schematic.is_key_with_embodiment(key, embodiment_id):
                    self.camera_keys[embodiment_id].append(key)
            for key in data_schematic.keys_of_type("proprio_keys", embodiment_id):
                if data_schematic.is_key_with_embodiment(key, embodiment_id):
                    self.proprio_keys[embodiment_id].append(key)
            for key in data_schematic.keys_of_type("lang_keys", embodiment_id):
                if data_schematic.is_key_with_embodiment(key, embodiment_id):
                    self.lang_keys[embodiment_id].append(key)

        model.finalize_modules()

        # Inject the frozen Cosmos VAE into any FutureImageRegressionHead so that
        # it can decode latents during loss computation.
        if self.cosmos_vae is not None:
            for domain, key_list in self.auxiliary_ac_keys.items():
                for key in key_list:
                    domain_key = model._resolve_aux_head_key(domain, key)
                    if domain_key in model.heads and hasattr(
                        model.heads[domain_key], "cosmos_vae"
                    ):
                        model.heads[domain_key].cosmos_vae = self.cosmos_vae

        # Cross-mode plumbing: cross-attend action_feats to trunk's future_feats.
        # Modules attached on `model` (HPTModel) — not the wrapper — so DDP and
        # `.to(device)` see them. Re-apply _init_weights since these are attached
        # AFTER finalize_modules().
        if self.wm_prediction_mode == "cross":
            cross_attn_kwargs = dict(cross_attn or {})
            embed_dim = model.embed_dim
            model.action_cross_attn = nn.ModuleDict(
                {
                    domain: CrossAttention(query_dim=embed_dim, **cross_attn_kwargs)
                    for domain in self.auxiliary_ac_keys
                }
            )
            model.action_cross_gate = nn.ParameterDict(
                {
                    domain: nn.Parameter(torch.zeros(()))
                    for domain in self.auxiliary_ac_keys
                }
            )
            model.action_cross_attn.apply(model._init_weights)
        model.emit_aux_at_eval = emit_aux_at_eval

        self.nets["policy"] = model
        self.nets = self.nets.float().to(self.device)

        # Share the single frozen DINOv2 (inside cosmos_vae) with any DINOv2Stem encoder.
        # Done after device placement so the (plain-list, unregistered) backbone reference
        # is final; nets.float() does not traverse it, so DINOv2Stem.forward casts inputs.
        if self.cosmos_vae is not None:
            for enc in model.encoders.values():
                if hasattr(enc, "set_backbone"):
                    enc.set_backbone(self.cosmos_vae.model.encoder)

        # Build learned VAE→trunk projectors after device placement.
        if self.use_shared_vae_stem and self.vae_stem_projection == "linear":
            self._build_vae_stem_projectors(model.embed_dim)

        # Apply stage freeze/unfreeze after all parameters are on device
        model.set_training_stage(self.training_stage)

        self.training_step = 0

    @override
    def process_batch_for_training(self, batch, attach_future_targets=True):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader
        Returns:
            batch (dict): processed dict of batchs of form
                front_img_1 torch.Size([32, 3, 480, 640])
                right_wrist_img: torch.Size([32, 3, 480, 640])
                joint_positions: torch.Size([32, 1, 7])
                actions_joints_act: torch.Size([32, 100, 7])
                demo_number: torch.Size([32])
                _index: torch.Size([32])
                pad_mask: torch.Size([32, 100, 1])
                embodiment: torch.Size([])
        """
        processed_batch = {}
        for embodiment_name, _batch in batch.items():
            embodiment_id = get_embodiment_id(embodiment_name)
            processed_batch[embodiment_id] = {}
            for key, value in _batch.items():
                key_name = self.data_schematic.zarr_key_to_keyname(key, embodiment_id)
                if key_name is not None:
                    processed_batch[embodiment_id][key_name] = value

            ac_key = self.ac_keys[embodiment_id]
            if len(processed_batch[embodiment_id][ac_key].shape) != 3:
                raise ValueError("Action shape in batch is not 2")

            B, S, _ = processed_batch[embodiment_id][ac_key].shape
            device = processed_batch[embodiment_id][ac_key].device
            processed_batch[embodiment_id]["pad_mask"] = torch.ones(
                B, S, 1, device=device
            )

            processed_batch[embodiment_id] = self.data_schematic.normalize_data(
                processed_batch[embodiment_id], embodiment_id
            )
            processed_batch[embodiment_id]["embodiment"] = torch.tensor(
                [embodiment_id], device=self.device, dtype=torch.int64
            )
            # TODO make this work with any fp type
            for key, value in processed_batch[embodiment_id].items():
                if isinstance(value, torch.Tensor):
                    value = value.to(self.device)
                    if value.is_floating_point():
                        value = value.float()
                    processed_batch[embodiment_id][key] = value

            if attach_future_targets:
                self._attach_future_targets(
                    processed_batch[embodiment_id], embodiment_id
                )

        return processed_batch

    def _attach_future_targets(self, batch, embodiment_id):
        if (
            self.future_proprio_key in batch
            and batch[self.future_proprio_key].ndim == 2
        ):
            batch[self.future_proprio_key] = batch[self.future_proprio_key].unsqueeze(1)

        if self.future_flow_key in batch and batch[self.future_flow_key].ndim == 4:
            # [B, 2, 28, 40] -> [B, 1, 2*28*40=2240]
            flow = batch[self.future_flow_key]
            batch[self.future_flow_key] = flow.reshape(flow.shape[0], 1, -1)

        if (
            not self._has_future_image_head
            or self.cosmos_vae is None
            or len(self.future_image_keys) == 0
        ):
            return

        latents = []
        layout = []
        for cam_key in self.future_image_keys:
            if cam_key not in batch:
                continue
            image = batch[cam_key].float()
            if image.max() > 1.5:
                image = image / 255.0
            image = F.interpolate(
                image,
                size=(self.future_image_resize, self.future_image_resize),
                mode="bilinear",
                align_corners=False,
            )
            videos = image.unsqueeze(2)
            with torch.no_grad():
                latent = self.cosmos_vae.encode(videos)

            # Per-camera token: [B, 1, C_lat*H_lat*W_lat] (e.g. 4096 for 128px input)
            cam_idx = len(latents)
            flat = latent.reshape(latent.shape[0], 1, -1).detach()
            latents.append(flat)
            layout.append(
                (cam_key, cam_idx, latent.shape[1], latent.shape[3], latent.shape[4])
            )

        if len(latents) == 0:
            return
        # Stack per camera along dim=1: [B, N_cams, cam_lat_dim] (e.g. [B, 3, 4096])
        batch[self.future_image_latent_key] = torch.cat(latents, dim=1)
        self._future_latent_layout[embodiment_id] = layout

    def _restore_flow_spatial_dims(self, tensor):
        if (
            tensor is None
            or tensor.ndim != 3
            or tensor.shape[1] != 1
            or tensor.shape[2] != 2 * 28 * 40
        ):
            return tensor
        return tensor.reshape(tensor.shape[0], 2, 28, 40)

    def _prepare_eval_denorm_tensors(self, data):
        prepared = {}
        for key, tensor in data.items():
            if key == self.future_flow_key:
                prepared[key] = self._restore_flow_spatial_dims(tensor)
            else:
                prepared[key] = tensor
        return prepared

    @override
    def forward_training(self, batch):
        """
        One iteration of training. Sequentially, forward pass loss, Compute forward pass and compute losses.  Return predictions dictionary.  HPT also calculates loss here.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training (see docstring for expected keys/shapes)
        Returns:
            predictions (dict): {ac_key: torch.Tensor (B, Seq, D), loss_key_name: torch.Tensor (1)}
        """

        predictions = OrderedDict()
        hpt_batches = {}
        self.training_step += 1
        for (
            embodiment_id,
            _batch,
        ) in batch.items():
            embodiment_name = get_embodiment(embodiment_id).lower()
            cam_keys = self.camera_keys[embodiment_id]
            proprio_keys = self.proprio_keys[embodiment_id]
            lang_keys = self.lang_keys[embodiment_id]
            ac_key = self.ac_keys[embodiment_id]
            aux_ac_keys = self.auxiliary_ac_keys.get(embodiment_name, [])

            data = self._robomimic_to_hpt_data(
                _batch, cam_keys, proprio_keys, lang_keys, ac_key, aux_ac_keys
            )
            hpt_batch = {
                "domain": embodiment_name,  # readability on config side
                "data": data,
            }
            hpt_batches[embodiment_id] = self._clone_batch(hpt_batch)

            if self.freeze_repr:
                loss = self.nets["policy"].compute_loss_depth(
                    hpt_batch, depth=self.freeze_depth
                )
            else:
                loss = self.nets["policy"].compute_loss(hpt_batch)

            predictions[f"{embodiment_name}_{ac_key}"] = _batch[ac_key]
            predictions[f"{embodiment_name}_loss"] = loss
            breakdown = getattr(self.nets["policy"], "_last_loss_breakdown", {})
            if breakdown:
                predictions[f"{embodiment_name}_action_head_loss"] = breakdown.get(
                    "action", torch.tensor(0.0, device=self.device)
                )
                predictions[f"{embodiment_name}_aux_head_loss"] = breakdown.get(
                    "aux", torch.tensor(0.0, device=self.device)
                )

        return predictions

    @override
    def forward_eval(self, batch, include_aux_predictions=True):
        """
        Compute forward pass and return network outputs in @predictions dict.
        Unnormalize data here.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training (see docstring for expected keys/shapes)
        Returns:
            unnorm_preds (dict): {<embodiment_name>_<ac_key>: torch.Tensor (B, Seq, D)}
        """
        unnorm_preds = {}
        for embodiment_id, _batch in batch.items():
            embodiment_name = get_embodiment(embodiment_id).lower()
            cam_keys = self.camera_keys[embodiment_id]
            proprio_keys = self.proprio_keys[embodiment_id]
            lang_keys = self.lang_keys[embodiment_id]
            ac_key = self.ac_keys[embodiment_id]
            aux_ac_keys = self.auxiliary_ac_keys.get(embodiment_name, [])
            data = self._robomimic_to_hpt_data(
                _batch, cam_keys, proprio_keys, lang_keys, ac_key, aux_ac_keys
            )
            hpt_batch = {
                "domain": embodiment_name,  # readability on config side
                "data": data,
            }

            actions = self.nets["policy"].forward(
                hpt_batch["domain"],
                hpt_batch["data"],
                include_aux_predictions=include_aux_predictions,
            )
            predictions = OrderedDict()

            for key in actions:
                if key == embodiment_name:
                    pred = actions[embodiment_name]
                    ref = _batch[ac_key]
                    name = ac_key
                elif key == "shared":
                    pred = actions[key]
                    ref = _batch[self.shared_ac_key]
                    name = self.shared_ac_key
                else:
                    pred = actions[key]
                    ref = _batch[key]
                    name = key

                if ref.ndim == 4:
                    # Point-flow trajectory: (B, H, N, D). Keep 4D so the panel
                    # renderer and flow_stats denormalization get the native
                    # per-point trajectory shape.
                    assert (
                        pred.shape == ref.shape
                    ), f"Prediction shape {pred.shape} does not match reference shape {ref.shape} for key {key}"
                    predictions[name] = pred
                    continue

                if ref.ndim == 2:
                    ref = ref.unsqueeze(1)
                elif ref.ndim > 3:
                    ref = ref.reshape(ref.shape[0], 1, -1)

                B, T, D = ref.shape
                if D == 12 and pred.shape[-1] == 14:
                    pred = torch.cat([pred[..., :6], pred[..., 7:13]], dim=-1)
                pred = pred[:, :T]
                assert (
                    pred.shape == ref.shape
                ), f"Prediction shape {pred.shape} does not match reference shape {ref.shape} for key {key}"
                predictions[name] = pred

            unnorm_actions = self.data_schematic.unnormalize_data(
                self._prepare_eval_denorm_tensors(predictions), embodiment_id
            )
            for key in unnorm_actions:
                unnorm_preds[f"{embodiment_name}_{key}"] = unnorm_actions[key]

        return unnorm_preds

    @override
    def forward_eval_logging(self, batch):
        """
        Called by pl_model to generate a dictionary of metrics and an image visualization
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training (see docstring for expected keys/shapes)
        Returns:
            metrics (dict):
                metricname: value (float)
            image: (B, 3, H, W)
        """
        preds = self.forward_eval(batch)

        # Validation-time training-loss mirror: run the same compute_loss path used
        # at training so wandb gets `Valid/{embodiment}_*loss*` keys paired with
        # the `Train/...` ones. Uses the trunk's eval-mode behavior (no random
        # horizon masking) because compute_loss now ties train_mode to self.training.
        loss_breakdown_per_domain: Dict[int, dict] = {}
        with torch.no_grad():
            for embodiment_id, _batch in batch.items():
                embodiment_name = get_embodiment(embodiment_id).lower()
                cam_keys = self.camera_keys[embodiment_id]
                proprio_keys = self.proprio_keys[embodiment_id]
                lang_keys = self.lang_keys[embodiment_id]
                ac_key = self.ac_keys[embodiment_id]
                aux_ac_keys = self.auxiliary_ac_keys.get(embodiment_name, [])
                hpt_batch = {
                    "domain": embodiment_name,
                    "data": self._robomimic_to_hpt_data(
                        _batch, cam_keys, proprio_keys, lang_keys, ac_key, aux_ac_keys
                    ),
                }
                total = self.nets["policy"].compute_loss(hpt_batch)
                br = getattr(self.nets["policy"], "_last_loss_breakdown", {}) or {}
                loss_breakdown_per_domain[embodiment_id] = {
                    "total": float(total.detach()),
                    "action": float(br.get("action", torch.tensor(0.0))),
                    "shared": float(br.get("shared", torch.tensor(0.0))),
                    "aux": float(br.get("aux", torch.tensor(0.0))),
                    "aux_per_key": {
                        k: float(v) for k, v in (br.get("aux_per_key") or {}).items()
                    },
                }

        metrics = {}
        images_dict = {}
        mse = MeanSquaredError()
        for embodiment_id, _batch in batch.items():
            _batch = self.data_schematic.unnormalize_data(
                self._prepare_eval_denorm_tensors(_batch), embodiment_id
            )
            embodiment_name = get_embodiment(embodiment_id).lower()
            ac_key = self.ac_keys[embodiment_id]

            br = loss_breakdown_per_domain.get(embodiment_id)
            if br is not None:
                metrics[f"Valid/{embodiment_name}_loss_total"] = br["total"]
                metrics[f"Valid/{embodiment_name}_action_head_loss"] = br["action"]
                metrics[f"Valid/{embodiment_name}_aux_head_loss"] = br["aux"]
                if self.shared_ac_key is not None:
                    metrics[f"Valid/{embodiment_name}_shared_loss"] = br["shared"]
                for aux_key, v in br["aux_per_key"].items():
                    metrics[f"Valid/{embodiment_name}_aux/{aux_key}_loss"] = v
            if f"{embodiment_name}_{ac_key}" in preds and ac_key != self.shared_ac_key:
                metrics[f"Valid/{embodiment_name}_{ac_key}_paired_mse_avg"] = mse(
                    (preds[f"{embodiment_name}_{ac_key}"]).cpu(), _batch[ac_key].cpu()
                )
                metrics[f"Valid/{embodiment_name}_{ac_key}_final_mse_avg"] = mse(
                    (preds[f"{embodiment_name}_{ac_key}"][:, -1]).cpu(),
                    _batch[ac_key][:, -1].cpu(),
                )
                fd = frechet_gaussian_over_time(
                    preds[f"{embodiment_name}_{ac_key}"], _batch[ac_key]
                )
                metrics[f"Valid/{embodiment_name}_{ac_key}_frechet_gauss_avg"] = (
                    fd.mean().item()
                )
                metrics[f"Valid/{embodiment_name}_{ac_key}_frechet_gauss_min"] = (
                    fd.min().item()
                )
                metrics[f"Valid/{embodiment_name}_{ac_key}_frechet_gauss_max"] = (
                    fd.max().item()
                )

            if embodiment_name in self.auxiliary_ac_keys:
                for aux_key in self.auxiliary_ac_keys[embodiment_name]:
                    pred_key = f"{embodiment_name}_{aux_key}"
                    if pred_key not in preds:
                        continue

                    is_image_latent = (
                        aux_key == self.future_image_latent_key
                        and self.cosmos_vae is not None
                    )
                    is_flow = aux_key == self.future_flow_key

                    if is_image_latent:
                        # Future image prediction: per-camera pixel MSE/L1 + latent MSE
                        pred_latents = preds[pred_key].to(self.device)
                        gt_latents = _batch[aux_key].to(self.device)
                        pred_frames = self._decode_future_latents(
                            embodiment_id, pred_latents
                        )
                        gt_frames = self._decode_future_latents(
                            embodiment_id, gt_latents
                        )
                        if len(pred_frames) > 0 and len(gt_frames) > 0:
                            total_pixel_mse = 0.0
                            total_pixel_l1 = 0.0
                            n_cam = 0
                            for cam_idx, cam_key in enumerate(self.future_image_keys):
                                if (
                                    cam_key not in pred_frames
                                    or cam_key not in gt_frames
                                ):
                                    continue
                                # Pixel-space: decoder output is in [0, 1]
                                p_img = (
                                    pred_frames[cam_key].float().clamp(0.0, 1.0).cpu()
                                )
                                g_img = gt_frames[cam_key].float().clamp(0.0, 1.0).cpu()
                                cam_mse = F.mse_loss(p_img, g_img).item()
                                cam_l1 = F.l1_loss(p_img, g_img).item()
                                # Short camera name for metric key
                                cam_short = (
                                    cam_key.replace("future_", "")
                                    .replace("_img_1", "")
                                    .replace("_img", "")
                                )
                                metrics[f"Valid/{pred_key}_pixel_mse_{cam_short}"] = (
                                    cam_mse
                                )
                                metrics[f"Valid/{pred_key}_pixel_l1_{cam_short}"] = (
                                    cam_l1
                                )
                                # Latent-space MSE per camera token
                                if (
                                    cam_idx < pred_latents.shape[1]
                                    and cam_idx < gt_latents.shape[1]
                                ):
                                    lat_mse = F.mse_loss(
                                        pred_latents[:, cam_idx, :].float(),
                                        gt_latents[:, cam_idx, :].float(),
                                    ).item()
                                    metrics[
                                        f"Valid/{pred_key}_latent_mse_{cam_short}"
                                    ] = lat_mse
                                total_pixel_mse += cam_mse
                                total_pixel_l1 += cam_l1
                                n_cam += 1
                            if n_cam > 0:
                                metrics[f"Valid/{pred_key}_pixel_mse_avg"] = (
                                    total_pixel_mse / n_cam
                                )
                                metrics[f"Valid/{pred_key}_pixel_l1_avg"] = (
                                    total_pixel_l1 / n_cam
                                )
                    else:
                        # Future proprio or other aux heads: MSE + L1 only
                        metrics[f"Valid/{pred_key}_paired_mse_avg"] = mse(
                            preds[pred_key].cpu(), _batch[aux_key].cpu()
                        )
                        metrics[f"Valid/{pred_key}_paired_l1_avg"] = F.l1_loss(
                            preds[pred_key].float().cpu(),
                            _batch[aux_key].float().cpu(),
                        ).item()

                        if is_flow:
                            # Flow-specific metrics (already unnormalized)
                            metrics[f"Valid/{pred_key}_mse"] = metrics[
                                f"Valid/{pred_key}_paired_mse_avg"
                            ]
                            metrics[f"Valid/{pred_key}_l1"] = metrics[
                                f"Valid/{pred_key}_paired_l1_avg"
                            ]

            if (
                self.shared_ac_key
                and f"{embodiment_name}_{self.shared_ac_key}" in preds
            ):
                pred_key = f"{embodiment_name}_{self.shared_ac_key}"
                metrics[f"Valid/{pred_key}_paired_mse_avg"] = mse(
                    preds[pred_key].cpu(), _batch[self.shared_ac_key].cpu()
                )
                metrics[f"Valid/{pred_key}_final_mse_avg"] = mse(
                    preds[pred_key][:, -1].cpu(),
                    _batch[self.shared_ac_key][:, -1].cpu(),
                )
                fd = frechet_gaussian_over_time(
                    preds[pred_key], _batch[self.shared_ac_key]
                )
                metrics[f"Valid/{pred_key}_frechet_gauss_avg"] = fd.mean().item()
                metrics[f"Valid/{pred_key}_frechet_gauss_min"] = fd.min().item()
                metrics[f"Valid/{pred_key}_frechet_gauss_max"] = fd.max().item()

            if self.rkl_samples and self.rkl_samples > 1:
                hpt_batch = {
                    "domain": embodiment_name,
                    "data": self._robomimic_to_hpt_data(
                        batch[embodiment_id],
                        self.camera_keys[embodiment_id],
                        self.proprio_keys[embodiment_id],
                        self.lang_keys[embodiment_id],
                        ac_key,
                        self.auxiliary_ac_keys.get(embodiment_name, []),
                    ),
                }
                rkl_targets = []

                if (
                    f"{embodiment_name}_{ac_key}" in preds
                    and ac_key != self.shared_ac_key
                ):
                    rkl_targets.append(
                        (
                            f"{embodiment_name}_{ac_key}",
                            _batch[ac_key].to(self.device),
                            embodiment_name,
                        )
                    )

                if embodiment_name in self.auxiliary_ac_keys:
                    for aux_key in self.auxiliary_ac_keys[embodiment_name]:
                        # Skip reverse KL for future prediction heads; keep it only for action
                        if aux_key in (
                            self.future_proprio_key,
                            self.future_image_latent_key,
                            "future_flow_2d_points",
                            "future_flow_3d_points",
                        ):
                            continue
                        aux_pred_key = f"{embodiment_name}_{aux_key}"
                        if aux_pred_key in preds:
                            rkl_targets.append(
                                (aux_pred_key, _batch[aux_key].to(self.device), aux_key)
                            )

                if self.shared_ac_key:
                    shared_pred_key = f"{embodiment_name}_{self.shared_ac_key}"
                    if shared_pred_key in preds:
                        rkl_targets.append(
                            (
                                shared_pred_key,
                                _batch[self.shared_ac_key].to(self.device),
                                "shared",
                            )
                        )

                M = int(self.rkl_samples)
                for pred_key_name, gt_tensor, head_key in rkl_targets:
                    samples = self._collect_policy_samples(
                        hpt_batch, ref=gt_tensor, key_name=head_key, M=M
                    )
                    rkl = reverse_kl_from_samples(samples, gt_tensor)
                    metrics[f"Valid/{pred_key_name}_reverse_kl_M{M}"] = rkl.item()

            # Build separate visualizations for action predictions and future image predictions
            # Action video visualization
            ims_action = self.visualize_preds(preds, _batch)

            # Future video visualization
            viz_img_key = self.data_schematic.viz_img_key()[embodiment_id]
            base_ims = (
                _batch[viz_img_key].cpu().numpy().transpose((0, 2, 3, 1)) * 255
            ).astype(np.uint8)
            ims_future = self._append_future_frame_panel(
                base_ims, embodiment_id, preds, _batch
            )

            images_dict[embodiment_id] = {
                "action": ims_action,
            }
            if ims_future is not None:
                images_dict[embodiment_id]["future"] = ims_future
            if self._has_future_flow_head:
                images_dict[embodiment_id]["flow"] = self._build_flow_panel(
                    base_ims, embodiment_id, preds, _batch
                )
            if self._has_point_flow_head:
                point_panel = self._build_point_flow_panel(
                    base_ims, embodiment_id, preds, _batch
                )
                if point_panel is not None:
                    images_dict[embodiment_id]["point_flow"] = point_panel
        return metrics, images_dict

    def _decode_future_latents(self, embodiment_id, latents):
        layout = self._future_latent_layout.get(embodiment_id, [])
        decoded = {}
        if self.cosmos_vae is None or len(layout) == 0:
            return decoded
        # latents: [B, N_cams, cam_lat_dim] — per-camera token format
        for cam_key, cam_idx, channels, h_lat, w_lat in layout:
            cam_latent = latents[:, cam_idx, :]  # [B, cam_lat_dim]
            cam_latent = cam_latent.reshape(latents.shape[0], channels, 1, h_lat, w_lat)
            with torch.no_grad():
                rec = self.cosmos_vae.decode(cam_latent)
            decoded[cam_key] = rec[:, :, 0]
        return decoded

    def _append_future_frame_panel(self, base_ims, embodiment_id, predictions, batch):
        pred_key = (
            f"{get_embodiment(embodiment_id).lower()}_{self.future_image_latent_key}"
        )
        if (
            self.future_image_latent_key not in batch
            or pred_key not in predictions
            or self.cosmos_vae is None
        ):
            return None

        gt_latents = batch[self.future_image_latent_key]
        pred_latents = predictions[pred_key]
        pred_latents = pred_latents[:, : gt_latents.shape[1], : gt_latents.shape[2]]
        gt_frames = self._decode_future_latents(embodiment_id, gt_latents)
        pred_frames = self._decode_future_latents(embodiment_id, pred_latents)
        if len(gt_frames) == 0:
            return None

        # Build a cam_key -> cam_idx lookup from the stored layout
        layout = self._future_latent_layout.get(embodiment_id, [])
        key_to_cam_idx = {entry[0]: entry[1] for entry in layout}

        # Display order: Front | Left | Right
        # Determined by substring matching so it is robust to key naming conventions
        def _cam_order(cam_key):
            k = cam_key.lower()
            if "front" in k:
                return 0
            if "left" in k:
                return 1
            if "right" in k:
                return 2
            return 3  # unknown cameras go last

        ordered_keys = sorted(
            [k for k in self.future_image_keys if k in gt_frames and k in pred_frames],
            key=_cam_order,
        )

        if not ordered_keys:
            return None

        # Ensure even cell dimensions so H.264 encoder doesn't crash
        H_cell = int(base_ims.shape[1])
        W_cell = int(base_ims.shape[2])
        if H_cell % 2 != 0:
            H_cell -= 1
        if W_cell % 2 != 0:
            W_cell -= 1

        panels = []
        for cam_key in ordered_keys:
            cam_idx = key_to_cam_idx.get(cam_key)

            # Decode per-camera latent directly via cam_idx for both GT and Pred
            if cam_idx is not None and cam_idx < gt_latents.shape[1]:
                cam_entry = next((e for e in layout if e[0] == cam_key), None)
                if cam_entry is not None:
                    _, _, channels, h_lat, w_lat = cam_entry
                    B_sz = gt_latents.shape[0]

                    def _decode_cam(latents_tensor):
                        lat = latents_tensor[:, cam_idx, :]
                        lat = lat.reshape(B_sz, channels, 1, h_lat, w_lat)
                        with torch.no_grad():
                            decoded = self.cosmos_vae.decode(lat)
                        return decoded[:, :, 0]  # [B, 3, H_dec, W_dec]

                    gt_img = _decode_cam(gt_latents)
                    pred_img = _decode_cam(pred_latents)
                else:
                    gt_img = gt_frames[cam_key]
                    pred_img = pred_frames[cam_key]
            else:
                gt_img = gt_frames[cam_key]
                pred_img = pred_frames[cam_key]

            # Resize both to even (H_cell, W_cell)
            gt_img = F.interpolate(
                gt_img.float().clamp(0.0, 1.0),
                size=(H_cell, W_cell),
                mode="bilinear",
                align_corners=False,
            )
            pred_img = F.interpolate(
                pred_img.float().clamp(0.0, 1.0),
                size=(H_cell, W_cell),
                mode="bilinear",
                align_corners=False,
            )

            # [B, 3, H_cell, 2*W_cell] -> then to numpy uint8
            pair_t = torch.cat([gt_img, pred_img], dim=3)
            pair_np = (
                pair_t.clamp(0.0, 1.0).cpu().numpy().transpose(0, 2, 3, 1) * 255
            ).astype(np.uint8)
            panels.append(pair_np)

        if not panels:
            return None

        # Final horizontal concatenation:
        # [front GT | front Pred | left GT | left Pred | right GT | right Pred]
        future_panel = np.concatenate(panels, axis=2)
        return future_panel

    def _build_point_flow_panel(self, base_ims, embodiment_id, predictions, batch):
        """Overlay GT vs Pred sparse point-flow motion on the base frame.

        For each kept valid point at pixel anchor (u, v), draws a GT arrow (cyan)
        and a Pred arrow (magenta) from (u, v) to (u + du, v + dv). du, dv come
        from the denormalized end-to-start delta of the trajectory. For 3D
        heads the world-frame XY delta is scaled into pixels via a heuristic so
        arrows are visible — compare direction + relative magnitude, not
        absolute pixel correspondence.

        Returns (B, H, W*K, 3) uint8 where K is the number of active point-flow
        heads (1 or 2), or None if nothing is wired.
        """
        import logging as _logging

        from egowam.rldb.zarr.flow_normalization import (
            denormalize_flow,
            get_flow_stats,
            has_flow_stats,
        )

        _log = _logging.getLogger(__name__)
        emb_name = get_embodiment(embodiment_id).lower()
        active = []
        for flow_key, dim in (
            ("future_flow_2d_points", 2),
            ("future_flow_3d_points", 3),
        ):
            pred_key = f"{emb_name}_{flow_key}"
            if flow_key not in batch:
                _log.warning(
                    "[point_flow_panel] skip %s: key not in batch (keys=%s)",
                    flow_key,
                    sorted(k for k in batch.keys() if "flow" in str(k))[:10],
                )
                continue
            if pred_key not in predictions:
                _log.warning(
                    "[point_flow_panel] skip %s: %s not in predictions (preds=%s)",
                    flow_key,
                    pred_key,
                    sorted(predictions.keys())[:10],
                )
                continue
            uv_key = f"{flow_key}__sampled_uv"
            if uv_key not in batch:
                _log.warning(
                    "[point_flow_panel] skip %s: %s not in batch",
                    flow_key,
                    uv_key,
                )
                continue
            stats_key = f"{emb_name}.{flow_key}"
            if not has_flow_stats(stats_key):
                _log.warning(
                    "[point_flow_panel] skip %s: flow_stats not registered for %s",
                    flow_key,
                    stats_key,
                )
                continue
            active.append(
                (
                    flow_key,
                    dim,
                    batch[flow_key],
                    predictions[pred_key],
                    batch.get(f"{flow_key}__validity_mask"),
                    batch[uv_key],
                    get_flow_stats(stats_key),
                    batch.get(f"{flow_key}__current_points"),
                )
            )
        if not active:
            return None

        B, H, W = base_ims.shape[:3]
        gt_color = (0, 255, 255)  # cyan for GT
        pred_color = (255, 0, 255)  # magenta for Pred
        scale_3d = 150.0  # heuristic m → px for 3D-XY arrows

        def _draw_arrows(bg, uv_px, delta_px, color, label):
            img = bg.copy()
            for i in range(uv_px.shape[0]):
                u, v = int(uv_px[i, 0]), int(uv_px[i, 1])
                du, dv = int(delta_px[i, 0]), int(delta_px[i, 1])
                cv2.arrowedLine(img, (u, v), (u + du, v + dv), color, 1, tipLength=0.25)
            cv2.putText(
                img,
                label,
                (5, 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            return img

        panels = []
        for b in range(B):
            per_head = []
            for (
                flow_key,
                dim,
                gt,
                pred,
                validity,
                sampled_uv,
                stats,
                current_points,
            ) in active:
                # Denormalize: NormalizeFlowDelta subtracted traj[0] and
                # normalized; absolute endpoint delta is (traj[-1] - traj[0])
                # which equals denormalized(traj[-1]) if we undo the norm.
                # Shape: (T, N, D)
                gt_b = denormalize_flow(gt[b].detach().cpu(), stats).numpy()
                pred_b = denormalize_flow(pred[b].detach().cpu(), stats).numpy()
                uv_b = sampled_uv[b].detach().cpu().numpy().astype(np.int64)
                n_points = gt_b.shape[1]
                if validity is not None:
                    valid_b = validity[b].detach().cpu().numpy().astype(bool)
                else:
                    valid_b = np.ones(n_points, dtype=bool)
                valid_idx = np.flatnonzero(valid_b)
                if valid_idx.size == 0:
                    # Degenerate frame — emit GT/Pred copies of the base image
                    # so the panel has a consistent layout across samples.
                    per_head.append(np.concatenate([base_ims[b], base_ims[b]], axis=1))
                    continue

                # End-to-start delta per point (all valid points, no subsample)
                gt_delta = gt_b[-1, valid_idx, :] - gt_b[0, valid_idx, :]
                pred_delta = pred_b[-1, valid_idx, :] - pred_b[0, valid_idx, :]
                delta_in_px = dim == 2
                if dim == 3:
                    # Project the 3D world flow into the image plane instead of
                    # the old world-XY*const heuristic (which clustered GT and
                    # sprayed Pred). Fit a per-sample affine camera from the
                    # absolute t0 3D points (__current_points) <-> their uv
                    # anchors, then push both GT and Pred endpoints through it.
                    cur3d = (
                        np.asarray(current_points[b].detach().cpu().numpy(), np.float64)
                        if current_points is not None
                        else None
                    )
                    if cur3d is not None and cur3d.shape[0] == n_points:
                        cv3 = cur3d[valid_idx]
                        uvv = uv_b[valid_idx].astype(np.float64)
                        A = np.concatenate([cv3, np.ones((cv3.shape[0], 1))], axis=1)
                        M, *_ = np.linalg.lstsq(A, uvv, rcond=None)  # (4, 2)

                        def _proj(p3, _M=M):
                            ones = np.ones((p3.shape[0], 1))
                            return np.concatenate([p3, ones], axis=1) @ _M

                        start2d = _proj(cv3)
                        gt_delta = _proj(cv3 + gt_delta) - start2d
                        pred_delta = _proj(cv3 + pred_delta) - start2d
                        delta_in_px = True  # projected deltas are in flow-source px
                    else:
                        gt_delta = gt_delta[:, :2] * scale_3d
                        pred_delta = pred_delta[:, :2] * scale_3d

                # Handle uv resolution mismatch: rescale anchors from the image
                # plane the flow was computed on to the displayed base frame.
                # Prefer the per-episode image_size (track4world metadata.json,
                # e.g. LIBERO 448x448) over the EVA-default constant (448x640).
                uv_hw_key = f"{flow_key}__image_hw"
                if batch.get(uv_hw_key) is not None:
                    _hw = batch[uv_hw_key]
                    _hw = _hw[b] if getattr(_hw, "ndim", 1) > 1 else _hw
                    _hw = (
                        _hw.detach().cpu().numpy()
                        if hasattr(_hw, "detach")
                        else np.asarray(_hw)
                    ).flatten()
                    uv_hw = (int(_hw[0]), int(_hw[1]))
                else:
                    uv_hw = (POINT_FLOW_2D_IMAGE_SHAPE[0], POINT_FLOW_2D_IMAGE_SHAPE[1])
                sx = W / float(uv_hw[1])
                sy = H / float(uv_hw[0])
                uv_sel = uv_b[valid_idx].astype(np.float64)
                uv_sel[:, 0] *= sx
                uv_sel[:, 1] *= sy
                if delta_in_px:
                    gt_delta = gt_delta * np.asarray([[sx, sy]])
                    pred_delta = pred_delta * np.asarray([[sx, sy]])

                gt_img = _draw_arrows(
                    base_ims[b],
                    uv_sel,
                    gt_delta,
                    gt_color,
                    f"{flow_key}  GT ({uv_sel.shape[0]} pts)",
                )
                pred_img = _draw_arrows(
                    base_ims[b],
                    uv_sel,
                    pred_delta,
                    pred_color,
                    f"{flow_key}  Pred ({uv_sel.shape[0]} pts)",
                )
                per_head.append(np.concatenate([gt_img, pred_img], axis=1))
            panels.append(np.concatenate(per_head, axis=1))

        return np.stack(panels, axis=0)

    def _build_flow_panel(self, base_ims, embodiment_id, predictions, batch):
        """
        Renders an arrow overlay video panel comparing GT vs predicted flow fields.
        The arrows are overlaid on the ground truth future frames.
        """
        pred_key = f"{get_embodiment(embodiment_id).lower()}_{self.future_flow_key}"
        if self.future_flow_key not in batch or pred_key not in predictions:
            return base_ims

        # [B, 2, 28, 40]
        gt_flow = batch[self.future_flow_key].reshape(-1, 2, 28, 40).cpu().numpy()
        pred_flow = predictions[pred_key].reshape(-1, 2, 28, 40).cpu().numpy()

        B = base_ims.shape[0]

        def _overlay_arrows(bg_ims, flow_uv, color=(255, 0, 255)):
            out_list = []
            for b in range(B):
                img = bg_ims[b].copy()
                if img.shape[0] % 2 != 0:
                    img = img[:-1, :, :]
                if img.shape[1] % 2 != 0:
                    img = img[:, :-1, :]
                h_img, w_img = img.shape[:2]
                _, h_flow, w_flow = flow_uv.shape[1:]

                step_y = h_img / h_flow
                step_x = w_img / w_flow

                # Draw arrows on a grid
                stride = 1
                for i in range(0, h_flow, stride):
                    for j in range(0, w_flow, stride):
                        start_pt = (int((j + 0.5) * step_x), int((i + 0.5) * step_y))
                        du, dv = flow_uv[b, 0, i, j], flow_uv[b, 1, i, j]
                        end_pt = (int(start_pt[0] + du), int(start_pt[1] + dv))

                        if np.linalg.norm([du, dv]) > 0.5:
                            cv2.arrowedLine(
                                img, start_pt, end_pt, color, 1, tipLength=0.3
                            )
                out_list.append(img)
            return np.stack(out_list, axis=0)

        gt_overlay = _overlay_arrows(
            base_ims, gt_flow, color=(255, 255, 0)
        )  # Cyan for GT
        pred_overlay = _overlay_arrows(
            base_ims, pred_flow, color=(255, 0, 255)
        )  # Purple for Pred

        # Concatenate side-by-side
        flow_panel = np.concatenate([gt_overlay, pred_overlay], axis=2)
        return flow_panel

    @override
    def visualize_preds(self, predictions, batch):
        """
        Helper function to visualize predictions on top of images
        Args:
            predictions (dict): {ac_key: torch.Tensor (B, Seq, D)}
            batch (dict): {ac_key: torch.Tensor (B, Seq, D), front_img_1: torch.Tensor (B, 3, H, W), embodiment: torch.Tensor (1)}
        Returns:
            ims (np.ndarray): (B, H, W, 3) - images with actions drawn on top
        """

        embodiment_id = batch["embodiment"][0].item()
        embodiment_name = get_embodiment(embodiment_id).lower()
        ac_key = self.ac_keys[embodiment_id]

        viz_img_key = self.data_schematic.viz_img_key()[embodiment_id]
        ims = (batch[viz_img_key].cpu().numpy().transpose((0, 2, 3, 1)) * 255).astype(
            np.uint8
        )
        for key in batch:
            if f"{embodiment_name}_{key}" in predictions:
                # Skip auxiliary future state predictions in the main action trajectory visualization
                if key.startswith("future"):
                    continue

                preds = predictions[f"{embodiment_name}_{key}"]
                gt = batch[key]

                # Only draw action-like predictions; skip latent/state heads (e.g. 12288-d image latent)
                if preds.shape[-1] not in (3, 6, 7, 12, 14):
                    continue

                if preds.shape[-1] == 12:
                    # pad to 14 for bimanual visualization
                    preds = torch.cat(
                        [
                            preds[..., :6],
                            torch.zeros_like(preds[..., :1]),
                            preds[..., 6:],
                            torch.zeros_like(preds[..., :1]),
                        ],
                        dim=-1,
                    )
                    gt = torch.cat(
                        [
                            gt[..., :6],
                            torch.zeros_like(gt[..., :1]),
                            gt[..., 6:],
                            torch.zeros_like(gt[..., :1]),
                        ],
                        dim=-1,
                    )

                if self.is_6dof and ac_key == "actions_cartesian":
                    gt, gt_rot = self._extract_xyz(gt)
                    preds, preds_rot = self._extract_xyz(preds)

                for b in range(ims.shape[0]):
                    if preds.shape[-1] == 7 or preds.shape[-1] == 14:
                        ac_type = "joints"
                    elif preds.shape[-1] == 3 or preds.shape[-1] == 6:
                        ac_type = "xyz"

                    # Determine arm from embodiment name, not action shape.
                    # Single-arm / uncalibrated embodiments (e.g. LIBERO sim) have
                    # no meaningful camera extrinsics here, so skip the action
                    # overlay and keep the raw frame rather than crashing. The WM
                    # future-frame panel is built separately and still logs.
                    if "bimanual" in embodiment_name:
                        arm = "both"
                    elif "left" in embodiment_name:
                        arm = "left"
                    elif "right" in embodiment_name:
                        arm = "right"
                    else:
                        arm = None
                    if arm is None or embodiment_name not in self.camera_transforms:
                        continue
                    ims[b] = draw_actions(
                        ims[b],
                        ac_type,
                        "Purples",
                        preds[b].cpu().numpy(),
                        self.camera_transforms[embodiment_name].extrinsics,
                        self.camera_transforms[embodiment_name].intrinsics,
                        arm=arm,
                        kinematics_solver=self.kinematics_solver,
                    )
                    ims[b] = draw_actions(
                        ims[b],
                        ac_type,
                        "Greens",
                        gt[b].cpu().numpy(),
                        self.camera_transforms[embodiment_name].extrinsics,
                        self.camera_transforms[embodiment_name].intrinsics,
                        arm=arm,
                        kinematics_solver=self.kinematics_solver,
                    )

                    if self.is_6dof and ac_key == "actions_cartesian":
                        ims[b] = draw_rotation_text(
                            ims[b], gt_rot[b][0], preds_rot[b][0], position=(340, 20)
                        )
        return ims

    @override
    def compute_losses(self, predictions, batch):
        """
        Compute losses based on network outputs in @predictions dict, using reference labels in @batch.
        Args:
            predictions (dict): dictionary containing network outputs, from @forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training (see docstring for expected keys/shapes)
        Returns:
            losses (dict): dictionary of losses computed over the batch
                loss_key_name: torch.Tensor (1)
        """
        total_action_loss = torch.tensor(0.0, device=self.device)
        loss_dict = OrderedDict()

        bc_weight = 1.0

        for embodiment_id, _batch in batch.items():
            embodiment_name = get_embodiment(embodiment_id).lower()
            bc_loss = predictions[f"{embodiment_name}_loss"]
            scaled_bc_loss = bc_weight * bc_loss
            total_action_loss += scaled_bc_loss
            loss_dict[f"{embodiment_name}_loss"] = bc_loss  # for logging
            if f"{embodiment_name}_action_head_loss" in predictions:
                loss_dict[f"{embodiment_name}_action_head_loss"] = predictions[
                    f"{embodiment_name}_action_head_loss"
                ]
            if f"{embodiment_name}_aux_head_loss" in predictions:
                loss_dict[f"{embodiment_name}_aux_head_loss"] = predictions[
                    f"{embodiment_name}_aux_head_loss"
                ]

        loss_dict["action_loss"] = total_action_loss / len(self.domains)
        return loss_dict

    @override
    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.
        Args:
            info (dict): dictionary of losses returned by compute_losses
                losses:
                    loss_key_name: torch.Tensor (1)
        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = OrderedDict()
        log["Loss"] = info["losses"]["action_loss"].item()
        for loss_key, loss in info["losses"].items():
            log[loss_key] = loss.item()
        return log

    @torch.no_grad()
    def _collect_policy_samples(self, hpt_batch, ref, key_name, M):
        """
        Collect policy samples for Reverse KL loss
        """
        point_flow = ref.ndim == 4
        if not point_flow:
            if ref.ndim == 2:
                ref = ref.unsqueeze(1)
            elif ref.ndim > 3:
                ref = ref.reshape(ref.shape[0], 1, -1)
            B, T, D = ref.shape
        samples = []
        was_training = self.nets.training
        self.nets.eval()
        for _ in range(M):
            out = self.nets["policy"].forward(
                hpt_batch["domain"], self._clone_batch(hpt_batch["data"])
            )
            if key_name in out:
                pred = out[key_name]
            else:
                pred = out[hpt_batch["domain"]]
            if not point_flow:
                if D == 12 and pred.shape[-1] == 14:
                    pred = torch.cat([pred[..., :6], pred[..., 7:13]], dim=-1)
                pred = pred[:, :T]
            assert (
                pred.shape == ref.shape
            ), f"Sampled prediction shape {pred.shape} does not match reference shape {ref.shape} for key {key_name}"
            # pred = pred[:, :T, :D]
            samples.append(pred.unsqueeze(0))
        if was_training:
            self.nets.train()
        return torch.cat(samples, dim=0)

    def _build_vae_stem_projectors(self, embed_dim: int):
        """
        Build one learned nn.Linear per VAE-stem camera: full_latent_dim → embed_dim.

        The latent dim is determined by running a single dummy encode with the frozen
        Cosmos VAE so we don't need to hard-code it in config.
        """
        if self.cosmos_vae is None or not self._vae_stem_camera_keys:
            return

        dummy = torch.zeros(
            1,
            3,
            self.future_image_resize,
            self.future_image_resize,
            device=self.device,
        )
        with torch.no_grad():
            dummy_lat = self.cosmos_vae.encode(dummy.unsqueeze(2))
        latent_dim = int(dummy_lat.reshape(1, -1).shape[-1])
        self._vae_latent_dim = latent_dim

        proj_dict = nn.ModuleDict(
            {
                cam_key.replace(".", "_"): nn.Linear(latent_dim, embed_dim)
                for cam_key in self._vae_stem_camera_keys
            }
        )
        self.nets["vae_stem_proj"] = proj_dict.float().to(self.device)
        cprint(
            f"[HPT] Built learned VAE stem projectors: {latent_dim} -> {embed_dim} "
            f"for cameras {sorted(self._vae_stem_camera_keys)}",
            color="cyan",
        )

    def _encode_current_visual_state(self, image, cam_key: str = "_default"):
        """
        Encode current observation image with shared Cosmos VAE and project to trunk embed dim.

        Projection mode is controlled by ``self.vae_stem_projection``:
          "truncate" (default) — keep/pad the first ``embed_dim`` dims of the flat latent.
          "linear"             — pass the full flat latent through a learned nn.Linear.
        """
        if self.cosmos_vae is None:
            raise RuntimeError(
                "use_shared_vae_stem=True requires cosmos_vae to be configured."
            )
        image = image.float()
        if image.max() > 1.5:
            image = image / 255.0
        image = F.interpolate(
            image,
            size=(self.future_image_resize, self.future_image_resize),
            mode="bilinear",
            align_corners=False,
        )
        videos = image.unsqueeze(2)  # [B, 3, H, W] -> [B, 1, 3, H, W]
        with torch.no_grad():
            latent = self.cosmos_vae.encode(videos)  # [B, 16, 16, 16]
        flat = latent.reshape(latent.shape[0], 1, -1)  # [B, 1, latent_dim]

        if self.vae_stem_projection == "linear":
            proj_key = cam_key.replace(".", "_")
            if (
                "vae_stem_proj" not in self.nets
                or proj_key not in self.nets["vae_stem_proj"]
            ):
                raise RuntimeError(
                    f"Learned VAE stem projector for camera '{cam_key}' not found. "
                    "Did _build_vae_stem_projectors run successfully?"
                )
            flat = self.nets["vae_stem_proj"][proj_key](flat)  # [B, 1, embed_dim]
        else:
            # "truncate": keep first embed_dim scalars; zero-pad if latent is smaller
            embed_dim = self.nets["policy"].embed_dim
            if flat.shape[-1] >= embed_dim:
                flat = flat[..., :embed_dim]
            else:
                pad = torch.zeros(
                    flat.shape[0],
                    flat.shape[1],
                    embed_dim - flat.shape[-1],
                    device=flat.device,
                    dtype=flat.dtype,
                )
                flat = torch.cat([flat, pad], dim=-1)
        return flat

    def _robomimic_to_hpt_data(
        self, batch, cam_keys, proprio_keys, lang_keys, ac_key, aux_ac_keys=[]
    ):
        """
        helper method that returns data in the format required for the HPT model
        """
        data = {}

        for key in proprio_keys:
            if key in batch:
                data[f"state_{key}"] = batch[key].unsqueeze(1)

        for key in cam_keys:
            if key in batch:
                _data = batch[key]
                if self.use_shared_vae_stem and key in self._vae_stem_camera_keys:
                    _data = self._encode_current_visual_state(_data, cam_key=key)
                    data[key] = _data.unsqueeze(2)
                else:
                    if not torch.all(_data == 0):
                        if self.nets.training and key in self.encoders:
                            _data = self.train_image_augs(_data)
                        elif self.eval_image_augs and key in self.encoders:
                            _data = self.eval_image_augs(_data)
                    data[key] = _data.unsqueeze(1).unsqueeze(1)

        for key in lang_keys:
            if key in batch:
                data[key] = batch[key]

        data["is_6dof"] = self.is_6dof
        data["pad_mask"] = batch["pad_mask"]
        data["embodiment"] = batch["embodiment"]

        for aux_ac_key in aux_ac_keys:
            if aux_ac_key in batch:
                aux_val = batch[aux_ac_key]
                if aux_val.ndim == 2:
                    aux_val = aux_val.unsqueeze(1)
                data[aux_ac_key] = aux_val
            # Point-flow heads publish sidecar keys alongside the main trajectory
            # tensor; passthrough so PointFMPolicy.build_aux_cond finds them.
            for sidecar in (
                f"{aux_ac_key}__current_points",
                f"{aux_ac_key}__validity_mask",
            ):
                if sidecar in batch:
                    data[sidecar] = batch[sidecar]

        if self.shared_ac_key:
            data["action"] = batch[self.shared_ac_key]
        else:
            data["action"] = batch[ac_key]
        return data

    def _clone_batch(self, batch):
        """Recursively clones all tensors inside a nested dictionary."""
        if isinstance(batch, dict):
            return {key: self._clone_batch(val) for key, val in batch.items()}
        elif isinstance(batch, torch.Tensor):
            return batch.clone()
        else:
            return batch  # Return as is for non-tensor types

    @staticmethod
    def _extract_xyz(x):
        """
        Extract xyz (3D position) and rotation from 6DoF or 6DoF+gripper actions.

        Supports:
        - 6: 6DoF (single arm)
        - 7: 6DoF + gripper (single arm)
        - 12: 2 arms × 6DoF
        - 14: 2 arms × (6DoF + gripper)

        Returns:
            xyz: Tensor with only xyz per arm (shape: ..., 3) or (..., 6) for dual-arm.
            rot: Tensor with only rotation per arm (shape: ..., 3) or (..., 6) for dual-arm.
        """
        if x.shape[-1] == 6:
            return x[..., :3], x[..., 3:6]
        elif x.shape[-1] == 7:
            return x[..., :3], x[..., 3:6]
        elif x.shape[-1] == 12:
            xyz_right = x[..., :3]
            rot_right = x[..., 3:6]
            xyz_left = x[..., 6:9]
            rot_left = x[..., 9:12]
            return torch.cat([xyz_right, xyz_left], dim=-1), torch.cat(
                [rot_right, rot_left], dim=-1
            )
        elif x.shape[-1] == 14:
            xyz_right = x[..., :3]
            rot_right = x[..., 3:6]
            xyz_left = x[..., 7:10]
            rot_left = x[..., 10:13]
            return torch.cat([xyz_right, xyz_left], dim=-1), torch.cat(
                [rot_right, rot_left], dim=-1
            )
        else:
            raise ValueError(f"Unexpected shape for 6DoF input: {x.shape}")
