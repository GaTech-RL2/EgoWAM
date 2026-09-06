from typing import Dict, Tuple

import torch
from overrides import override

from egowam.models.denoising_nets import ConditionalUnet1D
from egowam.models.denoising_policy import DenoisingPolicy


class FMPolicy(DenoisingPolicy):
    """
    A diffusion-based policy head.

    Args:
        model (ConditionalUnet1D): The model used for prediction.
        noise_scheduler: The noise scheduler used for the diffusion process.
        action_horizon (int): The number of time steps in the action horizon.
        output_dim (int): The dimension of the output.
        num_inference_steps (int, optional): The number of inference steps.
        **kwargs: Additional keyword arguments, including:
            time_dist (str): "beta" or "uniform" sampling of flow time.
            prediction_type (str): "v_prediction" (default) — network predicts flow velocity;
                "x_prediction" — network predicts clean data; training still uses v-loss via
                v_theta = (z_t - x_theta) / t (Li & He, "Back to Basics: Let Denoising Generative Models Denoise").
    """

    def __init__(
        self,
        model: ConditionalUnet1D,
        action_horizon,
        infer_ac_dims,
        num_inference_steps=None,
        **kwargs,
    ):
        super().__init__(
            model, action_horizon, infer_ac_dims, num_inference_steps, **kwargs
        )
        self.time_dist = kwargs.get("time_dist", "beta")
        self.prediction_type = kwargs.get("prediction_type", "v_prediction")
        if self.prediction_type not in ("v_prediction", "x_prediction"):
            raise ValueError(
                f"prediction_type must be 'v_prediction' or 'x_prediction' (legacy alias 'velocity' -> 'v_prediction'), got {self.prediction_type!r}"
            )

    @staticmethod
    def _expand_time_for_x(t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        """Broadcast scalar time [B] to x_t's shape for division."""
        te = t
        while te.ndim < x_t.ndim:
            te = te.unsqueeze(-1)
        return te

    def step(self, x_t, t, global_cond):
        if len(t.shape) != 1:
            t = torch.tensor([t], device=global_cond.device)
        if self.prediction_type == "x_prediction":
            x_theta = self.model(x_t, t, global_cond)
            te = self._expand_time_for_x(t, x_t)
            v_t = (x_t - x_theta) / te.clamp(min=1e-3)
        else:
            v_t = self.model(x_t, t, global_cond)
        return x_t + self.dt * v_t, t + self.dt

    @override
    def inference(self, noise, global_cond, generator=None) -> torch.Tensor:
        self.dt = -1.0 / self.num_inference_steps
        x_t = noise
        time = torch.ones((len(global_cond)), device=global_cond.device)
        while time[0] >= -self.dt / 2:
            x_t, time = self.step(x_t, time, global_cond)
        return x_t

    @override
    def predict(self, actions, global_cond) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn(actions.shape, device=actions.device)
        batch_shape = (actions.shape[0],)

        if self.time_dist == "beta":
            a, b = 1.5, 1.0
            time = torch.distributions.Beta(a, b).sample(batch_shape).to(actions.device)
        elif self.time_dist == "uniform":
            time = (
                torch.distributions.Uniform(0, 1).sample(batch_shape).to(actions.device)
            )
        time = time * 0.999 + 0.001

        time_expanded = time.unsqueeze(-1).unsqueeze(-1)
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        if self.prediction_type == "x_prediction":
            x_theta = self.model(x_t, time, global_cond)
            pred = (x_t - x_theta) / time_expanded.clamp(min=1e-3)
        else:
            pred = self.model(x_t, time, global_cond)

        target = u_t
        return pred, target


class PointFMPolicy(FMPolicy):
    """Point-conditioned flow-matching policy for the 2D and 3D point flow heads.

    Conditioning is a dict {trunk_feats, current_points, validity_mask} rather
    than a tensor. Targets are 4D [B, H, N, D] (delta trajectories, normalized
    by loader-side NormalizeFlowDelta). Overrides every base-class path that
    does tensor operations on `global_cond` or assumes a scalar `infer_ac_dims`.

    Capability API:
      - build_aux_cond(domain, features, batch) -> dict
      - sample_aux(cond) -> Tensor
    HPT-WM uses these from the unified aux dispatch so legacy FMPolicy heads
    still work unchanged (they use the DenoisingPolicy default).

    Loss contract (plan round-8 + round-6): NaN-safe target sanitize BEFORE
    self.predict(); broadcast mask over point axis only; normalize by
    valid_points * H * D for scale stability across point_dim.
    """

    def _batch_size(self, global_cond: Dict[str, torch.Tensor]) -> int:
        return int(global_cond["trunk_feats"].shape[0])

    def _noise_shape(self, embodiment_name: str) -> Tuple[int, ...]:
        """infer_ac_dims for a point head is a 3-tuple [H, N, D]."""
        dims = self.infer_ac_dims[embodiment_name]
        if isinstance(dims, int):
            raise ValueError(
                f"PointFMPolicy requires infer_ac_dims[{embodiment_name}] to be "
                f"[H, N, D]; got scalar {dims}"
            )
        return tuple(int(d) for d in dims)

    @override
    def preprocess_sampling(self, global_cond, embodiment_name, generator=None):
        if not isinstance(global_cond, dict):
            raise TypeError(
                "PointFMPolicy.preprocess_sampling expects dict global_cond, "
                f"got {type(global_cond)}"
            )
        trunk_feats = global_cond["trunk_feats"]
        current_points = global_cond["current_points"]
        # Read N (point count) from the actual batch tensor so the head stays
        # N-agnostic at inference. H and D come from the head's configured
        # shape; the config's N entry is ignored here on purpose.
        H, _, D = self._noise_shape(embodiment_name)
        N = int(current_points.shape[1])
        shape = (self._batch_size(global_cond), int(H), N, int(D))
        noise = torch.randn(
            shape,
            dtype=trunk_feats.dtype,
            device=trunk_feats.device,
            generator=generator,
        )
        return noise, global_cond

    @override
    def sample_action(self, global_cond, embodiment_name, generator=None):
        noise, cond = self.preprocess_sampling(global_cond, embodiment_name, generator)
        return self.inference(noise, cond, generator)

    @override
    def forward(self, global_cond):
        if not isinstance(global_cond, dict):
            raise TypeError(
                "PointFMPolicy.forward expects dict global_cond; "
                "update the aux dispatch to use head.sample_aux(head.build_aux_cond(...))"
            )
        embodiment_name = global_cond["embodiment"]
        return self.sample_action(global_cond, embodiment_name)

    @override
    def step(self, x_t, t, global_cond):
        if t.ndim == 0 or t.shape == ():
            t = torch.tensor([t.item()], device=global_cond["trunk_feats"].device)
        if self.prediction_type == "x_prediction":
            x_theta = self.model(x_t, t, global_cond)
            te = self._expand_time_for_x(t, x_t)
            v_t = (x_t - x_theta) / te.clamp(min=1e-3)
        else:
            v_t = self.model(x_t, t, global_cond)
        return x_t + self.dt * v_t, t + self.dt

    @override
    def inference(self, noise, global_cond, generator=None) -> torch.Tensor:
        self.dt = -1.0 / self.num_inference_steps
        x_t = noise
        B = self._batch_size(global_cond)
        time = torch.ones(
            (B,),
            device=global_cond["trunk_feats"].device,
            dtype=global_cond["trunk_feats"].dtype,
        )
        while time[0] >= -self.dt / 2:
            x_t, time = self.step(x_t, time, global_cond)
        return x_t

    @override
    def predict(self, actions, global_cond) -> Tuple[torch.Tensor, torch.Tensor]:
        """actions: target delta trajectory [B, H, N, D] (already normalized).
        Returns (pred, target) in flow-matching v_prediction convention."""
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        batch_shape = (actions.shape[0],)

        if self.time_dist == "beta":
            a, b = 1.5, 1.0
            time = torch.distributions.Beta(a, b).sample(batch_shape).to(actions.device)
        elif self.time_dist == "uniform":
            time = (
                torch.distributions.Uniform(0, 1).sample(batch_shape).to(actions.device)
            )
        time = time * 0.999 + 0.001

        # Broadcast time to [B, 1, 1, 1] so it multiplies the 4D target.
        time_expanded = time.view(-1, 1, 1, 1)
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        if self.prediction_type == "x_prediction":
            x_theta = self.model(x_t, time, global_cond)
            pred = (x_t - x_theta) / time_expanded.clamp(min=1e-3)
        else:
            pred = self.model(x_t, time, global_cond)

        return pred, u_t

    # --- Loss & capability API --------------------------------------------------

    # Deliberately NOT @override — signature differs (features, aux_data) to make
    # the dict contract explicit. The HPT-WM aux dispatch calls this directly.
    def compute_loss(self, features, aux_data):
        """aux_data = {current_points, target_delta_normed, validity_mask, embodiment}.

        features: trunk features [B, L_trunk, cond_dim]. Caller (HPT-WM aux loop)
        is expected to construct aux_data via build_aux_cond; this method does
        NOT call preprocess_compute_loss.
        """
        mask = (
            aux_data["validity_mask"]
            .to(dtype=features.dtype)
            .unsqueeze(1)
            .unsqueeze(-1)
        )
        # mask shape: [B, 1, N, 1] — broadcasts over H and D correctly.
        mask_bool = mask.bool()

        target = aux_data["target_delta_normed"]
        # NaN-safe: sanitize target BEFORE predict(). Round-6 finding.
        target = torch.where(mask_bool, target, torch.zeros_like(target))

        global_cond = {
            "trunk_feats": features,
            "current_points": aux_data["current_points"],
            "embodiment": aux_data.get("embodiment"),
        }

        pred, target_reconstructed = self.predict(target, global_cond)
        err = (pred - target_reconstructed).square()
        err = torch.where(mask_bool, err, torch.zeros_like(err))

        valid_points = (
            aux_data["validity_mask"].sum(dim=-1).clamp(min=1).to(dtype=features.dtype)
        )
        H = pred.shape[1]
        D = pred.shape[-1]
        per_sample_norm = valid_points * H * D
        return ((err * mask).sum(dim=(1, 2, 3)) / per_sample_norm).mean()

    # Capability hooks for the HPT-WM unified aux dispatch. Legacy FMPolicy
    # heads fall back to DenoisingPolicy.build_aux_cond below.
    def build_aux_cond(self, domain, features, batch):
        """Assemble the dict conditioning expected by compute_loss / sample_action.

        batch keys follow the loader sidecar convention from
        _load_point_flow_target:
          - "{flow_key}"                       [B, H, N, D]  target_delta_normed
          - "{flow_key}__current_points"       [B, N, D]
          - "{flow_key}__validity_mask"        [B, N]  (bool or float)
        """
        flow_key = batch["__flow_key__"] if "__flow_key__" in batch else None
        if flow_key is None:
            raise KeyError(
                "build_aux_cond expected batch['__flow_key__'] set by HPT-WM aux dispatch"
            )
        return {
            "trunk_feats": features,
            "current_points": batch[f"{flow_key}__current_points"],
            "validity_mask": batch[f"{flow_key}__validity_mask"].to(dtype=torch.bool),
            "target_delta_normed": batch[flow_key],
            "embodiment": domain,
        }

    def sample_aux(self, cond):
        return self.sample_action(cond, cond["embodiment"])


# Attach default (pass-through) capability hooks on DenoisingPolicy so the
# unified aux dispatch in HPT-WM works for both point heads and legacy tensor-
# cond heads without special-casing per type.
def _default_build_aux_cond(self, domain, features, batch):
    # Legacy contract: global_cond = (features, domain).
    if getattr(self, "_uses_tuple_cond", True):
        return (features, domain)
    return features


def _default_sample_aux(self, cond):
    # For tuple cond, DenoisingPolicy.forward already unpacks (cond, embodiment).
    return self(cond)


# Monkey-patch ONLY if not already defined (PointFMPolicy overrides above win).
if not hasattr(DenoisingPolicy, "build_aux_cond"):
    DenoisingPolicy.build_aux_cond = _default_build_aux_cond  # type: ignore[attr-defined]
if not hasattr(DenoisingPolicy, "sample_aux"):
    DenoisingPolicy.sample_aux = _default_sample_aux  # type: ignore[attr-defined]
