# Cosmos Policy Diffusion Model Generation Flow

This document provides detailed flow diagrams and explanations for the diffusion model generation process in Cosmos Policy. The diffusion model is the core component that generates actions, future states, and values from noise, conditioned on observations and task descriptions.

## Table of Contents

1. [Overview](#overview)
2. [Complete Generation Flow](#complete-generation-flow)
3. [Denoising Process Detail](#denoising-process-detail)
4. [Diffusion Sampling Loop (Detailed)](#diffusion-sampling-loop-detailed)
5. [Conditioning Mechanism](#conditioning-mechanism)
6. [Key Components](#key-components)
7. [File Locations](#file-locations)

---

## Overview

The diffusion model generation process (`model.generate_samples_from_batch()`) is the core of action prediction. It uses a stochastic differential equation (SDE) based diffusion process to generate actions, future states, and values from noise, conditioned on the current observation and task description.

**File Location**: `cosmos_policy/models/policy_text2world_model.py` (line 702)

### Key Concepts

- **Diffusion Process**: Iteratively denoises random noise to generate clean latents
- **Conditioning**: Uses current state (images, proprio) and text embeddings to guide generation
- **Mask-Based Generation**: Selectively generates only certain frames (actions, future states) while keeping others fixed
- **SDE Solver**: Uses differential equation solvers (Euler, Runge-Kutta) for efficient sampling

---

## Complete Generation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│        generate_samples_from_batch() Entry Point                │
│  Input:                                                          │
│    - data_batch: Processed observation data                     │
│    - guidance: Classifier-free guidance scale (default: 1.5)    │
│    - seed: Random seed for reproducibility                      │
│    - num_steps: Number of denoising steps (default: 5)           │
│    - use_variance_scale: Whether to scale noise variance       │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│              Step 1: Data Batch Normalization                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ _normalize_video_databatch_inplace(data_batch):         │  │
│  │   - Normalize pixel values to [0, 1]                     │  │
│  │                                                           │  │
│  │ _augment_image_dim_inplace(data_batch):                  │  │
│  │   - Ensure proper tensor dimensions                      │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│              Step 2: Determine State Shape                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ if state_shape is None:                                  │  │
│  │     _T, _H, _W = data_batch["video"].shape[-3:]          │  │
│  │     state_shape = [                                       │  │
│  │         state_ch,                                         │  │
│  │         tokenizer.get_latent_num_frames(_T),             │  │
│  │         _H // spatial_compression_factor,                 │  │
│  │         _W // spatial_compression_factor                  │  │
│  │     ]                                                     │  │
│  │     # Example: [16, 9, 28, 28] for LIBERO                │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│        Step 3: Build x0 Function (Conditioning)                  │
│        get_x0_fn_from_batch()                                    │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
    ┌─────────────────────────────────────────────────────────┐
    │  Sub-step 3.1: Get Condition from Data Batch            │
    │  ┌───────────────────────────────────────────────────┐  │
    │  │ conditioner.get_condition_uncondition():          │  │
    │  │   - Extract T5 text embeddings                    │  │
    │  │   - Encode images via VAE → latents               │  │
    │  │   - Build condition object with:                  │  │
    │  │     * gt_frames: Encoded image latents            │  │
    │  │     * t5_text_embeddings: Text embeddings         │  │
    │  │     * condition_video_input_mask: Mask for cond   │  │
    │  └───────────────────────────────────────────────────┘  │
    └────────────────────┬────────────────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────────────────────┐
    │  Sub-step 3.2: Inject Proprio into Latent                │
    │  ┌───────────────────────────────────────────────────┐  │
    │  │ if proprio is provided:                           │  │
    │  │     condition.gt_frames = replace_latent_with_   │  │
    │  │                        proprio(                   │  │
    │  │         condition.gt_frames,                      │  │
    │  │         proprio,                                  │  │
    │  │         current_proprio_latent_idx                 │  │
    │  │     )                                              │  │
    │  │     # Replace latent at proprio_idx with proprio   │  │
    │  │     # encoded as latent representation            │  │
    │  └───────────────────────────────────────────────────┘  │
    └────────────────────┬────────────────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────────────────────┐
    │  Sub-step 3.3: Create x0_fn Closure                      │
    │  ┌───────────────────────────────────────────────────┐  │
    │  │ def x0_fn(noise_x, sigma):                       │  │
    │  │     # For Cosmos Policy (always-conditional):    │  │
    │  │     cond_x0 = self.denoise(noise_x, sigma,       │  │
    │  │                            condition).x0          │  │
    │  │     return cond_x0                                │  │
    │  │                                                   │  │
    │  │     # Note: No CFG (uncondition=None) for        │  │
    │  │     # Cosmos Policy - always conditional          │  │
    │  └───────────────────────────────────────────────────┘  │
    └────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Step 4: Initialize Noise                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ if x_sigma_max is None:                                 │  │
│  │     x_sigma_max = random_noise(                         │  │
│  │         shape=(n_sample, *state_shape),                 │  │
│  │         seed=seed                                        │  │
│  │     ) * sigma_max * variance_scale                      │  │
│  │                                                           │  │
│  │ # Start with pure noise at maximum sigma level          │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│              Step 5: Variance Scaling (Optional)                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ if use_variance_scale:                                   │  │
│  │     sigma_max_scale = random(1.0, 3.0)                  │  │
│  │     sigma_min_scale = random(0.1, 1.0)                  │  │
│  │     # Increases diversity in generated samples           │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│              Step 6: Run Diffusion Sampler                       │
│              sampler(x0_fn, x_sigma_max, num_steps)              │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
    ┌─────────────────────────────────────────────────────────┐
    │  Sub-step 6.1: Generate Timestamps                       │
    │  ┌───────────────────────────────────────────────────┐  │
    │  │ sigmas_L = get_rev_ts(                            │  │
    │  │     t_min=sigma_min,                              │  │
    │  │     t_max=sigma_max,                              │  │
    │  │     nfe=num_steps,                                │  │
    │  │     order=rho                                      │  │
    │  │ )                                                  │  │
    │  │ # Creates schedule: [sigma_max, ..., sigma_min]   │  │
    │  │ # Example: [80.0, 40.0, 20.0, 10.0, 5.0, 0.002]  │  │
    │  └───────────────────────────────────────────────────┘  │
    └────────────────────┬────────────────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────────────────────┐
    │  Sub-step 6.2: Differential Equation Solver Loop         │
    │  ┌───────────────────────────────────────────────────┐  │
    │  │ for i, (t_cur, t_next) in enumerate(timestamps):  │  │
    │  │     # Current noisy sample: x_cur                  │  │
    │  │                                                      │  │
    │  │     # Optional: Add noise (S_churn)                │  │
    │  │     if S_min <= t_cur <= S_max:                    │  │
    │  │         gamma = min(S_churn / num_steps, ...)       │  │
    │  │         t_hat = t_cur + gamma * t_cur              │  │
    │  │         x_hat = x_cur + noise                      │  │
    │  │                                                      │  │
    │  │     # Denoise step                                  │  │
    │  │     x0_pred = x0_fn(x_hat, t_hat)                  │  │
    │  │     # Calls model.denoise() internally             │  │
    │  │                                                      │  │
    │  │     # Euler step                                    │  │
    │  │     d_cur = (x_hat - x0_pred) / t_hat             │  │
    │  │     x_next = x_hat + (t_next - t_hat) * d_cur      │  │
    │  │                                                      │  │
    │  │     # Optional: 2nd order correction                │  │
    │  │     if i < num_steps - 1:                          │  │
    │  │         x0_pred_next = x0_fn(x_next, t_next)       │  │
    │  │         d_prime = (x_next - x0_pred_next) / t_next │  │
    │  │         x_next = x_hat + (t_next - t_hat) *        │  │
    │  │                    (0.5 * d_cur + 0.5 * d_prime)   │  │
    │  └───────────────────────────────────────────────────┘  │
    └────────────────────┬────────────────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────────────────────┐
    │  Sub-step 6.3: Final Clean Step (if enabled)             │
    │  ┌───────────────────────────────────────────────────┐  │
    │  │ if sample_clean:                                  │  │
    │  │     ones = torch.ones(batch_size)                 │  │
    │  │     denoised_output = x0_fn(                      │  │
    │  │         denoised_output,                           │  │
    │  │         sigmas_L[-1] * ones                       │  │
    │  │     )                                              │  │
    │  │     # One final denoising step at minimum sigma   │  │
    │  └───────────────────────────────────────────────────┘  │
    └────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Step 7: Return Generated Latent                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Output: generated_latent_with_action                     │  │
│  │   Shape: (B, C'=16, T'=9, H'=28, W'=28)                  │  │
│  │                                                           │  │
│  │ Contains:                                                │  │
│  │   - Current state latents (conditioning)                 │  │
│  │   - Generated action latents                             │  │
│  │   - Generated future state latents                       │  │
│  │   - Generated value latents                              │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Denoising Process Detail

```
┌─────────────────────────────────────────────────────────────┐
│              model.denoise() Function                        │
│              (Called inside x0_fn)                           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Input:                                                       │
│    - noise_x: Noisy latent (B, C, T, H, W)                   │
│    - sigma: Noise level (B,)                                 │
│    - condition: Text2WorldCondition object                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Prepare Condition                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ condition.gt_frames: Encoded current state latents    │  │
│  │ condition.t5_text_embeddings: Task description         │  │
│  │ condition.condition_video_input_mask: Which frames     │  │
│  │   are conditioning (1) vs to be generated (0)          │  │
│  └───────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Concatenate Condition with Noise                    │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ # Mask-based concatenation:                           │  │
│  │ # - Conditioning frames: Use gt_frames (known)         │  │
│  │ # - Generation frames: Use noise_x (to be denoised)   │  │
│  │                                                       │  │
│  │ input_latent = condition_video_input_mask *          │  │
│  │                  condition.gt_frames +               │  │
│  │                  (1 - condition_video_input_mask) *   │  │
│  │                  noise_x                              │  │
│  └───────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: Forward Pass Through Network                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ # Transformer-based diffusion model                    │  │
│  │ x0_pred = self.net(                                   │  │
│  │     input_latent,                                     │  │
│  │     sigma,                                            │  │
│  │     condition.t5_text_embeddings                       │  │
│  │ )                                                     │  │
│  │                                                       │  │
│  │ # Network architecture:                              │  │
│  │ # - Spatial-temporal attention                        │  │
│  │ # - Cross-attention with text                        │  │
│  │ # - Predicts clean x0 from noisy input                │  │
│  └───────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 4: Return DenoisePrediction                            │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ return DenoisePrediction(                              │  │
│  │     x0=x0_pred,  # Predicted clean latent              │  │
│  │     eps=epsilon_pred,  # Predicted noise               │  │
│  │     v=None  # Velocity (if applicable)                │  │
│  │ )                                                     │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Diffusion Sampling Loop (Detailed)

```
┌─────────────────────────────────────────────────────────────┐
│         Iterative Denoising Process                          │
│         (num_steps=5 for action prediction)                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
    ┌───────────────────────────────────────────────────────┐
    │  Initialization:                                      │
    │    x_cur = x_sigma_max  (pure noise at sigma=80)      │
    │    timestamps = [80.0, 40.0, 20.0, 10.0, 5.0, 0.002] │
    └────────────────────┬──────────────────────────────────┘
                         │
                         ▼
    ┌───────────────────────────────────────────────────────┐
    │  Iteration 1: t_cur=80.0 → t_next=40.0                │
    │  ┌─────────────────────────────────────────────────┐  │
    │  │ x0_pred = x0_fn(x_cur, t_cur=80.0)              │  │
    │  │   → model.denoise(x_cur, 80.0, condition)      │  │
    │  │   → Returns: predicted clean latent              │  │
    │  │                                                   │  │
    │  │ d_cur = (x_cur - x0_pred) / 80.0                │  │
    │  │ x_next = x_cur + (40.0 - 80.0) * d_cur          │  │
    │  │ # Less noisy sample                              │  │
    │  └─────────────────────────────────────────────────┘  │
    └────────────────────┬──────────────────────────────────┘
                         │
                         ▼
    ┌───────────────────────────────────────────────────────┐
    │  Iteration 2: t_cur=40.0 → t_next=20.0                │
    │  ┌─────────────────────────────────────────────────┐  │
    │  │ x0_pred = x0_fn(x_cur, t_cur=40.0)             │  │
    │  │ d_cur = (x_cur - x0_pred) / 40.0               │  │
    │  │ x_next = x_cur + (20.0 - 40.0) * d_cur         │  │
    │  └─────────────────────────────────────────────────┘  │
    └────────────────────┬──────────────────────────────────┘
                         │
                         ▼
    ┌───────────────────────────────────────────────────────┐
    │  Iteration 3: t_cur=20.0 → t_next=10.0               │
    │  ┌─────────────────────────────────────────────────┐  │
    │  │ x0_pred = x0_fn(x_cur, t_cur=20.0)             │  │
    │  │ d_cur = (x_cur - x0_pred) / 20.0               │  │
    │  │ x_next = x_cur + (10.0 - 20.0) * d_cur         │  │
    │  └─────────────────────────────────────────────────┘  │
    └────────────────────┬──────────────────────────────────┘
                         │
                         ▼
    ┌───────────────────────────────────────────────────────┐
    │  Iteration 4: t_cur=10.0 → t_next=5.0                 │
    │  ┌─────────────────────────────────────────────────┐  │
    │  │ x0_pred = x0_fn(x_cur, t_cur=10.0)             │  │
    │  │ d_cur = (x_cur - x0_pred) / 10.0               │  │
    │  │ x_next = x_cur + (5.0 - 10.0) * d_cur          │  │
    │  └─────────────────────────────────────────────────┘  │
    └────────────────────┬──────────────────────────────────┘
                         │
                         ▼
    ┌───────────────────────────────────────────────────────┐
    │  Iteration 5: t_cur=5.0 → t_next=0.002                │
    │  ┌─────────────────────────────────────────────────┐  │
    │  │ x0_pred = x0_fn(x_cur, t_cur=5.0)               │  │
    │  │ d_cur = (x_cur - x0_pred) / 5.0                 │  │
    │  │ x_next = x_cur + (0.002 - 5.0) * d_cur          │  │
    │  └─────────────────────────────────────────────────┘  │
    └────────────────────┬──────────────────────────────────┘
                         │
                         ▼
    ┌───────────────────────────────────────────────────────┐
    │  Final Clean Step (if sample_clean=True):              │
    │  ┌─────────────────────────────────────────────────┐  │
    │  │ x_final = x0_fn(x_next, t=0.002)                │  │
    │  │ # One more denoising at minimum noise level     │  │
    │  └─────────────────────────────────────────────────┘  │
    └────────────────────┬──────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Clean Latent Output │
              │  (B, C, T, H, W)     │
              └──────────────────────┘
```

---

## Conditioning Mechanism

```
┌─────────────────────────────────────────────────────────────┐
│         How Conditioning Works in Cosmos Policy             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Condition Object Structure:                                 │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ condition.gt_frames: (B, C, T, H, W)                   │  │
│  │   - Contains encoded latents for:                      │  │
│  │     * Current proprio (injected)                       │  │
│  │     * Current wrist image (encoded)                    │  │
│  │     * Current primary image (encoded)                  │  │
│  │     * Action (noise, to be generated)                  │  │
│  │     * Future state (noise, to be generated)           │  │
│  │     * Value (noise, to be generated)                  │  │
│  │                                                         │  │
│  │ condition.condition_video_input_mask: (B, 1, T, H, W) │  │
│  │   - 1 = conditioning frame (use gt_frames)             │  │
│  │   - 0 = generation frame (use noise_x)                │  │
│  │                                                         │  │
│  │ condition.t5_text_embeddings: (B, seq_len, embed_dim)  │  │
│  │   - Task description embedding                         │  │
│  └───────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Mask-Based Concatenation in denoise():                      │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ input_latent =                                       │  │
│  │     mask * condition.gt_frames +                     │  │
│  │     (1 - mask) * noise_x                             │  │
│  │                                                       │  │
│  │ # Result:                                            │  │
│  │ # - Conditioning frames: Real encoded latents         │  │
│  │ # - Generation frames: Noisy latents to denoise      │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Components

### 1. x0_fn Function (Conditioning Closure)

**Location**: Created in `get_x0_fn_from_batch()` (line 569 in `policy_video2world_model.py`)

```python
def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    # For Cosmos Policy: always-conditional (no CFG)
    cond_x0 = self.denoise(noise_x, sigma, condition).x0
    return cond_x0
```

This closure captures the conditioning information and provides a function that the sampler can call repeatedly during the denoising process.

### 2. Sampler (Differential Equation Solver)

**Location**: `cosmos_policy/modules/cosmos_sampler.py` (class `CosmosPolicySampler`)

- Uses EDM (Elucidating the Design Space) sampling
- Supports multi-step and Runge-Kutta solvers
- Implements optional noise injection (S_churn) for diversity
- Performs 2nd order correction for accuracy

The sampler implements the iterative denoising process, calling `x0_fn` at each step to predict the clean latent.

### 3. Denoising Network

**Location**: Transformer-based diffusion model in the network

- Takes noisy latent + sigma + text embedding
- Uses spatial-temporal attention
- Cross-attends to text embeddings
- Predicts clean x0 from noisy input

The network is the core learned component that performs the actual denoising operation.

---

## File Locations

- **Main generation method**: `cosmos_policy/models/policy_text2world_model.py:702` (`generate_samples_from_batch`)
- **x0 function builder**: `cosmos_policy/models/policy_video2world_model.py:569` (`get_x0_fn_from_batch`)
- **Sampler**: `cosmos_policy/modules/cosmos_sampler.py:40` (`CosmosPolicySampler`)
- **Base denoising**: Inherited from `cosmos_policy/_src/predict2/models/text2world_model.py`

---

## Related Documentation

For more information on how the diffusion model is used in the overall system:

- [LIBERO Evaluation Flow](EVAL.md#libero-evaluation-script-flow)
- [get_action() Function Flow](EVAL.md#get_action-function-flow)

