# Cosmos Policy Training Guide

This document provides a comprehensive guide for training Cosmos Policy models using the training script.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Training Flow](#training-flow)
4. [Configuration](#configuration)
5. [Distributed Training](#distributed-training)
6. [Key Components](#key-components)
7. [Command-Line Options](#command-line-options)
8. [Examples](#examples)
9. [Troubleshooting](#troubleshooting)

---

## Overview

The training script (`cosmos_policy/scripts/train.py`) is the main entry point for training Cosmos Policy models. It supports:

- **Distributed Training**: Multi-GPU and multi-node training using PyTorch DDP and Megatron parallelism
- **Manual DataLoader Setup**: Custom DistributedSampler instantiation to avoid duplicate dataset creation
- **Config-Based Training**: Flexible configuration system using LazyConfig
- **Dry-Run Mode**: Validate configurations without starting training
- **Validation Support**: Optional validation during training

**File Location**: `cosmos_policy/scripts/train.py`

---

## Quick Start

### Basic Training Command

```bash
torchrun --nproc_per_node=1 \
  -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/experiment/your_config.py
```

### Multi-GPU Training

```bash
torchrun --nproc_per_node=4 \
  -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/experiment/your_config.py
```

### Dry-Run (Validate Config)

```bash
python -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/experiment/your_config.py \
  --dryrun
```

### Override Config Values

```bash
torchrun --nproc_per_node=1 \
  -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/experiment/your_config.py \
  trainer.max_iter=10000 \
  dataloader_train.batch_size=8
```

---

## Training Flow

### Complete Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    Script Entry Point                            │
│  Input:                                                          │
│    - --config: Path to config file                             │
│    - opts: Config overrides (key=value pairs)                  │
│    - --dryrun: Validate config without training                 │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│              Step 1: Parse Arguments & Load Config               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ parser.parse_args()                                      │  │
│  │ load_config(args.config, args.opts)                     │  │
│  │   - Load config from Python file                         │  │
│  │   - Apply command-line overrides                         │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  Dry-Run Mode?         │
         │  ┌──────────┐          │
         │  │   YES    │          │
         │  └────┬─────┘          │
         │       │                 │
         │       ▼                 │
         │  ┌──────────────────┐  │
         │  │ Print config     │  │
         │  │ Save to YAML     │  │
         │  │ Exit             │  │
         │  └──────────────────┘  │
         │                        │
         │  ┌──────────┐          │
         │  │    NO    │          │
         │  └────┬─────┘          │
         │       │                 │
         └───────┼─────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│              Step 2: Initialize Distributed Environment          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ with distributed_init():                                 │  │
│  │     distributed.init()                                   │  │
│  │   - Initialize PyTorch DDP                               │  │
│  │   - Setup Megatron parallelism                            │  │
│  │   - Must be done before config.validate()                │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│              Step 3: Validate & Freeze Config                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ config.validate()                                        │  │
│  │   - Check required fields                                │  │
│  │   - Synchronize across ranks (distributed)               │  │
│  │                                                           │  │
│  │ config.freeze()                                           │  │
│  │   - Prevent modifications during training                │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│              Step 4: Initialize Trainer                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ trainer = config.trainer.type(config)                    │  │
│  │   - Instantiate CosmosPolicyTrainer                      │  │
│  │   - Setup callbacks, checkpointer, etc.                  │  │
│  │                                                           │  │
│  │ log_reproducible_setup(config, args)                      │  │
│  │   - Set random seeds                                     │  │
│  │   - Log environment info                                 │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│              Step 5: Instantiate Model                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ with model_init():                                       │  │
│  │     model = instantiate(config.model)                    │  │
│  │   - Create CosmosPolicyDiffusionModel                    │  │
│  │   - Initialize VAE tokenizer, text encoder, etc.         │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│              Step 6: Setup DataLoaders                            │
│              (Manual DistributedSampler Setup)                    │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
    ┌─────────────────────────────────────────────────────────┐
    │  Training DataLoader Setup:                              │
    │  ┌───────────────────────────────────────────────────┐  │
    │  │ dataset = instantiate(config.dataloader_train.   │  │
    │  │                      dataset)                    │  │
    │  │                                                   │  │
    │  │ sampler = DistributedSampler(                    │  │
    │  │     dataset=dataset,                             │  │
    │  │     num_replicas=parallel_state.                 │  │
    │  │                  get_data_parallel_world_size(), │  │
    │  │     rank=parallel_state.                         │  │
    │  │          get_data_parallel_rank(),              │  │
    │  │     shuffle=True,                               │  │
    │  │     seed=0                                       │  │
    │  │ )                                                │  │
    │  │                                                   │  │
    │  │ dataloader_train = DataLoader(                   │  │
    │  │     dataset=dataset,                             │  │
    │  │     sampler=sampler,                             │  │
    │  │     batch_size=config.dataloader_train.         │  │
    │  │                  batch_size,                     │  │
    │  │     ...                                          │  │
    │  │ )                                                │  │
    │  └───────────────────────────────────────────────────┘  │
    └────────────────────┬────────────────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────────────────────┐
    │  Validation DataLoader Setup (if enabled):              │
    │  ┌───────────────────────────────────────────────────┐  │
    │  │ if config.trainer.run_validation:                 │  │
    │  │     dataset_val = instantiate(...)                │  │
    │  │     sampler_val = DistributedSampler(            │  │
    │  │         shuffle=False  # No shuffle for val       │  │
    │  │     )                                             │  │
    │  │     dataloader_val = DataLoader(...)             │  │
    │  └───────────────────────────────────────────────────┘  │
    └────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Step 7: Start Training                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ trainer.train(                                          │  │
│  │     model,                                              │  │
│  │     dataloader_train,                                   │  │
│  │     dataloader_val                                      │  │
│  │ )                                                       │  │
│  │                                                          │  │
│  │ Training loop (inside trainer.train()):                │  │
│  │   - Epoch loop                                          │  │
│  │   - Batch loop                                          │  │
│  │   - Forward pass                                        │  │
│  │   - Backward pass                                       │  │
│  │   - Optimizer step                                      │  │
│  │   - Validation (periodic)                               │  │
│  │   - Checkpointing (periodic)                            │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Training Loop Detail (Inside trainer.train())

```
┌─────────────────────────────────────────────────────────────┐
│         Training Loop (CosmosPolicyTrainer.train())          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Initialize:                                                 │
│  - Move model to CUDA                                       │
│  - Initialize optimizer, scheduler, grad_scaler            │
│  - Load checkpoint (if resuming)                            │
│  - Wrap model with DDP (if distributed)                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
    ┌───────────────────────────────────────────────────────┐
    │  Epoch Loop:                                           │
    │    epoch = 0                                           │
    │    while True:                                         │
    │        dataloader_train.sampler.set_epoch(epoch)       │
    │        dataloader_train_iter = iter(dataloader_train) │
    └────────────────────┬──────────────────────────────────┘
                         │
                         ▼
        ┌───────────────────────────────────────────────────┐
        │  Batch Loop:                                      │
        │    while True:                                    │
        │        data_batch = next(dataloader_train_iter)  │
        │                                                   │
        │        # Move to GPU                              │
        │        data_batch = misc.to(data_batch, "cuda")  │
        │                                                   │
        │        # Training step                            │
        │        output_batch, loss = model.training_step(  │
        │            data_batch,                            │
        │            iteration                               │
        │        )                                          │
        │                                                   │
        │        # Backward pass                            │
        │        loss.backward()                            │
        │                                                   │
        │        # Optimizer step (with grad accumulation) │
        │        if grad_accum_iter == grad_accum_steps:   │
        │            optimizer.step()                        │
        │            scheduler.step()                        │
        │            optimizer.zero_grad()                  │
        │                                                   │
        │        # Validation (periodic)                    │
        │        if iteration % val_freq == 0:             │
        │            run_validation()                       │
        │                                                   │
        │        # Checkpointing (periodic)                 │
        │        if iteration % checkpoint_freq == 0:       │
        │            checkpointer.save()                    │
        │                                                   │
        │        iteration += 1                             │
        │        if iteration >= max_iter:                  │
        │            break                                   │
        └────────────────────┬──────────────────────────────┘
                             │
                             ▼
                    epoch += 1
```

---

## Configuration

### Config File Structure

Training is configured through Python-based LazyConfig files located in `cosmos_policy/config/experiment/`. A typical config includes:

```python
# Example config structure
from cosmos_policy.config.config import get_config

config = get_config()
config.model = ...  # Model configuration
config.trainer = ...  # Trainer configuration
config.dataloader_train = ...  # Training dataloader config
config.dataloader_val = ...  # Validation dataloader config
config.optimizer = ...  # Optimizer configuration
config.scheduler = ...  # Learning rate scheduler config
config.job = ...  # Job paths and naming
```

### Key Config Sections

#### Model Configuration
- Model architecture (diffusion model, VAE, text encoder)
- Model-specific hyperparameters
- Tokenizer settings

#### Trainer Configuration
- `max_iter`: Maximum training iterations
- `run_validation`: Whether to run validation
- `distributed_parallelism`: DDP or other parallelism modes
- `grad_scaler_args`: Mixed precision training settings
- `memory_format`: Memory layout (channels_last, etc.)

#### DataLoader Configuration
- `dataset`: Dataset class and arguments
- `batch_size`: Batch size per GPU
- `num_workers`: Number of data loading workers
- `pin_memory`: Pin memory for faster GPU transfer
- `drop_last`: Drop last incomplete batch

#### Optimizer & Scheduler
- Optimizer type (Adam, AdamW, etc.)
- Learning rate
- Weight decay
- Scheduler type and parameters

---

## Distributed Training

### Why Manual DistributedSampler?

The training script manually creates `DistributedSampler` instead of using `instantiate(config.dataloader_train)` to avoid creating duplicate datasets. This is important for:

1. **Memory Efficiency**: Prevents creating the dataset twice (once in instantiate, once in DataLoader)
2. **Correct Parallelism**: Ensures each process gets the right data shard
3. **Epoch Synchronization**: Properly sets epoch for shuffling across processes

### Distributed Training Setup

```
┌─────────────────────────────────────────────────────────────┐
│         Distributed Training Architecture                     │
│                                                              │
│  Process 0 (Rank 0)        Process 1 (Rank 1)              │
│  ┌──────────────┐          ┌──────────────┐                │
│  │   Dataset    │          │   Dataset    │                │
│  │  (Full)      │          │  (Full)      │                │
│  └──────┬───────┘          └──────┬───────┘                │
│         │                         │                         │
│         ▼                         ▼                         │
│  ┌──────────────┐          ┌──────────────┐                │
│  │Distributed   │          │Distributed   │                │
│  │Sampler       │          │Sampler       │                │
│  │(rank=0)      │          │(rank=1)      │                │
│  └──────┬───────┘          └──────┬───────┘                │
│         │                         │                         │
│         ▼                         ▼                         │
│  ┌──────────────┐          ┌──────────────┐                │
│  │DataLoader    │          │DataLoader    │                │
│  │(shard 0)     │          │(shard 1)     │                │
│  └──────┬───────┘          └──────┬───────┘                │
│         │                         │                         │
│         ▼                         ▼                         │
│  ┌──────────────┐          ┌──────────────┐                │
│  │   Model      │          │   Model      │                │
│  │  (DDP)       │          │  (DDP)       │                │
│  └──────────────┘          └──────────────┘                │
│         │                         │                         │
│         └───────────┬─────────────┘                         │
│                     │                                       │
│                     ▼                                       │
│            AllReduce (Gradients)                            │
└─────────────────────────────────────────────────────────────┘
```

### Multi-Node Training

For multi-node training, use:

```bash
torchrun \
  --nnodes=2 \
  --node_rank=0 \
  --nproc_per_node=8 \
  --master_addr=<master_node_ip> \
  --master_port=29500 \
  -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/experiment/your_config.py
```

---

## Key Components

### 1. Distributed Initialization

**Location**: `cosmos_policy/_src/imaginaire/utils/context_managers.py`

```python
with distributed_init():
    distributed.init()
    # Initialize PyTorch DDP
    # Setup Megatron parallelism groups
```

**Why First?**: `config.validate()` synchronizes buffers across ranks, so distributed must be initialized first.

### 2. Config Validation

**Location**: `cosmos_policy/_src/imaginaire/config.py`

- Validates required fields
- Broadcasts job name across ranks for consistency
- Ensures checkpoint paths are aligned

### 3. Model Instantiation

**Location**: `cosmos_policy/models/policy_text2world_model.py`

- Creates `CosmosPolicyDiffusionModel`
- Initializes VAE tokenizer
- Sets up text encoder (T5)
- Configures SDE and sampler

### 4. Manual DataLoader Setup

**Why Manual?**: Avoids duplicate dataset creation when using `DistributedSampler`.

**Key Points**:
- Dataset instantiated once per process
- `DistributedSampler` splits data across processes
- Each process gets a unique shard
- Epoch is set via `sampler.set_epoch(epoch)` for proper shuffling

### 5. Trainer

**Location**: `cosmos_policy/trainer.py` (`CosmosPolicyTrainer`)

**Responsibilities**:
- Training loop management
- Optimizer and scheduler initialization
- Checkpoint loading/saving
- Validation scheduling
- Callback execution
- Gradient accumulation
- Mixed precision training

---

## Command-Line Options

### Required Arguments

- `--config`: Path to the config file (Python-based LazyConfig)

### Optional Arguments

- `opts`: Config overrides in `path.key=value` format
  - Example: `trainer.max_iter=10000 dataloader_train.batch_size=8`
- `--dryrun`: Validate config without starting training
  - Prints config
  - Saves config to YAML
  - Exits without training

### Examples

```bash
# Basic training
torchrun --nproc_per_node=1 -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/experiment/my_config.py

# Override config values
torchrun --nproc_per_node=1 -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/experiment/my_config.py \
  trainer.max_iter=50000 \
  dataloader_train.batch_size=16

# Dry-run to validate config
python -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/experiment/my_config.py \
  --dryrun

# Multi-GPU training
torchrun --nproc_per_node=4 -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/experiment/my_config.py
```

---

## Examples

### Example 1: Single GPU Training

```bash
torchrun --nproc_per_node=1 \
  -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/experiment/cosmos_predict2_2b_480p_libero.py
```

### Example 2: Multi-GPU Training with Overrides

```bash
torchrun --nproc_per_node=4 \
  -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/experiment/cosmos_predict2_2b_480p_libero.py \
  trainer.max_iter=100000 \
  dataloader_train.batch_size=4 \
  optimizer.lr=1e-4
```

### Example 3: Resume Training from Checkpoint

```bash
torchrun --nproc_per_node=1 \
  -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/experiment/cosmos_predict2_2b_480p_libero.py \
  checkpointer.resume_from_checkpoint=/path/to/checkpoint.pt
```

### Example 4: Validate Config Before Training

```bash
# First, validate the config
python -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/experiment/my_config.py \
  --dryrun

# If config is valid, start training
torchrun --nproc_per_node=1 \
  -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/experiment/my_config.py
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Reduce `batch_size` in config
- Reduce `num_workers` for dataloader
- Enable gradient checkpointing
- Use mixed precision training (already enabled by default)

#### 2. Distributed Initialization Errors

**Symptoms**: `RuntimeError: NCCL error` or `Address already in use`

**Solutions**:
- Ensure all processes can communicate (check firewall)
- Use different `--master_port` if port is in use
- Verify `--nproc_per_node` matches available GPUs

#### 3. Config Validation Errors

**Symptoms**: `AssertionError` during `config.validate()`

**Solutions**:
- Check that all required config fields are set
- Use `--dryrun` to see full config and identify missing fields
- Ensure distributed is initialized before validation

#### 4. DataLoader Hanging

**Symptoms**: Training hangs during data loading

**Solutions**:
- Reduce `num_workers` (try 0 for debugging)
- Check dataset paths are accessible from all processes
- Verify `timeout` is set appropriately

#### 5. Checkpoint Loading Errors

**Symptoms**: `FileNotFoundError` or shape mismatches when loading checkpoint

**Solutions**:
- Verify checkpoint path is correct
- Check checkpoint was saved with compatible model config
- Use `checkpointer.resume_from_checkpoint` config option

### Debugging Tips

1. **Use Dry-Run**: Always validate config with `--dryrun` before training
2. **Single GPU First**: Test with `--nproc_per_node=1` before multi-GPU
3. **Check Logs**: Training logs are saved to `config.job.path_local`
4. **Monitor GPU**: Use `nvidia-smi` to monitor GPU memory and utilization
5. **Reduce Batch Size**: Start with small batch size to verify setup

---

## File Locations

- **Training script**: `cosmos_policy/scripts/train.py`
- **Trainer**: `cosmos_policy/trainer.py` (`CosmosPolicyTrainer`)
- **Model**: `cosmos_policy/models/policy_text2world_model.py` (`CosmosPolicyDiffusionModel`)
- **Config files**: `cosmos_policy/config/experiment/`
- **Base trainer**: `cosmos_policy/_src/imaginaire/trainer.py` (`ImaginaireTrainer`)

---

## Related Documentation

- [EVAL.md](EVAL.md) - Evaluation and inference flow
- [DIFFUSION_MODEL_GENERATION.md](DIFFUSION_MODEL_GENERATION.md) - Diffusion model generation details
- [SETUP.md](SETUP.md) - Environment setup instructions

---

## Notes

- The training script uses **manual DistributedSampler** to avoid duplicate dataset creation
- Distributed initialization **must** happen before config validation
- Config is **frozen** after validation to prevent accidental changes
- Training supports **gradient accumulation** for effective larger batch sizes
- **Mixed precision training** is enabled by default via GradScaler
- **Checkpointing** happens periodically based on config settings
- **Validation** can be enabled and runs at specified intervals

