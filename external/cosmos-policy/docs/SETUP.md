# Setup Guide

This guide walks you through setting up Cosmos Policy using Docker.

## System Requirements

* NVIDIA GPU with CUDA support
* Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
* Linux x86-64

## Installation

### Build the Docker Image

From the project root, build the image:

```bash
docker build -t cosmos-policy docker
```

### Launch the Docker Container

Start an interactive session:

```bash
docker run \
  -u root \
  -e HOST_USER_ID=$(id -u) \
  -e HOST_GROUP_ID=$(id -g) \
  -v $HOME/.cache:/home/cosmos/.cache \
  -v $(pwd):/workspace \
  --gpus all \
  --ipc=host \
  -it \
  --rm \
  -w /workspace \
  --entrypoint bash \
  cosmos-policy
```

**Optional arguments:**
* `-v $HOME/.cache:/home/cosmos/.cache`: Reuses host cache for `uv`, `huggingface`, etc.
* `--ipc=host`: Shares host's inter-process communication namespace. Required for PyTorch's parallel data loading to avoid out-of-memory errors (containers get only 64MB shared memory by default). If your security policy doesn't allow this, try using `--shm-size 32g` instead to allocate sufficient isolated shared memory.

## Running an Example

Once you are inside the container, you can run (1) the Quick Start steps in the [README](../README.md) and (2) the LIBERO or RoboCasa evaluation commands (see [LIBERO.md](LIBERO.md) and [ROBOCASA.md](ROBOCASA.md)).
