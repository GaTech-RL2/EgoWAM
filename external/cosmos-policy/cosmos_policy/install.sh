#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Installation script for Cosmos Policy
#
# Usage:
#   ./install.sh [cu128|cu130]
#
# Arguments:
#   cu128 - Install with CUDA 12.8 support (default)
#   cu130 - Install with CUDA 13.0 support

set -e  # Exit on error

CUDA_VERSION="${1:-cu128}"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Cosmos Policy Installation Script${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: pyproject.toml not found!${NC}"
    echo "Please run this script from the cosmos_policy directory."
    exit 1
fi

# Validate CUDA version argument
if [ "$CUDA_VERSION" != "cu128" ] && [ "$CUDA_VERSION" != "cu130" ]; then
    echo -e "${RED}Error: Invalid CUDA version '$CUDA_VERSION'${NC}"
    echo "Usage: ./install.sh [cu128|cu130]"
    exit 1
fi

echo -e "${YELLOW}Installing with $CUDA_VERSION support...${NC}"
echo ""

# Check if uv is installed
if command -v uv &> /dev/null; then
    echo -e "${GREEN}✓ uv is installed${NC}"
    USE_UV=true
else
    echo -e "${YELLOW}⚠ uv is not installed. Using pip instead.${NC}"
    echo "  For faster installation, consider installing uv:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo ""
    USE_UV=false
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo -e "Python version: $PYTHON_VERSION"

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo -e "${RED}Error: Python 3.10 or higher is required${NC}"
    echo "Current version: $PYTHON_VERSION"
    exit 1
fi
echo -e "${GREEN}✓ Python version OK${NC}"
echo ""

# Initialize git submodules
echo -e "${YELLOW}Initializing git submodules...${NC}"
if [ -d ".git" ]; then
    git submodule update --init --recursive
    echo -e "${GREEN}✓ Submodules initialized${NC}"
else
    echo -e "${YELLOW}⚠ Not a git repository. Skipping submodule initialization.${NC}"
    echo "  If you cloned the repo, make sure .git directory exists."
fi
echo ""

# Install Cosmos Policy
echo -e "${YELLOW}Installing Cosmos Policy and dependencies...${NC}"
if [ "$USE_UV" = true ]; then
    uv pip install -e ".[$CUDA_VERSION]"
else
    # Install PyTorch first with correct index
    if [ "$CUDA_VERSION" = "cu128" ]; then
        pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128
    else
        pip install torch==2.9.0 torchvision --index-url https://download.pytorch.org/whl/cu130
    fi

    # Install the package
    pip install -e .

    # Install GPU packages
    echo -e "${YELLOW}Installing GPU-accelerated packages...${NC}"
    if [ "$CUDA_VERSION" = "cu128" ]; then
        pip install flash-attn>=2.7.3 transformer-engine>=2.2.0
        pip install xformers --index-url https://download.pytorch.org/whl/cu128
    else
        pip install flash-attn>=2.7.4 transformer-engine>=2.8.0
        pip install xformers --index-url https://download.pytorch.org/whl/cu130
    fi
fi
echo -e "${GREEN}✓ Cosmos Policy installed${NC}"
echo ""

# Install robot environments
echo -e "${YELLOW}Installing robot simulation environments...${NC}"

# LIBERO
if [ -d "LIBERO" ]; then
    echo "Installing LIBERO..."
    (cd LIBERO && pip install -e .)
    echo -e "${GREEN}✓ LIBERO installed${NC}"
else
    echo -e "${YELLOW}⚠ LIBERO directory not found. Skipping.${NC}"
fi

# robosuite
if [ -d "robosuite" ]; then
    echo "Installing robosuite..."
    (cd robosuite && pip install -e .)
    echo -e "${GREEN}✓ robosuite installed${NC}"
else
    echo -e "${YELLOW}⚠ robosuite directory not found. Skipping.${NC}"
fi

# RoboCasa
if [ -d "robocasa" ]; then
    echo "Installing RoboCasa..."
    (cd robocasa && pip install -e .)
    echo -e "${GREEN}✓ RoboCasa installed${NC}"
else
    echo -e "${YELLOW}⚠ RoboCasa directory not found. Skipping.${NC}"
fi
echo ""

# Verify installation
echo -e "${YELLOW}Verifying installation...${NC}"
echo ""

if [ -f "verify_installation.py" ]; then
    python verify_installation.py
    VERIFY_EXIT_CODE=$?

    if [ $VERIFY_EXIT_CODE -eq 0 ]; then
        echo ""
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}Installation completed successfully!${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        echo "Next steps:"
        echo "  1. For LIBERO evaluation: See LIBERO.md"
        echo "  2. For RoboCasa evaluation: See ROBOCASA.md"
        echo "  3. For ALOHA robot tasks: See ALOHA.md"
    else
        echo ""
        echo -e "${YELLOW}========================================${NC}"
        echo -e "${YELLOW}Installation completed with warnings${NC}"
        echo -e "${YELLOW}========================================${NC}"
        echo ""
        echo "Some optional components may be missing."
        echo "See SETUP.md for troubleshooting."
    fi
else
    echo -e "${YELLOW}⚠ verify_installation.py not found. Skipping verification.${NC}"
    echo ""
    echo -e "${GREEN}Installation completed!${NC}"
    echo "Please manually verify your installation."
fi
