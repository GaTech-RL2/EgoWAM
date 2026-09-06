#!/usr/bin/env python3
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

"""
Verification script for Cosmos Policy installation.

This script checks that all required dependencies are installed correctly
and that the environment is properly configured.

Usage:
    python verify_installation.py
"""

import sys


def check_import(module_name: str, package_name: str | None = None) -> bool:
    """Check if a module can be imported.

    Args:
        module_name: Name of the module to import
        package_name: Display name of the package (defaults to module_name)

    Returns:
        True if import succeeds, False otherwise
    """
    if package_name is None:
        package_name = module_name

    try:
        __import__(module_name)
        print(f"✓ {package_name} is installed")
        return True
    except ImportError as e:
        print(f"✗ {package_name} is NOT installed: {e}")
        return False


def check_cuda() -> bool:
    """Check CUDA availability and version."""
    try:
        import torch

        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "N/A"

            print("✓ CUDA is available")
            print(f"  - CUDA version: {cuda_version}")
            print(f"  - Number of GPUs: {device_count}")
            print(f"  - GPU 0: {device_name}")
            return True
        else:
            print("✗ CUDA is NOT available")
            return False
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
        return False


def check_torch_version() -> bool:
    """Check PyTorch version."""
    try:
        import torch

        version = torch.__version__
        print(f"✓ PyTorch version: {version}")

        # Check for minimum version
        major, minor = map(int, version.split(".")[:2])
        if major >= 2 and minor >= 7:
            return True
        else:
            print(f"  ⚠ Warning: PyTorch {version} detected. Recommended: >= 2.7.0")
            return True
    except Exception as e:
        print(f"✗ Error checking PyTorch version: {e}")
        return False


def check_optional_gpu_packages() -> dict[str, bool]:
    """Check optional GPU-accelerated packages."""
    results = {}

    print("\nOptional GPU packages:")
    results["flash_attn"] = check_import("flash_attn", "flash-attn")
    results["transformer_engine"] = check_import("transformer_engine", "transformer-engine")
    results["xformers"] = check_import("xformers", "xformers")

    return results


def check_robot_environments() -> dict[str, bool]:
    """Check robot simulation environments."""
    results = {}

    print("\nRobot simulation environments:")
    results["libero"] = check_import("libero", "LIBERO")
    results["robocasa"] = check_import("robocasa", "RoboCasa")
    results["robosuite"] = check_import("robosuite", "robosuite")

    return results


def check_deployment_packages() -> dict[str, bool]:
    """Check packages needed for deployment."""
    results = {}

    print("\nDeployment packages (optional):")
    results["fastapi"] = check_import("fastapi", "FastAPI")
    results["uvicorn"] = check_import("uvicorn", "uvicorn")
    results["json_numpy"] = check_import("json_numpy", "json-numpy")

    return results


def main() -> int:
    """Run all verification checks."""
    print("=" * 70)
    print("Cosmos Policy Installation Verification")
    print("=" * 70)

    print("\nPython version:")
    print(f"  {sys.version}")

    # Check core dependencies
    print("\nCore dependencies:")
    core_checks = [
        ("numpy", None),
        ("torch", "PyTorch"),
        ("torchvision", "torchvision"),
        ("PIL", "Pillow"),
        ("cv2", "opencv-python"),
        ("h5py", "h5py"),
        ("imageio", "imageio"),
        ("av", "PyAV"),
        ("einops", "einops"),
        ("attrs", "attrs"),
        ("omegaconf", "omegaconf"),
        ("hydra", "hydra-core"),
        ("draccus", "draccus"),
        ("tqdm", "tqdm"),
        ("wandb", "wandb"),
        ("transformers", "transformers"),
    ]

    core_results = []
    for module, package in core_checks:
        core_results.append(check_import(module, package))

    # Check Cosmos dependencies
    print("\nCosmos dependencies:")
    cosmos_results = []
    cosmos_results.append(check_import("cosmos_predict2", "cosmos-predict2"))

    # Try to import Cosmos Policy modules
    print("\nCosmos Policy modules:")
    policy_checks = [
        ("cosmos_policy.models.policy_text2world_model", "Policy Model"),
        ("cosmos_policy.datasets.dataset_common", "Datasets"),
        ("cosmos_policy.modules.cosmos_sampler", "Modules"),
    ]

    policy_results = []
    for module, package in policy_checks:
        policy_results.append(check_import(module, package))

    # Check PyTorch version
    print("\nPyTorch configuration:")
    torch_ok = check_torch_version()

    # Check CUDA
    cuda_ok = check_cuda()

    # Check optional packages
    gpu_results = check_optional_gpu_packages()
    robot_results = check_robot_environments()
    deployment_results = check_deployment_packages()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_core = len(core_results) + len(cosmos_results) + len(policy_results)
    passed_core = sum(core_results) + sum(cosmos_results) + sum(policy_results)

    print(f"\nCore dependencies: {passed_core}/{total_core} passed")
    print(f"PyTorch/CUDA: {'✓' if torch_ok and cuda_ok else '✗'}")

    gpu_passed = sum(gpu_results.values())
    print(f"Optional GPU packages: {gpu_passed}/{len(gpu_results)} installed")

    robot_passed = sum(robot_results.values())
    print(f"Robot environments: {robot_passed}/{len(robot_results)} installed")

    deployment_passed = sum(deployment_results.values())
    print(f"Deployment packages: {deployment_passed}/{len(deployment_results)} installed")

    # Final verdict
    print("\n" + "=" * 70)
    if passed_core == total_core and torch_ok and cuda_ok:
        print("✓ INSTALLATION SUCCESSFUL!")
        print("\nYour Cosmos Policy environment is ready for inference.")

        if robot_passed < len(robot_results):
            print("\n⚠ Note: Some robot environments are missing.")
            print("   Install them if you need to run evaluations:")
            print("   - cd LIBERO && pip install -e .")
            print("   - cd robosuite && pip install -e .")
            print("   - cd robocasa && pip install -e .")

        if gpu_passed < len(gpu_results):
            print("\n⚠ Note: Some GPU packages are missing.")
            print("   These are optional but recommended for better performance.")
            print("   Install with: pip install flash-attn transformer-engine xformers")

        return 0
    else:
        print("✗ INSTALLATION INCOMPLETE")
        print("\nPlease install missing dependencies.")
        print("See SETUP.md for detailed installation instructions.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
