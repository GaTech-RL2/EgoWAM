default:
  just --list

package_name := "cosmos-policy"
module_name := "cosmos_policy"
short_name := "policy"

default_cuda_name := "cu128"

# Install the repository
install cuda_name=default_cuda_name *args:
  echo {{cuda_name}} > .cuda-name
  uv sync --extra={{cuda_name}} {{args}}

# Setup pre-commit
_pre-commit-setup:
  uv tool install "pre-commit>=4.3.0"
  pre-commit install -c .pre-commit-config-base.yaml

# Run pre-commit
pre-commit *args: _pre-commit-setup
  pre-commit run -a {{args}} || pre-commit run -a {{args}}

# Run pyrefly
pyrefly *args:
  uv run --group dev pyrefly check --output-format=min-text --remove-unused-ignores {{args}}

# Run linting and formatting
lint: pre-commit

# https://spdx.org/licenses/
allow_licenses := "MIT BSD-2-CLAUSE BSD-3-CLAUSE APACHE-2.0 ISC"
ignore_package_licenses := "nvidia-* hf-xet certifi filelock matplotlib typing-extensions sentencepiece matplotlib-inline"

# Run licensecheck
_licensecheck *args:
  uvx licensecheck --show-only-failing --only-licenses {{allow_licenses}} --ignore-packages {{ignore_package_licenses}} --zero {{args}}

# Run pip-licenses
_pip-licenses *args: install
  uvx pip-licenses --python .venv/bin/python --format=plain-vertical --with-license-file --no-license-path --no-version --with-urls --output-file ATTRIBUTIONS.md {{args}}
  pre-commit run --files ATTRIBUTIONS.md || true

# Update the license
license: _licensecheck _pip-licenses

# Run link-check
_link-check *args:
  pre-commit run -a --hook-stage manual link-check {{args}}
