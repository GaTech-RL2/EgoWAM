"""Preflight for the gated Cosmos tokenizer -- fail in seconds, not 70 minutes.

Every world-model config instantiates ``CosmosVAEWrapper``, including the 3D-flow variant that
never uses the image latents. That instantiation happens *after* the S3 data sync and the
norm-stat pass, so a missing-access failure surfaces over an hour into a run. This checks the
same asset up front, using only ``huggingface_hub`` (no torch import).

Two ways to pass, because either is sufficient for the real run:
  1. the file already resolves from the local cache (no network, no token needed), or
  2. the account has access, so it will download on first use.
"""

import sys

REPO = "nvidia/Cosmos-Predict2-2B-Video2World"
FILE = "tokenizer/tokenizer.pth"


def main() -> int:
    import os

    from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url

    try:
        path = hf_hub_download(REPO, FILE, local_files_only=True)
        print(f"[gated-assets] OK -- cached at {path}")
        return 0
    except Exception:
        pass  # not cached; fall through to an access check

    # HEAD the file's resolve URL, not model_info: a gated repo still serves public metadata
    # to unauthorized callers, so model_info returns 200 and the download then 401s. The gate
    # is enforced per-file at resolve/, which is exactly where the real failure surfaces.
    try:
        get_hf_file_metadata(hf_hub_url(REPO, FILE))
        print(f"[gated-assets] OK -- authorized for {REPO} (downloads on first use)")
        return 0
    except Exception as exc:
        print(
            f"[gated-assets] FAIL -- cannot reach {REPO}/{FILE}\n"
            f"  {type(exc).__name__}: {str(exc).splitlines()[0][:120]}\n"
            f"  HF_HOME={os.environ.get('HF_HOME', '<unset>')}\n"
            "\n"
            "  Every world-model config builds the Cosmos VAE, so this asset is required even\n"
            "  for the 3D-flow variant. Either:\n"
            f"    - request access at https://huggingface.co/{REPO} then run `hf auth login`, or\n"
            "    - point HF_HOME at a cache that already holds the tokenizer.\n"
            "\n"
            "  Failing now rather than after the data sync and norm-stat pass.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
