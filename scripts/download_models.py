"""
Prefetch the model weights we want resident locally, into ./models/hf.

    python scripts/download_models.py            # the default set
    python scripts/download_models.py --set all  # + the second VLM candidate
    python scripts/download_models.py --list     # show sets, download nothing

What runs where, given a 4GB GTX 1650:
  - The CLIP/SigLIP-class frame encoders are the always-on filter stage of the
    cascade (Cerberus's stage 1). They fit comfortably and run at useful speed
    locally, so the whole coarse pipeline can be built and demoed on this laptop.
  - Qwen2.5-VL-3B is the fine-grained reasoning stage. In 4-bit it just fits;
    in bf16 it does not. Either way it is the thing to prototype against.
  - Fine-tuning anything, and any 4B+ backbone, is Kaggle/Colab work. Those
    environments download weights far faster than this connection does, so
    there is no point pulling them here.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# Set HF_HOME before huggingface_hub is imported, so the cache actually lands on D:.
hf_home = os.getenv("HF_HOME", "./models/hf")
if not os.path.isabs(hf_home):
    hf_home = str((ROOT / hf_home).resolve())
os.environ["HF_HOME"] = hf_home
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

# The HF cache symlinks snapshots/ at blobs/. Creating a symlink on Windows needs
# either Developer Mode or an elevated process, and without one you get
# "WinError 1314: A required privilege is not held by the client" *after* the
# bytes have already been downloaded. Copying instead costs some disk (the blob
# and the snapshot copy) and buys not having to elevate anything.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

SETS: dict[str, list[tuple[str, str]]] = {
    # (repo_id, what it is for)
    "encoders": [
        ("google/siglip2-base-patch16-224",
         "stage-1 frame encoder; the SigLIP2 we already validated on 22-Aug"),
        ("openai/clip-vit-base-patch16",
         "CLIP baseline - what Alert-CLIP measures against, and the cheapest filter"),
        ("facebook/PE-Core-L14-336",
         "Perception Encoder; the exact stage-1 model Cerberus reports 98.41 FPS with"),
    ],
    "vlm": [
        ("Qwen/Qwen2.5-VL-3B-Instruct",
         "stage-2 reasoning; smallest backbone with real video support"),
    ],
    "vlm_alt": [
        ("Qwen/Qwen3-VL-4B-Instruct",
         "the other backbone worth A/B-ing; needs a T4 to be comfortable"),
    ],
}
SETS["default"] = SETS["encoders"] + SETS["vlm"]
SETS["all"] = SETS["encoders"] + SETS["vlm"] + SETS["vlm_alt"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="default", choices=sorted(SETS))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for name, items in sorted(SETS.items()):
            print(f"\n{name}:")
            for repo, why in items:
                print(f"  {repo:42s}  {why}")
        return 0

    from huggingface_hub import snapshot_download

    token = os.getenv("HF_TOKEN") or None
    if not token or token.startswith("hf_..."):
        token = None
        print("! No HF_TOKEN - public repos still work, just rate-limited.\n")

    print(f"cache: {hf_home}\n")
    failed = []
    for repo, why in SETS[args.set]:
        print(f"=== {repo}\n    ({why})")
        try:
            path = snapshot_download(
                repo_id=repo,
                token=token,
                # Skip the duplicate .bin copies when safetensors exist - roughly
                # halves the transfer on repos that ship both.
                ignore_patterns=["*.pth", "*.bin", "*.msgpack", "*.h5"],
            )
            print(f"    -> {path}\n")
        except Exception as e:  # noqa: BLE001
            print(f"    FAILED: {e}\n", file=sys.stderr)
            failed.append(repo)

    if failed:
        print(f"\n{len(failed)} repo(s) failed: {failed}", file=sys.stderr)
        print("Gated repos need `huggingface-cli login` or an HF_TOKEN with access.",
              file=sys.stderr)
        return 1

    print("All weights cached.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
