"""
Generate kaggle/gpu_server/gpu_server.ipynb from the two cell sources.

The .py files are the single source of truth so the notebook can't drift from
them. The notebook is only a delivery vehicle - ready to paste or import into the
Kaggle browser editor. Nothing here pushes or executes anything on Kaggle.

    python scripts/build_kaggle_notebook.py
    python scripts/build_kaggle_notebook.py --clip fetch    # cell -> clipboard
    python scripts/build_kaggle_notebook.py --clip server

Cell order matters: the fetch cell finishes, the server cell never returns.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GPU = ROOT / "kaggle" / "gpu_server"
FETCH = GPU / "fetch_dataset_cell.py"
SERVER = GPU / "kaggle_gpu_server.py"
OUT = GPU / "gpu_server.ipynb"

HEADER = """# AHC hackathon - Kaggle session

One interactive session does both jobs: it holds the dataset and it holds the
models. Set it up once and drive it from the laptop.

**Session options:** Accelerator **GPU T4 x2**, Internet **On**.
**Add-ons -> Secrets:** `KAGGLE_REMOTE_TOKEN` (same value as the laptop's `.env`),
and `GDRIVE_TOKEN` if you want the authenticated dataset route
(`python scripts/rclone_token.py` on the laptop mints it).

Run the cells **in order** - cell 1 finishes, cell 2 never returns.
"""

FETCH_MD = """## 1 - Get the dataset into Kaggle

The training pack only exists to fine-tune on, and that happens here, so it never
needs to touch the laptop. Kaggle's egress is much faster than a home connection,
and once this is saved as a Kaggle Dataset it mounts read-only at
`/kaggle/input/<slug>/` in every future session - downloaded exactly once, ever.

(Kaggle has no `drive.mount()`; that's Colab. This is the equivalent, and it's
better, because the result is permanent rather than re-mounted every session.)
"""

SERVER_MD = """## 2 - Start the GPU server

Opens a `trycloudflare` tunnel and serves a token-guarded API. Paste the printed
URL into the laptop's `.env` as `KAGGLE_REMOTE_URL`.

**This cell never returns on purpose** - a busy kernel is what stops Kaggle
idling the session out. Run it last.
"""


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", choices=["fetch", "server"], default=None,
                    help="copy one cell's source to the clipboard")
    args = ap.parse_args()

    fetch_src = FETCH.read_text(encoding="utf-8")
    server_src = SERVER.read_text(encoding="utf-8")

    nb = {
        "cells": [md(HEADER), md(FETCH_MD), code(fetch_src), md(SERVER_MD), code(server_src)],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  cell 1 (fetch dataset): {len(fetch_src.splitlines())} lines")
    print(f"  cell 2 (gpu server):    {len(server_src.splitlines())} lines")

    if args.clip:
        src = fetch_src if args.clip == "fetch" else server_src
        subprocess.run("clip", input=src.encode("utf-8"), check=False, shell=True)
        print(f"\n{args.clip} cell copied to clipboard - paste into the Kaggle editor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
