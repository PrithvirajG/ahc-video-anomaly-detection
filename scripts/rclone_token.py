"""
Mint a Google Drive OAuth token that Kaggle (or Colab, or any headless box) can reuse.

    python scripts/rclone_token.py

Runs `rclone authorize "drive"`, which opens a browser for Google's consent
screen and prints a one-line JSON token. That token is what lets a machine with
no browser - a Kaggle session - authenticate as you.

Paste the output into Kaggle under **Add-ons -> Secrets** as `GDRIVE_TOKEN`, and
`kaggle/gpu_server/fetch_dataset_cell.py` picks it up automatically.

Scope is read-only: this can list and download from Drive, nothing else.

The token is a credential - treat it like a password. It is printed to the
terminal on purpose (it has to be copied into Kaggle) but is never written to
this repo.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def find_rclone() -> str | None:
    """winget only edits PATH for *new* shells, so look in its package dir too."""
    exe = shutil.which("rclone")
    if exe:
        return exe
    import os

    pkgs = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    hits = sorted(pkgs.glob("Rclone.Rclone*/**/rclone.exe")) if pkgs.exists() else []
    return str(hits[0]) if hits else None


def main() -> int:
    exe = find_rclone()
    if not exe:
        print("rclone not found. Install it with:\n"
              "    winget install --id Rclone.Rclone -e", file=sys.stderr)
        return 1

    print("A browser window will open for Google sign-in (read-only Drive scope).")
    print("Approve it, then copy the JSON line printed below into Kaggle Secrets")
    print("as GDRIVE_TOKEN.\n")
    print("-" * 70)

    # rclone prints the token between its own banner lines; pass it straight through
    # so the user can copy it, and never persist it anywhere in the repo.
    return subprocess.run([exe, "authorize", "drive", "--auth-no-open-browser"]).returncode


if __name__ == "__main__":
    raise SystemExit(main())
