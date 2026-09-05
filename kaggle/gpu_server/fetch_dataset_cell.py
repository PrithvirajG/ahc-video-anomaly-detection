# ============================================================================
# AHC hackathon - pull the dataset straight into Kaggle, once, then never again.
#
# Run this as a SECOND CELL in the same interactive session as the GPU server.
#
# The point: the fine-tuning happens on Kaggle, so there is no reason to drag
# 15-17GB down a home connection and push it back up. Kaggle's egress is far
# faster, and once the pack is saved as a Kaggle Dataset it mounts read-only at
# /kaggle/input/<slug>/ in every future session - no re-download, ever. That is
# the closest thing Kaggle has to Colab's drive.mount(), and it is better,
# because it is paid for exactly once.
#
# Three routes, tried in order. Route A is free to attempt and sometimes just
# works; B is the reliable one; C is the last resort.
# ============================================================================

import os
import subprocess
import sys

DEST = "/kaggle/working/data"
os.makedirs(DEST, exist_ok=True)

# The five mirrors are alternates of the SAME pack. Folder ids only - the rest of
# the URL is noise to every tool here.
MIRRORS = [
    "1sEFKR7ctd5GfFw-nMlYd_MnTw1VVYz9K",
    "13E_CePn14lcbwMA_yZEiHpAVx6i09UIG",
    "13V8JqgZRMzn2TCF0HTsCqVgUH0UOMmpb",
    "1fS_i7QKXRDI6mnaI6UWqYzKSOYWG8rFv",
    "1efhUZhB6Kyvpw3RulZJSwd0brb8KhuZf",
]


def size_gb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 1e9


def count_mp4(path: str) -> int:
    return sum(len([f for f in files if f.endswith(".mp4")]) for _, _, files in os.walk(path))


# --- Route A: anonymous gdown -----------------------------------------------
# This failed from the laptop at 04:15 on event day - Google's per-file quota,
# on all five mirrors. That quota is nominally global to the file, not per-IP,
# so this "should" fail here too. It is attempted anyway because it costs two
# minutes, needs no credentials, and the 24h lockout may simply have aged out.
def route_a() -> bool:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "gdown"], check=False)
    import gdown

    for i, fid in enumerate(MIRRORS, 1):
        print(f"\n--- [A] mirror {i} via anonymous gdown")
        try:
            gdown.download_folder(id=fid, output=DEST, quiet=False,
                                  use_cookies=False, resume=True)
            if count_mp4(DEST) > 100:
                return True
        except Exception as e:
            head = str(e).split("\n")[0][:160]
            print(f"    mirror {i} failed: {head}")
    return False


# --- Route B: rclone with your own OAuth ------------------------------------
# Authenticated access uses your account rather than the anonymous pool. Get the
# token by running this ONCE on the laptop:
#
#     python scripts/rclone_token.py
#
# It prints a one-line JSON token. Paste it into Kaggle under
# Add-ons -> Secrets as GDRIVE_TOKEN, then this route configures itself.
def route_b() -> bool:
    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("GDRIVE_TOKEN")
    except Exception:
        token = os.environ.get("GDRIVE_TOKEN", "")
    if not token:
        print("\n--- [B] skipped: no GDRIVE_TOKEN secret "
              "(run `python scripts/rclone_token.py` on the laptop)")
        return False

    if not os.path.exists("/usr/bin/rclone") and not os.path.exists("/kaggle/working/rclone"):
        subprocess.run("curl -s https://rclone.org/install.sh | sudo bash",
                       shell=True, check=False)

    subprocess.run(
        ["rclone", "config", "create", "gdrive", "drive",
         "scope", "drive.readonly", "config_token", token],
        check=False,
    )

    for i, fid in enumerate(MIRRORS, 1):
        print(f"\n--- [B] mirror {i} via authenticated rclone")
        rc = subprocess.run(
            ["rclone", "copy", "gdrive:", DEST,
             "--drive-root-folder-id", fid,
             "--transfers", "12", "--checkers", "24",
             "--drive-chunk-size", "64M", "--tpslimit", "10",
             "--retries", "3", "--low-level-retries", "10",
             "--stats", "20s", "--progress"],
        ).returncode
        if rc == 0 and count_mp4(DEST) > 100:
            return True
        print(f"    mirror {i} exited {rc}")
    return False


print("=" * 78)
ok = route_a() or route_b()
print("=" * 78)
print(f"{count_mp4(DEST)} mp4 files, {size_gb(DEST):.2f} GB in {DEST}")

if not ok:
    print("""
Both automated routes failed. Route C, by hand, ~2 minutes and it always works:

  1. Open a mirror in your browser (signed in to Google).
  2. Select the train/ and test/ folders -> right-click -> "Make a copy",
     or "Organise -> Add shortcut to Drive" into a folder you own.
     Copying is server-side: no bytes move, and the copy is YOURS, so it
     carries no "too many users" quota.
  3. Re-run this cell - route B will find your copy through your own account.

Note the copy trick is reported to be unreliable for large binaries when a file
is already quota-locked; if Drive refuses, the shortcut route or asking the
organisers for a Kaggle Dataset mirror is the fallback.
""")
else:
    print("""
Now make it permanent so no future session re-downloads anything:

    File -> Save Version -> "Save & Run All"   (output becomes a Dataset)
  or, from a cell:
    !kaggle datasets init -p /kaggle/working/data
    !kaggle datasets create -p /kaggle/working/data --dir-mode zip

Afterwards attach it via Add data -> Your Datasets, and it mounts read-only at
/kaggle/input/<slug>/ in every session, instantly. Check the free-tier dataset
size cap first - this pack is 15-17GB and /kaggle/working itself caps at ~20GB.
""")
