"""
Pull the AHC train+test pack (~15-17GB) from the shared Drive mirrors.

    python scripts/download_dataset.py --rclone    # authenticated (USE THIS)
    python scripts/download_dataset.py             # anonymous gdown, all mirrors
    python scripts/download_dataset.py --mirror 3  # anonymous, from mirror 3 on
    python scripts/download_dataset.py --verify    # audit data/, download nothing

Two backends, because the anonymous one is already known to fail here:

  --rclone  Drive API with your own OAuth credentials. Requires a one-time
            `rclone config create gdrive drive scope=drive.readonly`, which opens
            a browser. This is the backend that works.

  (default) gdown, no credentials. Enumerates all 16,170 files across every
            mirror fine, then dies on the first actual byte with "Too many users
            have viewed or downloaded this file recently" - Google's anonymous
            per-file quota, tripped on all five mirrors at 04:15 on event day.
            This is the same wall that cost time on the 22-Aug hackathon; it is
            kept here only so the failure is reproducible rather than surprising.

Both backends resume, so re-running after a failure only fetches what is missing.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_ROOT = Path(os.getenv("DATA_ROOT", ROOT / "data"))
if not DATA_ROOT.is_absolute():
    DATA_ROOT = (ROOT / DATA_ROOT).resolve()

# Google's quota refusal comes back as an HTML body, not an HTTP error code, so
# gdown surfaces it as a generic failure - match on the wording instead.
QUOTA_MARKERS = ("quota", "too many users", "cannot retrieve the folder", "permission")


def mirrors() -> list[tuple[int, str]]:
    out = []
    for i in range(1, 6):
        url = os.getenv(f"AHC_DRIVE_MIRROR_{i}", "").strip()
        if url and "folders/" in url:
            out.append((i, url))
    return out


def try_mirror(n: int, url: str) -> bool:
    import gdown

    print(f"\n=== mirror {n}: {url}")
    try:
        # gdown 6.x dropped `remaining_ok` (5.x needed it to get past the 50-file
        # folder cap); `resume` is the one that matters now - it makes a re-run
        # after a mirror failure skip whatever already landed.
        gdown.download_folder(
            url=url,
            output=str(DATA_ROOT),
            quiet=False,
            use_cookies=False,
            resume=True,
        )
        return True
    except Exception as e:  # noqa: BLE001 - any failure means "try the next mirror"
        msg = str(e).lower()
        why = "quota/permission" if any(m in msg for m in QUOTA_MARKERS) else "error"
        print(f"[mirror {n}] failed ({why}): {e}", file=sys.stderr)
        return False


def pull_test_only() -> int:
    """Pull just the 34-video public test set.

    The split that matters: the ~15-17GB train pack only exists to fine-tune on,
    and that happens on Kaggle - so it never needs to touch this machine (see
    kaggle/gpu_server/fetch_dataset_cell.py). The test set is small, has public
    ground truth, and is what the local dashboard and scoring pipeline are built
    against. That is the only piece worth pulling down a home connection.
    """
    import gdown

    fid = os.getenv("AHC_TEST_FOLDER_ID", "").strip()
    if not fid:
        print("AHC_TEST_FOLDER_ID not set in .env", file=sys.stderr)
        return 1

    out = DATA_ROOT / "test"
    out.mkdir(parents=True, exist_ok=True)
    print(f"pulling public test set only -> {out}")
    try:
        gdown.download_folder(id=fid, output=str(out), quiet=False,
                              use_cookies=False, resume=True)
    except Exception as e:
        print(f"failed: {str(e).splitlines()[0]}", file=sys.stderr)
        print("\nSame anonymous quota as the full pack. Use --rclone, or let Kaggle "
              "fetch it (docs/KAGGLE_REMOTE.md).", file=sys.stderr)
        return 1
    verify()
    return 0


def find_rclone() -> str | None:
    """winget installs rclone but only edits PATH for *new* shells, so look for it."""
    import shutil

    exe = shutil.which("rclone")
    if exe:
        return exe
    packages = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    hits = sorted(packages.glob("Rclone.Rclone*/**/rclone.exe")) if packages.exists() else []
    return str(hits[0]) if hits else None


def rclone_remote_exists(exe: str, name: str) -> bool:
    p = subprocess.run([exe, "listremotes"], capture_output=True, text=True)
    return f"{name}:" in p.stdout


def rclone_pull(remote: str = "gdrive") -> int:
    """Authenticated pull. Uses the folder id as the remote root, so the shared
    'Anyone with the link' folder is walked directly - no shortcut-to-Drive step."""
    exe = find_rclone()
    if not exe:
        print("rclone not found. Install it with:\n"
              "    winget install --id Rclone.Rclone -e", file=sys.stderr)
        return 1

    if not rclone_remote_exists(exe, remote):
        print(f"""
rclone has no '{remote}' remote yet. This is a one-time browser sign-in.

Run this yourself (it opens a browser for Google's consent screen - I can't
complete an interactive OAuth flow for you):

    ! "{exe}" config create {remote} drive scope=drive.readonly

Accept the read-only scope, then re-run:

    python scripts/download_dataset.py --rclone
""".strip(), file=sys.stderr)
        return 2

    ms = mirrors()
    if not ms:
        print("No AHC_DRIVE_MIRROR_* entries in .env", file=sys.stderr)
        return 1

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    for n, url in ms:
        folder_id = url.rstrip("/").split("/folders/")[-1].split("?")[0]
        print(f"\n=== mirror {n} via rclone (folder id {folder_id})")
        cmd = [
            exe, "copy", f"{remote}:", str(DATA_ROOT),
            "--drive-root-folder-id", folder_id,
            "--progress",
            "--transfers", "8",       # Drive API is happier with parallel small files
            "--checkers", "16",
            "--drive-chunk-size", "64M",
            "--tpslimit", "10",       # stay under the per-user API rate limit
            "--retries", "3",
            "--low-level-retries", "10",
            "--stats", "15s",
        ]
        rc = subprocess.run(cmd).returncode
        if rc == 0:
            print(f"\nmirror {n} completed.")
            verify()
            return 0
        print(f"[mirror {n}] rclone exited {rc}; trying the next mirror", file=sys.stderr)

    print("\nEvery mirror failed even authenticated. Next moves:", file=sys.stderr)
    print("  - ask the organisers for a Kaggle Dataset or torrent mirror", file=sys.stderr)
    print("  - or copy a mirror into your own Drive in the browser and pull that", file=sys.stderr)
    print("    copy instead (a file you own has no 'too many users' quota)", file=sys.stderr)
    return 1


def verify() -> None:
    """Audit what actually landed against the layout the dataset doc promises."""
    print(f"\n=== audit of {DATA_ROOT}")
    if not DATA_ROOT.exists():
        print("  (nothing downloaded yet)")
        return

    videos = list(DATA_ROOT.rglob("*.mp4"))
    csvs = list(DATA_ROOT.rglob("*.csv"))
    total_gb = sum(p.stat().st_size for p in videos) / 1e9
    print(f"  {len(videos)} mp4 files, {total_gb:.2f} GB")
    print(f"  {len(csvs)} csv files")

    # train/<class>/videos/*.mp4 + videos.csv + ground_truth.csv per class folder
    train = DATA_ROOT / "train"
    if train.exists():
        print("\n  train/ class folders:")
        for d in sorted(p for p in train.iterdir() if p.is_dir()):
            n = len(list(d.rglob("*.mp4")))
            gt = "gt" if (d / "ground_truth.csv").exists() else "--"
            vc = "vids.csv" if (d / "videos.csv").exists() else "--"
            print(f"    {d.name:38s} {n:5d} clips   {gt:3s} {vc}")
    else:
        print("  ! no train/ directory")

    test = DATA_ROOT / "test"
    if test.exists():
        n = len(list(test.rglob("*.mp4")))
        print(f"\n  test/: {n} clips (doc says 34 videos / ~56 min)")
    else:
        print("  ! no test/ directory")

    # The twelve label strings must match exactly - a typo here silently tanks scoring.
    expected = {
        "normal", "traffic_accident", "traffic_congestion",
        "stalled_or_broken_down_vehicle", "vehicle_blocking_traffic",
        "wrong_way_driving", "road_spill_or_debris", "waterlogging_or_flood",
        "fire", "smoke", "fighting_or_violence", "loitering_or_suspicious_presence",
    }
    if train.exists():
        found = {p.name for p in train.iterdir() if p.is_dir()}
        missing, extra = expected - found, found - expected
        if missing:
            print(f"\n  ! missing expected classes: {sorted(missing)}")
        if extra:
            print(f"  ! unexpected folders: {sorted(extra)}")
        if not missing and not extra:
            print("\n  all 12 label folders present and correctly named")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", type=int, default=None, help="start at this mirror (1-5)")
    ap.add_argument("--verify", action="store_true", help="audit data/ only, no download")
    ap.add_argument("--rclone", action="store_true",
                    help="authenticated Drive API pull - the one that works")
    ap.add_argument("--remote", default="gdrive", help="rclone remote name")
    ap.add_argument("--test-only", action="store_true",
                    help="just the 34-video public test set (train belongs on Kaggle)")
    args = ap.parse_args()

    if args.verify:
        verify()
        return 0

    if args.test_only:
        return pull_test_only()

    if args.rclone:
        return rclone_pull(args.remote)

    ms = mirrors()
    if not ms:
        print("No AHC_DRIVE_MIRROR_* entries in .env", file=sys.stderr)
        return 1
    if args.mirror:
        ms = [m for m in ms if m[0] >= args.mirror]

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"target: {DATA_ROOT}   ({len(ms)} mirror(s) to try)")

    for n, url in ms:
        if try_mirror(n, url):
            print(f"\nmirror {n} completed.")
            verify()
            return 0

    print("\nAll mirrors failed. Options, in order of how fast they are:", file=sys.stderr)
    print("  1. open a mirror in the browser and 'Add shortcut to Drive', then use", file=sys.stderr)
    print("     an authenticated pull (gdown --folder with cookies, or rclone)", file=sys.stderr)
    print("  2. download in the browser and unzip into data/ by hand", file=sys.stderr)
    print("  3. ask an organiser for a Kaggle Dataset mirror - that is what we", file=sys.stderr)
    print("     fell back to on 22-Aug when the quota hit (see docs/KAGGLE.md)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
