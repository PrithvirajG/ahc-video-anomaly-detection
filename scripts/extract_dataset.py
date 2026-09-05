"""
Extract the Google-Drive-split dataset zips into data/.

Drive splits a large folder download into N *independent* zips, not a spanned
multi-part archive - each one is a complete, openable zip holding a subset of
the tree. So they extract in any order, and a missing part costs you those files
rather than corrupting the whole set.

Every part is rooted at "Train and Test/"; that prefix is stripped so the result
is data/train/<class>/... and data/test/..., which is the layout the notebook's
find_data_root() and download_dataset.py --verify both expect.

    python scripts/extract_dataset.py                      # default glob in Downloads
    python scripts/extract_dataset.py --src "C:/some/dir"
    python scripts/extract_dataset.py --dest data --force  # re-extract everything

Re-running skips files already on disk at the right size, so an interrupted run
resumes instead of starting over.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = Path.home() / "Downloads"
DEFAULT_GLOB = "Train and Test-*.zip"
STRIP_PREFIX = "Train and Test/"


def human(n: float) -> str:
    return f"{n / 1e9:.2f} GB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC), help="directory holding the zips")
    ap.add_argument("--glob", default=DEFAULT_GLOB)
    ap.add_argument("--dest", default=str(ROOT / "data"))
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    args = ap.parse_args()

    src = Path(args.src)
    dest = Path(args.dest)
    parts = sorted(src.glob(args.glob))
    if not parts:
        print(f"no zips matching {args.glob!r} in {src}", file=sys.stderr)
        return 1

    dest.mkdir(parents=True, exist_ok=True)
    print(f"{len(parts)} parts -> {dest}\n")

    t0 = time.time()
    written = skipped = failed = 0
    bytes_out = 0

    for pi, part in enumerate(parts, 1):
        print(f"[{pi}/{len(parts)}] {part.name}")
        try:
            zf = zipfile.ZipFile(part)
        except zipfile.BadZipFile as e:
            print(f"    ! unreadable, skipping: {e}", file=sys.stderr)
            failed += 1
            continue

        with zf:
            infos = [i for i in zf.infolist() if not i.is_dir()]
            for n, info in enumerate(infos, 1):
                rel = info.filename
                if rel.startswith(STRIP_PREFIX):
                    rel = rel[len(STRIP_PREFIX):]
                if not rel:
                    continue
                # Zip entries are attacker-controlled paths in general; resolve
                # and confirm containment before writing anything.
                out = (dest / rel).resolve()
                if not str(out).startswith(str(dest.resolve())):
                    print(f"    ! refusing path outside dest: {info.filename}", file=sys.stderr)
                    failed += 1
                    continue

                if out.exists() and not args.force and out.stat().st_size == info.file_size:
                    skipped += 1
                    continue

                out.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as fsrc, open(out, "wb") as fdst:
                    while chunk := fsrc.read(1 << 20):
                        fdst.write(chunk)
                written += 1
                bytes_out += info.file_size
                if n % 50 == 0 or n == len(infos):
                    print(f"    {n}/{len(infos)}  {human(bytes_out)} written, "
                          f"{skipped} skipped", flush=True)

    dt = time.time() - t0
    print(f"\ndone in {dt / 60:.1f} min: {written} written ({human(bytes_out)}), "
          f"{skipped} already present, {failed} failed")

    mp4 = list(dest.rglob("*.mp4"))
    size = sum(p.stat().st_size for p in mp4)
    print(f"\n{len(mp4)} mp4 files, {human(size)} in {dest}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
