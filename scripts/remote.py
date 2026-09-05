"""
CLI for the persistent Kaggle GPU session. See docs/KAGGLE_REMOTE.md.

    python scripts/remote.py token                  # generate a shared token
    python scripts/remote.py health                 # GPU, uptime, what's loaded
    python scripts/remote.py run "print(1+1)"       # run a snippet there
    python scripts/remote.py runfile pipeline/x.py  # send a local file
    python scripts/remote.py shell "nvidia-smi"
    python scripts/remote.py ls /kaggle/working
    python scripts/remote.py push data/test/videos/T001.mp4 /kaggle/working/clips
    python scripts/remote.py pull /kaggle/working/scores.parquet runs/
"""
from __future__ import annotations

import json
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.kaggle_remote import Remote, RemoteError  # noqa: E402


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _show_run(r: dict) -> None:
    if r.get("stdout"):
        print(r["stdout"], end="")
    if r.get("stderr"):
        print(r["stderr"], end="", file=sys.stderr)
    if r.get("result") is not None:
        print(r["result"])
    print(f"\n[{'ok' if r['ok'] else 'FAILED'} in {r['elapsed_sec']}s]", file=sys.stderr)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    cmd, args = sys.argv[1], sys.argv[2:]

    # `token` must work before a session exists, so it comes first.
    if cmd == "token":
        t = secrets.token_urlsafe(24)
        print(t)
        print("\nPut this in BOTH places:", file=sys.stderr)
        print(f"  laptop .env :  KAGGLE_REMOTE_TOKEN={t}", file=sys.stderr)
        print("  Kaggle      :  Add-ons -> Secrets -> KAGGLE_REMOTE_TOKEN", file=sys.stderr)
        return 0

    try:
        r = Remote()
    except RemoteError as e:
        print(e, file=sys.stderr)
        return 1

    try:
        if cmd == "health":
            _print(r.health())
        elif cmd == "run":
            _show_run(r.run(args[0], raise_on_error=False))
        elif cmd == "runfile":
            _show_run(r.run_file(args[0], raise_on_error=False))
        elif cmd == "shell":
            res = r.shell(args[0])
            print(res["stdout"], end="")
            print(res["stderr"], end="", file=sys.stderr)
        elif cmd == "ls":
            _print(r.ls(args[0] if args else "/kaggle/working"))
        elif cmd == "push":
            _print(r.push(args[0], args[1] if len(args) > 1 else "/kaggle/working"))
        elif cmd == "pull":
            print(r.pull(args[0], args[1] if len(args) > 1 else "."))
        else:
            print(__doc__)
            return 1
    except RemoteError as e:
        print(f"remote error: {e}", file=sys.stderr)
        return 1
    except IndexError:
        print(f"missing argument for '{cmd}'\n{__doc__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
