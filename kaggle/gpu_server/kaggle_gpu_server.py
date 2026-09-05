# ============================================================================
# AHC hackathon - persistent GPU server for a Kaggle interactive session.
#
# PASTE THIS WHOLE FILE INTO ONE CELL of a Kaggle notebook and run it.
#   Session options -> Accelerator: GPU T4 x2, Internet: ON.
#
# Why it exists: `kaggle kernels push` re-executes the entire notebook in a
# fresh container every time, so a one-line fix costs a full round of pip
# installs and multi-GB weight downloads. Instead this cell runs ONCE, holds
# the models in memory, and exposes them over a public URL. From then on we
# iterate by sending code and requests from the laptop - installs and weights
# are paid for exactly once per session.
#
# It is a remote Jupyter kernel by another name: same trust model (your own
# session, your own token), just reachable from the laptop instead of only
# from the browser tab.
#
# The cell deliberately never returns - a busy kernel is what keeps the
# interactive session from idle-disconnecting.
# ============================================================================

import os
import sys
import io
import re
import time
import threading
import traceback
import subprocess
import contextlib

PORT = 8000
STATE = {"tunnel_url": None, "started": time.time()}

# --- 0. auth ---------------------------------------------------------------
# A trycloudflare URL is public and unguessable-but-not-secret, so every route
# sits behind a bearer token. Set the same value in the laptop's .env as
# KAGGLE_REMOTE_TOKEN. Kaggle Secrets (Add-ons -> Secrets) is the tidy way to
# supply it; the env fallback keeps this cell self-contained.
try:
    from kaggle_secrets import UserSecretsClient

    TOKEN = UserSecretsClient().get_secret("KAGGLE_REMOTE_TOKEN")
except Exception:
    TOKEN = os.environ.get("KAGGLE_REMOTE_TOKEN", "")

if not TOKEN:
    raise SystemExit(
        "No KAGGLE_REMOTE_TOKEN. Add it under Add-ons -> Secrets (recommended), or set "
        "os.environ['KAGGLE_REMOTE_TOKEN'] in a cell above this one. Use the same value "
        "the laptop's .env has - `python scripts/remote.py token` prints a fresh one."
    )

# --- 1. deps ---------------------------------------------------------------
# The Kaggle image already ships torch/transformers; only the server bits are missing.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "fastapi", "uvicorn[standard]", "python-multipart"],
    check=False,
)

# --- 2. cloudflared --------------------------------------------------------
# Quick tunnels need no Cloudflare account and no domain: `cloudflared tunnel
# --url` prints a random *.trycloudflare.com hostname. Capped at 200 concurrent
# requests, which is far more than one laptop needs.
CF = "/kaggle/working/cloudflared"
if not os.path.exists(CF):
    subprocess.run(
        ["wget", "-q", "-O", CF,
         "https://github.com/cloudflare/cloudflared/releases/latest/download/"
         "cloudflared-linux-amd64"],
        check=True,
    )
    os.chmod(CF, 0o755)

# --- 3. the app ------------------------------------------------------------
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

app = FastAPI(title="AHC Kaggle GPU server")

# The namespace every /run call shares. This is the whole point: a model loaded
# by one call is still resident in VRAM for the next one.
NS: dict = {"__name__": "__remote__"}


def auth(authorization):
    if not authorization or authorization.removeprefix("Bearer ").strip() != TOKEN:
        raise HTTPException(401, "bad or missing bearer token")


@app.get("/health")
def health(authorization: str = Header(None)):
    auth(authorization)
    info = {
        "ok": True,
        "uptime_sec": round(time.time() - STATE["started"], 1),
        "tunnel_url": STATE["tunnel_url"],
        "python": sys.version.split()[0],
        # What is currently resident - i.e. what you do NOT have to reload.
        "ns_keys": sorted(k for k in NS if not k.startswith("__")),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpus"] = [
                {
                    "name": torch.cuda.get_device_name(i),
                    "total_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2),
                    "alloc_gb": round(torch.cuda.memory_allocated(i) / 1e9, 2),
                }
                for i in range(torch.cuda.device_count())
            ]
    except Exception as e:
        info["torch_error"] = str(e)
    return info


class RunReq(BaseModel):
    code: str
    mode: str = "exec"  # "exec" runs statements; "eval" returns the expression's repr


@app.post("/run")
def run(req: RunReq, authorization: str = Header(None)):
    """Run a snippet in the persistent namespace and return whatever it printed.

    This is the iteration loop: edit a file on the laptop, send it here, read the
    output - with the models still loaded from the previous call. Equivalent to
    typing into the notebook's own kernel, which is exactly what it is.
    """
    auth(authorization)
    out, err = io.StringIO(), io.StringIO()
    result, ok = None, True
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            if req.mode == "eval":
                result = repr(eval(req.code, NS))
            else:
                exec(req.code, NS)
    except Exception:
        ok = False
        err.write(traceback.format_exc())
    return {
        "ok": ok,
        "result": result,
        "stdout": out.getvalue(),
        "stderr": err.getvalue(),
        "elapsed_sec": round(time.time() - t0, 3),
    }


class ShellReq(BaseModel):
    cmd: str
    timeout: int = 600


@app.post("/shell")
def shell(req: ShellReq, authorization: str = Header(None)):
    """nvidia-smi, du, ls -la, pip install - the things you'd type in a `!` cell."""
    auth(authorization)
    p = subprocess.run(req.cmd, shell=True, capture_output=True, text=True, timeout=req.timeout)
    return {"returncode": p.returncode, "stdout": p.stdout[-40000:], "stderr": p.stderr[-40000:]}


@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    dest: str = Form("/kaggle/working"),
    authorization: str = Header(None),
):
    auth(authorization)
    os.makedirs(dest, exist_ok=True)
    path = os.path.join(dest, os.path.basename(file.filename))
    with open(path, "wb") as f:
        f.write(await file.read())
    return {"path": path, "bytes": os.path.getsize(path)}


@app.get("/download")
def download(path: str, authorization: str = Header(None)):
    auth(authorization)
    if not os.path.isfile(path):
        raise HTTPException(404, f"no such file: {path}")
    return FileResponse(path, filename=os.path.basename(path))


@app.get("/ls")
def ls(path: str = "/kaggle/working", authorization: str = Header(None)):
    auth(authorization)
    if not os.path.isdir(path):
        raise HTTPException(404, f"no such directory: {path}")
    entries = []
    for name in sorted(os.listdir(path)):
        p = os.path.join(path, name)
        entries.append(
            {
                "name": name,
                "dir": os.path.isdir(p),
                "bytes": os.path.getsize(p) if os.path.isfile(p) else None,
            }
        )
    return {"path": path, "entries": entries}


# --- 4. run ----------------------------------------------------------------
def _serve():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")


threading.Thread(target=_serve, daemon=True).start()
time.sleep(4)

proc = subprocess.Popen(
    [CF, "tunnel", "--url", f"http://localhost:{PORT}", "--no-autoupdate"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)

# cloudflared prints the hostname inside an ASCII box a few seconds in.
deadline = time.time() + 90
while time.time() < deadline and not STATE["tunnel_url"]:
    line = proc.stdout.readline()
    if not line:
        break
    m = re.search(r"https://[-\w]+\.trycloudflare\.com", line)
    if m:
        STATE["tunnel_url"] = m.group(0)

print("=" * 78)
if STATE["tunnel_url"]:
    print("  TUNNEL URL :", STATE["tunnel_url"])
    print()
    print("  Put this in the laptop's .env (token must match this session's):")
    print(f"    KAGGLE_REMOTE_URL={STATE['tunnel_url']}")
    print()
    print("  Then:  python scripts/remote.py health")
else:
    print("  cloudflared never reported a URL.")
    print("  Check Session options -> Internet is ON, then re-run this cell.")
print("=" * 78)
print("\nLeave this cell running - a busy kernel is what stops Kaggle idling out.\n")

# Keep the kernel occupied, and surface tunnel trouble if the connection drops.
while True:
    line = proc.stdout.readline()
    if not line:
        time.sleep(5)
        continue
    if any(k in line for k in ("ERR", "error", "Retrying", "failed")):
        print("[cloudflared]", line.rstrip())
