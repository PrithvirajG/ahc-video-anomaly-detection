"""
Client for the persistent Kaggle GPU server (kaggle/gpu_server/kaggle_gpu_server.py).

The problem this solves: `kaggle kernels push` re-executes a notebook from a cold
container every time, so every one-line fix re-pays pip installs and multi-GB
weight downloads. Here the Kaggle session stays up with the models resident, and
we drive it from the laptop - the expensive setup happens once.

    from pipeline.kaggle_remote import Remote
    r = Remote()                       # reads KAGGLE_REMOTE_URL / _TOKEN from .env
    r.health()
    r.run("import torch; print(torch.cuda.get_device_name(0))")
    r.run_file("pipeline/stage1_filter.py")   # send a local file, keep the session's state
    r.push("data/test/videos/T001.mp4", "/kaggle/working/clips")
    r.pull("/kaggle/working/scores.parquet", "runs/")
"""
from __future__ import annotations

import os
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


class RemoteError(RuntimeError):
    pass


class Remote:
    def __init__(self, url: str | None = None, token: str | None = None, timeout: int = 900):
        self.url = (url or os.getenv("KAGGLE_REMOTE_URL", "")).rstrip("/")
        self.token = token or os.getenv("KAGGLE_REMOTE_TOKEN", "")
        self.timeout = timeout
        if not self.url:
            raise RemoteError(
                "KAGGLE_REMOTE_URL is unset. Start the server cell in a Kaggle "
                "interactive session; it prints the trycloudflare URL to paste into .env."
            )
        if not self.token:
            raise RemoteError("KAGGLE_REMOTE_TOKEN is unset (must match the Kaggle session's).")

    @property
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"}

    def _check(self, resp: requests.Response) -> dict:
        if resp.status_code == 401:
            raise RemoteError("401 - token mismatch between .env and the Kaggle session.")
        if resp.status_code >= 400:
            raise RemoteError(f"{resp.status_code}: {resp.text[:500]}")
        return resp.json()

    # --- introspection ------------------------------------------------------
    def health(self) -> dict:
        return self._check(requests.get(f"{self.url}/health", headers=self._headers, timeout=30))

    def ls(self, path: str = "/kaggle/working") -> dict:
        return self._check(
            requests.get(f"{self.url}/ls", params={"path": path},
                         headers=self._headers, timeout=60)
        )

    # --- execution ----------------------------------------------------------
    def run(self, code: str, mode: str = "exec", raise_on_error: bool = True) -> dict:
        """Execute in the session's persistent namespace. Models stay loaded between calls."""
        r = self._check(
            requests.post(f"{self.url}/run", json={"code": code, "mode": mode},
                          headers=self._headers, timeout=self.timeout)
        )
        if raise_on_error and not r["ok"]:
            raise RemoteError(r["stderr"])
        return r

    def run_file(self, path: str | Path, **kw) -> dict:
        """Send a local file's source. This is the edit-here / run-there loop."""
        return self.run(Path(path).read_text(encoding="utf-8"), **kw)

    def eval(self, expr: str):
        return self.run(expr, mode="eval")["result"]

    def shell(self, cmd: str, timeout: int = 600) -> dict:
        return self._check(
            requests.post(f"{self.url}/shell", json={"cmd": cmd, "timeout": timeout},
                          headers=self._headers, timeout=timeout + 30)
        )

    # --- file transfer ------------------------------------------------------
    def push(self, local: str | Path, dest: str = "/kaggle/working") -> dict:
        local = Path(local)
        with open(local, "rb") as fh:
            return self._check(
                requests.post(
                    f"{self.url}/upload",
                    files={"file": (local.name, fh)},
                    data={"dest": dest},
                    headers=self._headers,
                    timeout=self.timeout,
                )
            )

    def pull(self, remote_path: str, local_dir: str | Path = ".") -> Path:
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        out = local_dir / Path(remote_path).name
        with requests.get(
            f"{self.url}/download",
            params={"path": remote_path},
            headers=self._headers,
            stream=True,
            timeout=self.timeout,
        ) as resp:
            if resp.status_code >= 400:
                raise RemoteError(f"{resp.status_code}: {resp.text[:500]}")
            with open(out, "wb") as f:
                for chunk in resp.iter_content(1 << 20):
                    f.write(chunk)
        return out
