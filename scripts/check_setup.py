"""
Verify every piece of the hackathon setup. Run after filling in .env.

    python scripts/check_setup.py

Adapted from the 22-Aug repo's version, with the provider notes that were earned
the hard way there left intact, plus checks for what this problem statement needs
that the last one didn't: local weights, the dataset pack, and the remote GPU session.
"""
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

RESET, GREEN, RED, YELLOW, DIM = "\033[0m", "\033[32m", "\033[31m", "\033[33m", "\033[2m"


def ok(msg):
    print(f"{GREEN}[OK]{RESET} {msg}")


def fail(msg):
    print(f"{RED}[FAIL]{RESET} {msg}")


def warn(msg):
    print(f"{YELLOW}[WARN]{RESET} {msg}")


def run_with_timeout(func, seconds=20):
    """Belt-and-suspenders on top of each client's own timeout=.

    Guarantees no single unresponsive provider hangs the whole script, regardless
    of whether a given SDK's timeout argument works the way its docs claim - which
    is exactly what bit us with NVIDIA's 90B endpoint on 22-Aug.
    """
    t = threading.Thread(target=func, daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        fail(f"{func.__name__} timed out after {seconds}s (hung network call) - skipping")


def unset(val, *placeholders):
    return not val or val in ("", "...") or any(val.startswith(p) for p in placeholders)


# --- local toolchain --------------------------------------------------------
def check_ffmpeg():
    if not shutil.which("ffmpeg"):
        fail("ffmpeg not found on PATH")
        return
    out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    ok(f"ffmpeg: {out.stdout.splitlines()[0][:60]}")


def check_torch_cuda():
    try:
        import torch
    except ImportError:
        fail("torch not installed - run `uv sync`")
        return
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        ok(f"torch {torch.__version__} sees CUDA: {name} ({vram:.1f} GB)")
        if vram < 6:
            print(f"     {DIM}{vram:.0f}GB fits the CLIP/SigLIP filter stage and a 3B VLM in "
                  f"4-bit. Fine-tuning goes to Kaggle.{RESET}")
    else:
        warn(f"torch {torch.__version__} installed but CUDA unavailable - CPU only")


def check_rclone():
    exe = shutil.which("rclone")
    if not exe:
        pkgs = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
        hits = sorted(pkgs.glob("Rclone.Rclone*/**/rclone.exe")) if pkgs.exists() else []
        if not hits:
            warn("rclone not installed - needed for the authenticated dataset pull")
            return
        exe = str(hits[0])
    remotes = subprocess.run([exe, "listremotes"], capture_output=True, text=True).stdout
    if "gdrive:" in remotes:
        ok("rclone configured with a 'gdrive' remote")
    else:
        warn("rclone installed but no 'gdrive' remote - anonymous gdown is quota-blocked, "
             "so run: rclone config create gdrive drive scope=drive.readonly")


# --- data and weights -------------------------------------------------------
def check_dataset():
    data = Path(os.getenv("DATA_ROOT", ROOT / "data"))
    if not data.is_absolute():
        data = ROOT / data
    if not data.exists():
        warn(f"{data} does not exist - run scripts/download_dataset.py --rclone")
        return
    vids = list(data.rglob("*.mp4"))
    if not vids:
        warn(f"{data} exists but holds no mp4s yet")
        return
    gb = sum(p.stat().st_size for p in vids) / 1e9
    msg = f"dataset: {len(vids)} clips, {gb:.1f} GB"
    (ok if gb > 12 else warn)(msg + ("" if gb > 12 else "  (pack is ~15-17 GB - incomplete)"))


def check_weights():
    hf = Path(os.getenv("HF_HOME", ROOT / "models/hf"))
    if not hf.is_absolute():
        hf = ROOT / hf
    hub = hf / "hub"
    if not hub.exists():
        warn(f"no HF cache at {hub} - run scripts/download_models.py")
        return
    repos = sorted(p.name.removeprefix("models--").replace("--", "/")
                   for p in hub.glob("models--*"))
    gb = sum(f.stat().st_size for f in hub.rglob("*") if f.is_file()) / 1e9
    ok(f"weights cached ({gb:.1f} GB): {', '.join(repos) if repos else 'none'}")


def check_remote():
    url = os.getenv("KAGGLE_REMOTE_URL", "")
    token = os.getenv("KAGGLE_REMOTE_TOKEN", "")
    if not token:
        warn("KAGGLE_REMOTE_TOKEN unset - run `python scripts/remote.py token`")
        return
    if not url:
        warn("KAGGLE_REMOTE_URL unset - no Kaggle session running (expected when idle). "
             "Start kaggle/gpu_server/gpu_server.ipynb to get one.")
        return
    try:
        sys.path.insert(0, str(ROOT))
        from pipeline.kaggle_remote import Remote

        h = Remote().health()
        gpus = ", ".join(g["name"] for g in h.get("gpus", [])) or "no GPU"
        ok(f"Kaggle session up: {gpus}, uptime {h['uptime_sec']:.0f}s, "
           f"loaded: {h['ns_keys'] or 'nothing yet'}")
    except Exception as e:
        fail(f"Kaggle session unreachable: {str(e)[:160]}")


# --- hosted providers -------------------------------------------------------
def check_nvidia_nim():
    key = os.getenv("NVIDIA_API_KEY", "")
    if unset(key, "nvapi-..."):
        warn("NVIDIA_API_KEY not set - skipping NIM test")
        return
    try:
        from openai import OpenAI

        client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=key, timeout=15.0)
        # Deliberately NOT meta/llama-3.2-90b-vision-instruct (the prereqs guide's
        # example): on 22-Aug the key/network/catalog all checked out via raw curl,
        # but that endpoint hung forever - likely every team hitting the same model.
        # The 11B variant answered in <1s and suits bulk work better anyway.
        resp = client.chat.completions.create(
            model="meta/llama-3.2-11b-vision-instruct",
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=10,
        )
        ok(f"NVIDIA NIM: {resp.choices[0].message.content!r}")
    except Exception as e:
        fail(f"NVIDIA NIM failed: {str(e)[:200]}")


def check_gemini():
    key = os.getenv("GEMINI_API_KEY", "")
    if unset(key, "YOUR-KEY"):
        warn("GEMINI_API_KEY not set - skipping")
        return
    try:
        from google import genai

        client = genai.Client(api_key=key)
        resp = client.models.generate_content(
            model="gemini-2.5-flash", contents="Reply with exactly: OK"
        )
        ok(f"Gemini: {resp.text.strip()!r}  (teacher/pseudo-labeller, not a runtime component)")
    except Exception as e:
        fail(f"Gemini failed: {str(e)[:200]}")


def check_openrouter():
    key = os.getenv("OPENROUTER_API_KEY", "")
    if unset(key, "sk-or-v1-..."):
        warn("OPENROUTER_API_KEY not set - skipping")
        return
    try:
        from openai import OpenAI

        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key, timeout=15.0)
        resp = client.chat.completions.create(
            model="nvidia/nemotron-3.5-lightning:free",
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=10,
        )
        ok(f"OpenRouter: {resp.choices[0].message.content!r}")
    except Exception as e:
        fail(f"OpenRouter failed: {str(e)[:200]}\n"
             f"       free-model IDs rotate - check openrouter.ai/models?max_price=0")


def check_groq():
    key = os.getenv("GROQ_API_KEY", "")
    if unset(key, "gsk_..."):
        warn("GROQ_API_KEY not set - skipping")
        return
    try:
        from openai import OpenAI

        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=key, timeout=15.0)
        resp = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=10,
        )
        ok(f"Groq: {resp.choices[0].message.content!r}")
    except Exception as e:
        fail(f"Groq failed: {str(e)[:200]}")


if __name__ == "__main__":
    print(f"=== AHC Visual Intelligence Hackathon - setup check ===\n")
    print(f"Python: {sys.version.split()[0]}\n")

    print("--- local toolchain ---")
    check_ffmpeg()
    check_torch_cuda()
    check_rclone()

    print("\n--- data and weights ---")
    check_dataset()
    check_weights()

    print("\n--- remote GPU ---")
    check_remote()

    print("\n--- hosted providers (development only: the PS bars them from runtime) ---")
    run_with_timeout(check_nvidia_nim)
    run_with_timeout(check_gemini)
    run_with_timeout(check_openrouter)
    run_with_timeout(check_groq)
