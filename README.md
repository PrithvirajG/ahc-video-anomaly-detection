# AHC Visual Intelligence Hackathon — 05 Sep 2026

Near-real-time video anomaly detection over drone / CCTV / dashcam footage using
small vision-language models.

- **What's being asked:** `docs/PROBLEM_STATEMENT.md`
- **What the literature says to do:** `docs/RESEARCH_NOTES.md`
- **The pipeline, one notebook, runs entirely on Kaggle:** `kaggle/ahc_pipeline.ipynb`
- **How we use Kaggle's GPU without the cold-start tax (optional):** `docs/KAGGLE_REMOTE.md`
- **Kaggle reference from the 22-Aug hackathon (every bug hit):** `docs/KAGGLE.md`

## Quick start

```bash
uv sync                            # Python env (.venv), torch is CUDA-12.8
python scripts/check_setup.py      # verifies toolchain, weights, data, API keys

cd web && npm install && npm run dev    # dashboard on :5173
uv run uvicorn pipeline.api:app --reload --port 8010 # detector service on :8010
```

## Status

| Piece | State |
|---|---|
| Python env (uv, py3.11, torch 2.11+cu128) | ready — CUDA live on the GTX 1650 |
| Node dashboard (Vite + React 19 + TS) | ready — builds clean, wired to the API |
| Weights: SigLIP2, CLIP-B/16, PE-Core-L14-336, Qwen2.5-VL-3B | ready — 12 GB in `models/hf` |
| Papers (4 primer + 5 carried over) | ready — `papers/`, notes in `docs/RESEARCH_NOTES.md` |
| Remote Kaggle GPU session | ready — needs a session started to get a URL |
| **Dataset (16.0 GB, 3,207 clips)** | **local** — `data/`, all 12 classes + 34 test clips |
| Detector | stub; `pipeline/api.py` streams synthetic alerts so the UI is verified end to end |

### The dataset

On disk at `data/` — **3,207 mp4, 16.00 GB**, all twelve train class folders
correctly named, plus the 34-video public test set with ground truth.

All five Drive mirrors refuse **anonymous** downloads (*"Too many users have
viewed or downloaded this file recently"*), on distinct file IDs, on every
mirror. That quota does not apply to a signed-in browser: downloading a mirror
in Chrome gives 8 independent ~2 GB zips (Drive splits large folder downloads;
they are separate archives, not a spanned set, each rooted at `Train and Test/`).

```bash
python scripts/extract_dataset.py        # 8 zips -> data/train + data/test, ~6.5 min
python scripts/download_dataset.py --verify
```

#### Getting it onto Kaggle

**Kaggle's "link a Google Drive URL" importer does not work for folders.**
Pointed at a mirror it produced a 361 KB dataset holding one file named after
the folder ID whose content is the Drive *web page HTML*. It does a plain
unauthenticated HTTP GET and saves whatever comes back; only direct single-file
URLs work.

Two routes that do work:

1. **rclone Drive → Kaggle inside a session** — no home upload at all. Run
   `python scripts/rclone_token.py` once, paste the JSON into Kaggle under
   **Add-ons → Secrets** as `GDRIVE_TOKEN`. Route B of cell 2 in
   `kaggle/ahc_pipeline.ipynb` already implements this.
2. **Upload the local copy** — `kaggle datasets create`, 16 GB from home.

Either way, finish with **File → Save Version** so the result mounts read-only
at `/kaggle/input/<slug>/` in every future session — downloaded exactly once.

## Layout

```
data/            the train/test pack (gitignored)
docs/            problem statement, research synthesis, Kaggle workflow
kaggle/ahc_pipeline.ipynb  THE notebook - cascade, eval, submission
kaggle/cells/    its cell sources (edit these, then build_notebook.py)
kaggle/gpu_server/  optional: turns a session into a GPU service over a tunnel
models/hf/       HF weight cache (gitignored)
papers/          PDFs + extracted text
pipeline/        api.py (detector service), kaggle_remote.py (remote client)
scripts/         setup check, dataset/model downloads, remote CLI
web/             Vite + React dashboard
```

## Scripts

| Command | Does |
|---|---|
| `python scripts/check_setup.py` | end-to-end preflight |
| `python scripts/extract_dataset.py` | 8 Drive-split zips -> `data/train` + `data/test` |
| `python scripts/build_notebook.py` | rebuild `kaggle/ahc_pipeline.ipynb` from `kaggle/cells/*.py` |
| `python scripts/build_notebook.py --check` | syntax-check the cells without writing the notebook |
| `python scripts/download_dataset.py --rclone` | authenticated full pull (no longer needed locally) |
| `python scripts/rclone_token.py` | mint a Drive token for Kaggle Secrets |
| `python scripts/download_dataset.py --verify` | audit `data/` against the expected layout |
| `python scripts/download_models.py --list` | show weight sets before pulling |
| `python scripts/build_kaggle_notebook.py --clip fetch` | regenerate cells, copy the data-fetch one to clipboard |
| `python scripts/remote.py health` | talk to the live Kaggle session |

## Two constraints worth re-reading before designing anything

1. **Hosted models cannot be part of the runtime detector.** Gemini/NIM/OpenRouter
   are for development, comparison and generating training data only. An
   architecture that calls a hosted VLM per frame does not satisfy the brief.
2. **False alarms count as much as misses.** The PS is explicit that an alerting
   system firing on ordinary activity stops being used.

And one measured fact that shapes local work: **fp32 is 3× faster than fp16 on this
GPU** (33 vs 11 frames/s for SigLIP2-base). The GTX 1650 is TU117 — compute
capability 7.5 but with the tensor cores fused off, so half precision has no fast
path. Don't default to fp16 locally; do use it on Kaggle's T4s.
