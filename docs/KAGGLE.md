# Kaggle: setup, mental model, and every bug we hit

Written as a standalone reference — assumes no memory of the session that produced it.
If you're picking this project back up cold, read this before touching Kaggle again.

## 1. Account & auth

- Free tier: **30 GPU-hours/week**, T4 x2 available per session.
- This CLI version (2.2.4) uses a **single-token** auth scheme — **not** the classic
  `kaggle.json` `{username, key}` pair from older tutorials/docs. Don't be misled by those.
- Token (prefixed `KGAT_`) goes in `~/.kaggle/access_token`, or export `KAGGLE_API_TOKEN`.
  `KAGGLE_USERNAME` isn't required for auth but is handy to have on file. Both are recorded
  in this repo's `.env` (gitignored).

## 2. The mental model that explains every surprise below

**A Kaggle kernel push always executes the whole notebook, top to bottom, in a fresh
container.** There is no "just update the source, don't run it" mode via the API — pushing
*is* running. This single fact explains most of what follows:

- **No live sync exists** between local files and Kaggle, in either direction. Not via API,
  not via the interactive browser editor. The only mechanisms:
  - **Local → Kaggle**: `kaggle kernels push` (always runs it), or manually paste code into
    the browser editor, or use the editor's *File → Import Notebook → GitHub* (pulls a
    GitHub file's content into the editor — a manual, one-shot click, not automatic, and
    doesn't run anything by itself).
  - **Kaggle → Local**: *File → Download → .ipynb*, or *File → Link to GitHub* (manual export).
- **Two fundamentally different session types, don't confuse them:**
  - *Interactive* (opened in the browser, "Session options → Accelerator → GPU T4 x2"):
    stays alive for 9–12 hours, cell state and loaded models persist between cell re-runs,
    but has an **idle-disconnect timeout** if left untouched too long.
    You pick the GPU explicitly from a dropdown — no ambiguity.
  - *API-pushed* (`kaggle kernels push`): a one-shot batch job. Runs everything cold,
    then **the container is destroyed on completion regardless of pass/fail** — no idle-timeout
    risk, but also nothing persists afterward except the deliberately-saved Output (see §5).
    GPU type is **not** chosen interactively — see `machine_shape` in §4, or you may
    silently get the wrong one.

**Practical workflow that follows from this:** do actual iterative pipeline-building in an
*interactive* session (fix one cell, re-run just that cell, seconds not minutes). Reserve
`kaggle kernels push` for occasional "does this still work from a clean container" checks —
e.g. once before a demo — not as the main edit loop. Every push-based debug cycle here cost
~5–10 minutes because the *entire* notebook (pip installs + model downloads) reruns from
scratch every time, even to test a one-line fix.

## 3. CLI reference

```bash
kaggle kernels push -p kaggle/                       # push + execute (see §2)
kaggle kernels status <username>/<kernel-slug>        # poll: RUNNING / COMPLETE / ERROR
kaggle kernels logs <username>/<kernel-slug>           # fast, text-only stdout/stderr
kaggle kernels output <username>/<kernel-slug> -p DIR  # pulls ALL output files - can be
                                                        # multi-GB (model caches). Slow.
                                                        # Use `logs` for debugging; `output`
                                                        # only when you actually need the data.
```

**Windows-specific gotchas:**
- `python`/`pip` on PATH may resolve to a WindowsApps stub instead of the real install if a
  shell session started before Python was added to PATH, or if a store-alias shadows it.
  Use the `py` launcher (`py -m kaggle ...`) or the full interpreter path if `python`/`pip`
  silently fail.
- `kaggle kernels logs` can crash with a charmap codec error on Unicode log content. Set
  both env vars first: `$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8="1"` (PowerShell) or
  `export PYTHONIOENCODING=utf-8 PYTHONUTF8=1` (bash).

## 4. `kernel-metadata.json` — the field that actually matters: `machine_shape`

```json
{
  "id": "prithvirajgotepatil/flytbase-hackathon-starter",
  "title": "FlytBase Hackathon Starter",
  "code_file": "kaggle_starter.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "machine_shape": "NvidiaTeslaT4",
  "enable_internet": true,
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": []
}
```

- **`machine_shape`** controls which GPU an API-pushed kernel actually gets. Without it set
  explicitly, our first push was silently assigned a **Tesla P100** instead of a T4 — see
  bug #1 in §6. Valid values (found by grepping the installed `kagglesdk` package source
  directly, since this isn't obviously documented): `NvidiaTeslaT4`, `NvidiaTeslaP100`,
  `Tpu1VmV38`.
- **`kernel_sources`**: lets you mount a *different* kernel's (or **this same kernel's own
  past**) saved Output as a read-only input at `/kaggle/input/<kernel-slug>/` in a new run.
  This is the mechanism for cross-run caching — see §5.

## 5. What persists across runs, and what doesn't

- **`/kaggle/working/`** is the working directory during a run. Whatever's in it when the
  run ends becomes that kernel version's **Output** — retained on Kaggle's servers
  indefinitely, even after the container itself is destroyed. Viewable/downloadable from the
  kernel's "Output" tab in the browser, or via `kaggle kernels output`.
- **The live container is still gone.** "Output persists" ≠ "the session is still running."
  A finished API-pushed run's GPU/RAM/loaded-models are gone the moment it completes; only
  the frozen files survive.
- **To actually reuse cached weights on a later run** (skip re-downloading multi-GB model
  files): add `"kernel_sources": ["<username>/<this-kernel-slug>"]` (self-reference) to
  `kernel-metadata.json`, then in the notebook check for the mounted path before downloading:
  ```python
  local_cache = "/kaggle/input/<kernel-slug>/model_cache"
  cache_dir = local_cache if os.path.exists(local_cache) else "/kaggle/working/model_cache"
  # from_pretrained(..., cache_dir=cache_dir)
  ```
  This does **not** speed up pip installs (esp. `transformers` built from source) — only
  model weight downloads. Confirmed via direct check: after our v6 run, its Output genuinely
  contained the full HF cache tree for both SigLIP2 and Qwen3-VL-4B-Instruct (~4.5GB+), so
  the mechanism is real and available — just not wired in, since the interactive-session
  workflow (§2) makes it lower priority than it first seemed.

## 6. Every bug found, in the order we hit them

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | `CUDA error: no kernel image is available for execution on the device` deep inside a forward pass | Kaggle assigned a **Tesla P100** (compute capability sm_60) — below what modern `torch` builds support (sm_70+). Looks identical to a torch-version mismatch bug but isn't one. | Set `"machine_shape": "NvidiaTeslaT4"` explicitly in `kernel-metadata.json` (§4). |
| 2 | `ImportError` on `tokenizers` at first use of transformers-from-source | `--no-deps` on the transformers install blocked pip from resolving `tokenizers>=0.23.1,<0.24.0`, which transformers-main actually needs, leaving the older pinned version in place | Drop `--no-deps`. Instead, capture `torch`/`torchvision`'s *already-correct* versions right after the platform preinstalls them, and pass those as explicit pins on every subsequent install (`pip install ... "torch==X" "torchvision==Y"`) so only those two are locked — everything else (tokenizers, etc.) resolves normally. |
| 3 | `AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'shape'` | This transformers-main build changed `get_image_features()` to return a wrapped output object instead of a bare tensor — a bleeding-edge API surface change | Handle both shapes defensively: `embedding = getattr(output, "pooler_output", output)` |
| 4 | `CUDA out of memory` loading the second model (Qwen3-VL), GPU 0 | The first model (SigLIP2) was still resident on GPU 0; `device_map="auto"` didn't spill the second model onto the empty second GPU the way you'd expect | Pin devices explicitly (`.to("cuda:0")`, `qwen_device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"`) instead of relying on `"auto"`. Explicitly `del` + `gc.collect()` + `torch.cuda.empty_cache()` after each model's use, before the next one loads. |
| 5 | `CUDA out of memory` again, now on GPU 1, a dedicated empty GPU | Confirmed the device-pinning fix worked (correctly used GPU 1) — but the model genuinely doesn't fit: full-precision 4B weights (~8GB) + one vision-tower attention call spiking 7+GB doesn't fit in a T4's ~14.5GB, even alone | **4-bit quantization** (`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)`, passed as `quantization_config=` at `from_pretrained(..., device_map=qwen_device)` — not `.to()` afterward, which quantized models don't support) shrinks weights to ~2GB. **Downscaling the input image** (`PIL.Image.thumbnail((512,512))` before passing it into the chat template, as a PIL object not a URL) caps the vision tower's attention sequence length, capping that spike too. |
| 6 | (Same category, generally) `pip install -U` on a foundational package can silently upgrade `torch`/`pandas` past what the platform's own preinstalled stack (RAPIDS cuDF, etc.) depends on — breaks *other* things, not the package you're upgrading, and can fail at import time or deep in a forward pass with no warning at install time | Never `-U` a foundational package on a platform image. Capture the platform's already-correct versions and pin them explicitly (see #2) rather than letting pip's resolver touch them. |

**End state, fully verified (not assumed):** after fix #5, a full `kaggle kernels push`
reached `KernelWorkerStatus.COMPLETE`, and the actual pulled log output showed both real
GPUs detected, correct pinned torch/torchvision versions, all installs clean, `SigLIP2
loaded OK. Embedding shape: torch.Size([1, 1152])`, and a genuine, coherent Qwen3-VL-4B
caption of the real demo image. Two harmless non-fatal `[ERROR]` lines about undocumented
`min_frames`/`max_frames` kwargs appeared in that same log but didn't affect the outcome —
they're transformers' internal docstring-validation noise, not a runtime failure.

## 7. Output is only ever retrievable for the *current* version — pull before you push again

`kaggle kernels output <owner>/<kernel>` only ever serves the most recently pushed version's
output. Passing a version suffix (`<owner>/<kernel>/<version>`) does **not** retrieve that
historical version's output - confirmed by checking the actual file contents after requesting
an old version's output and getting the current version's data back silently, mislabeled as
if it were the old one. Once you push a new version to the same kernel, the previous run's
output is gone from the CLI's reach - if you need multiple runs' outputs, **pull each one
before pushing the next**, or use separate kernel IDs per run you want to keep.

## 8. Files in this repo

- `kaggle/kernel-metadata.json` — push config (§4)
- `kaggle/kaggle_starter.ipynb` — the smoke-test notebook itself (GPU check → dep pinning →
  core installs → transformers-from-source → SigLIP2 smoke test → Qwen3-VL smoke test)
- `kaggle/output/` — gitignored, transient scratch space for manual `kernels output` pulls;
  not source, safe to delete anytime
