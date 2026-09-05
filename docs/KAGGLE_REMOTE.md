# Using Kaggle's GPU without the cold-start tax

## The problem this solves

`kaggle kernels push` **always executes the whole notebook, top to bottom, in a
fresh container.** There is no "update the source without running it" mode. On the
22-Aug hackathon that made every one-line fix cost 5–10 minutes of re-running pip
installs and multi-GB weight downloads, and it was the single biggest time sink of
the day. (Full autopsy of that workflow, including every bug hit, is in
`docs/KAGGLE.md`.)

**So we don't push from here.** Nothing in this repo runs or pushes a Kaggle
notebook. The session runs in the browser; the laptop talks to it over HTTP.

## How it works

A single cell runs in an **interactive** Kaggle session and:

1. installs `fastapi`/`uvicorn` (torch and transformers are already in the image),
2. downloads `cloudflared` and opens a **quick tunnel** — a random
   `*.trycloudflare.com` URL, no Cloudflare account and no domain needed,
3. serves a small API on that URL, guarded by a bearer token,
4. never returns, because a busy kernel is what stops Kaggle idling the session out.

The important part is `/run`: it executes code in a **persistent namespace**. A
model loaded by one call is still resident in VRAM for the next. That is the whole
point — installs and weight downloads are paid for once per session, then we
iterate from the laptop at the speed of an HTTP round trip.

It is a remote Jupyter kernel by another name: your own session, your own token,
just reachable from the terminal instead of only from the browser tab.

## Getting the dataset in — Kaggle has no `drive.mount()`

That function is Colab-only; Kaggle notebooks have no native Drive mount. The
equivalent here is better anyway, because it's permanent:

1. **Download Drive → Kaggle once**, inside the session
   (`kaggle/gpu_server/fetch_dataset_cell.py`, cell 1 of the notebook).
2. **Save it as a Kaggle Dataset** (`File → Save Version`, or
   `kaggle datasets create -p /kaggle/working/data`).
3. From then on, `Add data → Your Datasets` mounts it read-only at
   `/kaggle/input/<slug>/` in **every** future session, instantly.

Downloaded exactly once, ever — versus Colab, where a mount is re-established
each session and the bytes still cross the network on every read.

This is also why the training pack never needs to be on the laptop: the
fine-tuning runs here, Kaggle's egress is far faster than a home connection, and
a 15 GB round trip down-then-up buys nothing. Locally, pull only the 34-video
public test set (`python scripts/download_dataset.py --test-only`) for the
dashboard and the scoring pipeline.

### The mirrors are quota-blocked

All five refuse anonymous downloads with *"Too many users have viewed or
downloaded this file recently."* `gdown` enumerates all 16,170 files fine and then
fails on the first byte — confirmed for the full pack **and** for the test set
alone, so it is the per-file quota, not a size limit. It is global to the file,
not per-IP, so retrying from Kaggle is not automatically a fix (the fetch cell
tries anyway, because it costs two minutes and the 24h lockout may have aged out).

The route that works needs one browser sign-in:

```bash
python scripts/rclone_token.py     # on the laptop; opens Google's consent screen
```

Paste the JSON it prints into Kaggle under **Add-ons → Secrets** as `GDRIVE_TOKEN`.
The fetch cell then authenticates as you instead of anonymously.

If Drive still refuses, copy a mirror into your own Drive in the browser and point
the fetch cell at your copy. Copying is server-side — no bytes move — and a file
you own carries no "too many users" quota. Reports differ on whether Drive allows
copying a large binary that is *already* quota-locked; if it balks, "Add shortcut
to Drive" or asking the organisers for a Kaggle Dataset mirror is the fallback.

## Setup

**Once, on the laptop** — a token already exists in `.env`; to mint a new one:

```bash
python scripts/remote.py token
```

**Each session, on Kaggle:**

1. `python scripts/build_kaggle_notebook.py --clip` copies the cell to the clipboard
   (and regenerates `kaggle/gpu_server/gpu_server.ipynb` if you'd rather import a file).
2. New notebook → **Session options → Accelerator: GPU T4 x2**, **Internet: On**.
3. **Add-ons → Secrets** → add `KAGGLE_REMOTE_TOKEN` with the same value as `.env`.
4. Paste the cell, run it. It prints:

   ```
   TUNNEL URL : https://something-random.trycloudflare.com
   ```

5. Put that in `.env` as `KAGGLE_REMOTE_URL`. The URL changes every session; the
   token does not.

## Use

```bash
python scripts/remote.py health                    # GPU, uptime, what's loaded
python scripts/remote.py shell "nvidia-smi"
python scripts/remote.py run "import torch; print(torch.cuda.get_device_name(0))"
python scripts/remote.py runfile pipeline/stage1_filter.py
python scripts/remote.py push data/test/videos/T001.mp4 /kaggle/working/clips
python scripts/remote.py pull /kaggle/working/scores.parquet runs/
```

Or from Python, which is where it actually earns its keep:

```python
from pipeline.kaggle_remote import Remote
r = Remote()

r.run("""
from transformers import AutoModel
model = AutoModel.from_pretrained("google/siglip2-base-patch16-224").cuda().eval()
print("loaded")
""")                              # ~60s, once

r.run("print(model.config.hidden_size)")   # instant - model is still in VRAM
r.run_file("pipeline/stage1_filter.py")    # edit locally, run there, state intact
```

`health()` lists `ns_keys` — everything currently resident, i.e. everything you do
*not* have to reload.

## Things that will bite

- **Internet: On is not the default.** Without it `cloudflared` never gets a URL
  and the cell reports so.
- **The URL dies with the session.** Re-run the cell, paste the new URL. A stale
  `KAGGLE_REMOTE_URL` shows up as a connection error, not a 401.
- **A 401 means token drift** between `.env` and Kaggle Secrets, not a dead session.
- **Interactive sessions run 9–12 hours** and idle out if untouched. The blocking
  cell handles the idle part; the hard cap it cannot.
- **Quick tunnels cap at 200 concurrent requests.** Irrelevant for one laptop.
- **`/kaggle/working` is what persists** as the session's Output. Anything
  elsewhere dies with the container — `pull` what you need before you stop.
- **Only the free-tier weekly budget is real:** 30 GPU-hours/week. A session left
  running with a blocking cell is burning them, so stop it when you break for lunch.

## Alternative, if the tunnel is blocked

The venue's network may block `trycloudflare.com`. Fallbacks, in order:

1. **ngrok** — swap the `cloudflared` block for `pyngrok`; needs a free authtoken
   in Kaggle Secrets. Best-documented option for Kaggle specifically.
2. **Claude in Chrome** on the Kaggle editor tab — paste code into cells directly,
   no tunnel at all. Slower per iteration but nothing to block.
3. **Colab** with the same cell — identical mechanism, different host.
