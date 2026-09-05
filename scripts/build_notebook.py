"""
Assemble kaggle/ahc_pipeline.ipynb from the cell sources in kaggle/cells/.

The .py files are the source of truth: .ipynb diffs are unreadable in review,
and these cells get edited in the Kaggle browser editor as often as here. Keep
them the thing you edit, re-run this, and the notebook follows.

    python scripts/build_notebook.py
    python scripts/build_notebook.py --check     # syntax-check the cells only

Nothing here pushes or executes anything on Kaggle - the notebook is uploaded by
hand, once, and then lives in the browser session.
"""
from __future__ import annotations

import argparse
import ast
import re
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / "kaggle" / "cells"
OUT = ROOT / "kaggle" / "ahc_pipeline.ipynb"

# (cell file, markdown that precedes it)
PLAN: list[tuple[str, str]] = [
    ("c01_starter.py", """## 1 — What's attached

Kaggle's starter cell. It lists everything under `/kaggle/input`, capped at 20
lines because the full pack is 3,200+ files and the stock loop prints one line
each."""),

    ("c02_datacheck.py", """## 2 — Is the dataset actually there?

**Verification only — nothing here downloads anything.** The pack is attached as
a Kaggle Dataset and mounts read-only.

This cell exists because "the dataset is attached" and "the dataset has videos in
it" are different claims, and the gap between them is silent. It resolves the
mount (including the `datasets/<owner>/<slug>/` shape kagglehub uses), checks all
twelve class folders, the test set and the ground truth, and sets `DATA_OK`.

It also recognises one specific failure by signature: Kaggle's *link a Google
Drive URL* importer cannot authenticate and cannot walk a folder, so pointed at
one it stores a ~360 KB file containing the Drive **web page HTML**. That mounts
happily and holds no data."""),

    ("c03_config.py", """## 3 — Libraries, config, index

Every knob lives in `CFG`, so nothing below carries a magic number. Re-run this
cell after changing one; nothing downstream caches config. It also builds the
ground-truth index (`GT_TRAIN`, `GT_TEST`, `VIDEO_PATHS`) that every later cell
reads.

`USE_FP16` is decided from the GPU rather than assumed. On Kaggle's T4 (sm_75,
real tensor cores) fp16 is right. On a GTX 1650 it is a 3× *slowdown* — TU117
reports capability 7.5 but has the tensor cores fused off, so half precision
falls back to a slow path."""),

    ("c04_sampling.py", """## 4 — Stage 0: sampling and the motion gate

One frame-differencing pass does two jobs, as in Cerberus: decide whether a
frame is worth encoding, and locate the moving region so a red circle can be
drawn on it as a visual prompt.

**The keepalive is ours, not the paper's, and it matters.** Cerberus gates on
motion because its anomalies are motion events. Three of our twelve labels are
not: `waterlogging_or_flood` and `road_spill_or_debris` are static conditions,
and `stalled_or_broken_down_vehicle` is *defined* by the absence of motion. A
pure motion gate discards exactly their evidence, so one frame is forced through
every `static_keepalive_sec` regardless of score."""),

    ("c05_stage1.py", """## 5 — Stage 1: rule-deviation scoring

`health(x) = Σ_{c ∈ topk(x)} w_c · sim(x, c)`, with `w = +1` for normal rules and
`w = −1` for perturbed action labels. Escalate when health is low.

Two reasons this beats the obvious approach of listing anomalies and matching
against them. First, Cerberus measured that: prompting an LLM for possible
anomalies gave **27.13% recall on ShanghaiTech, 21.81% on NWPU** — enumeration
misses most of what happens. Second, Alert-CLIP shows CLIP's normal and abnormal
text embeddings are *entangled*, so asking it to compare "a normal street"
against "an anomalous street" is measurably unreliable. Here we never ask that
question; we ask which concrete descriptions the frame is nearest and let the
signs do the work.

The threshold is **calibrated on known-normal training footage**, not guessed,
which turns `escalate_pct` into a compute budget you can reason about."""),

    ("c05b_probe.py", """## 5b — A learned prior, because the written one is inverted

The health score above is the foundation under escalation, clustering and
extent measurement. Measured against ground truth, it is **flat or backwards**
on three of four test videos — anomalous frames sit at the *65th percentile* of
their own video's health. A congested road looks like a road; a loiterer looks
like a person.

So this cell spends the **3,173 labelled training clips** we have so far used to
compute exactly one number. Each clip is embedded once — 8 frames spanning 16s,
centred on the labelled event — and a plain multinomial logistic regression is
fitted over the frozen SigLIP2 vectors. The encoder stays frozen and zero-shot:
no backprop, no fine-tuning, no GPU for the fit. Ninety-three English sentences
are replaced by coefficients learned from examples.

**8 frames over 16s, not 4 over 2s**, on both sides. Our inference window is ~2s
while the median real event is 20s, so training wide and predicting narrow would
rebuild the exact mismatch this is meant to remove.

Embeddings are cached (`WORK`, then any attached dataset), because the expensive
step is pixels→vectors and the step worth iterating on is the classifier over
them. Nothing downstream consumes the probe yet — that decision waits on the
held-out numbers printed here."""),

    ("c06_stage2.py", """## 6 — Stage 2: the small VLM

Runs only on what stage 1 could not clear. **Qwen3-VL-4B**, not the 3B this
started as — that was sized for a 4GB laptop GPU and left most of a T4's 16GB
unused. Qwen3-VL adds video-specific architecture (interleaved MRoPE, textual
timestamps, temporally dense captions) Qwen2.5-VL lacks. It's also independently
validated for this exact task: QVAD (arXiv:2604.03040), a VAD paper in the
organizers' own SOTA deck, uses Qwen3-VL-4B-Instruct for captioning, and 2 of
the top 3 accepted-paper teams on the AI City Challenge 2026 traffic-anomaly
leaderboard ran Qwen3-VL-8B.

Two things keep it cheap and honest beyond the model swap:

**Shortlisting** — stage 1's embedding already ranks the eleven anomaly labels,
so we ask about the top three. Shorter prompt, and the model is not invited to
hallucinate through eight irrelevant options.

**ASK-Hint prompting** — every label expands into concrete, observable
questions. "Is there any anomaly?" misses what "Do you see punching, kicking, or
wrestling on the ground?" catches on the same input. This is a text file rather
than a training run, which makes it the best accuracy-per-minute available."""),

    ("c07_aggregate.py", """## 7 — Per-class temporal aggregation

The PS's own table says an accident is over in ~1s, congestion builds gradually,
a stalled vehicle is anomalous only after persisting, and waterlogging is a
static condition. That is four different aggregators, not one — a min-duration
long enough to stop congestion flickering erases every accident.

So persistence is per class, and events open/close with **hysteresis**. That is
where false-alarm suppression lives: one confident window opens nothing, while a
real event survives a brief occlusion instead of fragmenting into five alerts."""),

    ("c08_pipeline.py", """## 8 — End to end

`process_video()` runs stage 0 → 1 → 2 → aggregation and reports a realtime
factor per video.

`LIMIT = 3` deliberately. Prove the wiring on three videos before spending
GPU-hours; raise it to `None` for the full 34-video public test set."""),

    ("c09_evaluate.py", """## 9 — Score against the local public test set

Diagnostics only, against the T00x videos we can actually see ground truth
for. The **false-alarm rate gets its own line** rather than being buried in
accuracy — a model that wins on F1 by flagging everything has failed the
actual brief. Level 2/3 temporal scoring now uses the arena's real gate
(**IoU ≥ 0.5**, correct class, at most one predicted event may match — extra
overlapping fragments count *against* you), not a loose diagnostic threshold."""),

    ("c10_submission.py", """## 10 — The arena submission file

A different, stricter schema than anything above: JSON, not CSV; scored
against a **private** video set (`E001, E002, …`) from `manifest.json`,
downloaded from the arena's Benchmark tab — not the local `T00x` test set.
Drop the fetched manifest into `/kaggle/working/manifest.json`; until then this
cell uses the local test set's own ground truth as a stand-in so it's testable
now.

Two silent-rejection traps this builds around: a normal video is `"events": []`
— never `"class_name": "normal"` — and Level-1 timestamps must be `null`, not
omitted. And two scoring rules the *aggregation* step (cell 7) already exists
to satisfy: a false alarm on a truly-normal Level-2/3 video scores that video
**zero**, and fragmenting one real event into several predictions only lets the
best-overlapping one match — the rest count against you."""),

    ("c11_visualize.py", """## 11 — See it, don't just read the JSON

A grid of real frames: one per detected event (predicted class + confidence,
green border if the class matches ground truth, red if it doesn't), plus a
few genuinely missed anomalies for honest contrast. Doubles as the example
frames the architecture write-up and 2-slide PPT are asked to include —
"prefer visuals over long text.\""""),
]

HEADER = """# AHC Visual Intelligence Hackathon — detection pipeline

Real-time video anomaly detection over drone / CCTV / dashcam footage, in one
notebook that runs entirely on Kaggle.

**Session options:** Accelerator **GPU T4 ×2**, Internet **On**.
**Add data:** the AHC train+test pack, attached as a Kaggle Dataset.

## The architecture

A three-tier cascade, following *Cerberus* (arXiv 2510.16290), which the problem
statement blesses directly ("a lightweight always-on stage paired with a heavier
verification step"):

| Tier | What | Cost | Sees |
|---|---|---|---|
| 0 | motion gate — frame differencing | ~free | every sampled frame |
| 1 | SigLIP2 + rule-deviation health score | ~30–100 fps | what survives the gate |
| 2 | Qwen3-VL-4B with ASK-Hint prompts | ~1–3 fps | ~12% that stage 1 escalates |

Then per-class temporal aggregation turns window verdicts into events with
timestamps.

**No hosted model is in this path.** The PS's sharpest constraint is that
Gemini/NIM/OpenRouter may inform development but cannot be part of what makes
the detector work at runtime. Everything above runs on the T4.

**Run the cells in order.** Cell 2 only *verifies* the mounted dataset — nothing
in this notebook downloads the pack.
"""


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="syntax-check the cell sources and exit")
    ap.add_argument("--clip", metavar="CELL", default=None,
                    help="copy a cell to the clipboard by its NOTEBOOK label "
                         "(1, 2, ... 5b, 6 ...), not its position in the list")
    args = ap.parse_args()

    sources = {}
    for name, _ in PLAN:
        p = CELLS / name
        if not p.exists():
            print(f"missing cell source: {p}")
            return 1
        sources[name] = p.read_text(encoding="utf-8")

    # Cells share one namespace at runtime, so they only parse standalone. Check
    # each in isolation for syntax, and the concatenation for anything worse.
    for name, src in sources.items():
        try:
            ast.parse(src, filename=name)
        except SyntaxError as e:
            print(f"SYNTAX ERROR in {name}:{e.lineno}: {e.msg}")
            return 1
    try:
        ast.parse("\n".join(sources.values()), filename="<concat>")
    except SyntaxError as e:
        print(f"SYNTAX ERROR in concatenated cells: {e}")
        return 1
    print(f"syntax ok: {len(sources)} cells")
    if args.check:
        return 0

    if args.clip is not None:
        # Match on the LABEL in the filename (c05b -> "5b"), not on position in
        # PLAN. Inserting c05b made those two disagree - "cell 5b" sits at index
        # 6 - and silently pasting cell 6 where 5b was meant is a mistake you
        # only notice several cells later.
        labels = {re.sub(r"^c0*", "", n.split("_")[0]): n for n, _ in PLAN}
        want = str(args.clip).strip().lower()
        name = labels.get(want)
        if name is None:
            print(f"--clip must be one of: {', '.join(labels)}")
            return 1
        import subprocess
        subprocess.run("clip", input=sources[name].encode("utf-8"),
                       check=False, shell=True)
        print(f"cell {want} ({name}) copied to clipboard - "
              "paste into the Kaggle editor")
        return 0

    cells = [md(HEADER)]
    for name, header in PLAN:
        cells.append(md(header))
        cells.append(code(sources[name]))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")

    total = sum(len(s.splitlines()) for s in sources.values())
    print(f"\nwrote {OUT.relative_to(ROOT)}  ({len(cells)} cells, {total} lines of code)")
    for name, _ in PLAN:
        print(f"  {name:22s} {len(sources[name].splitlines()):4d} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
