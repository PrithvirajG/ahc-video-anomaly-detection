# Post-mortem and plan: why we scored 37.5, and what to do about it

**Date:** 2026-09-05
**Status:** for review
**Scored:** 37.5 / 100 on the blind evaluation pack, rank 28 of 28
**Context:** hackathon is over. No deadline. This is for correctness and interest.

---

## 1. Where we are

### Arena scores, measured

| Pack | Submission | Score |
|---|---|---|
| Practice | `arena_submission.json` | 27.3 |
| Practice | `arena_submission (1).json` | 33.4 |
| Practice | `arena_submission (2).json` | **37.4** |
| Evaluation | `arena_submission (4).json` | **37.5** |

The work did land: 27.3 → 33.4 → 37.4, and the blind pack came in at the same
level, which means it generalised rather than fitting the practice set.

### The evaluation breakdown

| | Marks | P | R | Found | FA |
|---|---|---|---|---|---|
| D1 — what happened | 11.8 / 25 | 78% | 41% | 7/17 | 2 |
| D2 — what + when | **23.7 / 35** | **100%** | 17% | 2/12 | **0** |
| D3 — what + when + why | **2.0 / 40** | 0% | 0% | 0/6 | 6 |

Two facts worth holding on to:

- **Our D2 precision is 100%, the only such score in the field.** Next best is
  33%. The cascade's discipline is real and worth protecting.
- **Our D3 is 2.0/40, the floor.** All 6 false alarms are one video's fragments.

### Where the marks are

Fitted over the 28 public leaderboard rows (`marks ~ found + log1p(FA)`):

| Difficulty | Marks per event found | Fit quality |
|---|---|---|
| D1 | +1.0 | R² 0.85 |
| D2 | +2.1 | R² 0.32 |
| D3 | **+5.0** | R² 0.73 |

One D3 event is worth five D1 videos. D3 is also where we score zero.

---

## 2. The diagnosis

This is the section that changes the plan. Every number below is measured on
the practice set with `scripts/arena_score.py` and the ceiling analysis.

### The evidence chain

Of the 26 ground-truth timed events in the practice set:

```
  1. a VLM window overlapped it at all          23 / 26   (88%)
  2. a window of the CORRECT CLASS overlapped   3 / 26   (12%)
  3. an oracle could reach IoU >= 0.5           3 / 26   (12%)
  4. we actually score                          2 / 26
```

Line 1 says **coverage is solved**. We looked at 88% of real events. The scan
floor and last-resort look did their job.

Line 3 is the ceiling on every aggregation change. An oracle allowed to pick
any contiguous subset of our own same-class windows, with any buffer, reaches 3.
We score 2.

> **Aggregation headroom is one event.** Everything we have argued about — the
> 15s floor, the 180s cap, the 30s/90s merge gaps — is worth 1 event between
> them, about 5 marks.

### What we said when we looked at a real anomaly

47 windows overlapped a real ground-truth event:

```
  said "normal"           37   (79%)
  wrong anomaly class      4    (9%)
  CORRECT                  6   (13%)
```

### Why "normal" so often — the bug

`kaggle/cells/c06_stage2.py:233`, inside `build_prompt()`:

```python
'{"anomaly": true|false, "class": "<one of: ' + ", ".join(candidates + ["normal"]) + '>", '
```

`candidates` is 3 classes out of 11, chosen by `shortlist_classes()`. The VLM is
required to answer with one of those three, or `normal`.

**How often is the correct class on that menu? 34%.** Random selection of 3 from
11 would give 27%.

Splitting the 47 overlapping windows by whether the right answer was available:

```
  right class NEVER on the menu    31   (66%)   <- our bug
  menu correct, model still wrong  10   (21%)   <- genuine model failure
  correct                           6   (13%)
```

**Two-thirds of our failures are a multiple-choice question with the correct
answer deleted.** On 62% of real events the VLM said "normal" because `normal`
was the only remaining option that was not definitionally wrong.

### The invariant the code states and then breaks

`shortlist_classes()` docstring, `c06_stage2.py:139`:

> *"It is used only to decide which questions to ask, where being roughly right
> is sufficient and being wrong just wastes a question."*

That would hold if candidates only selected which ASK-HINT question banks to
include. But `build_prompt()` also injects them into the required JSON schema
90 lines later, turning a hint into a hard constraint. Being wrong does not
waste a question — it removes the correct answer.

### Supporting measurements

| Measurement | Value | Implication |
|---|---|---|
| Median clip handed to the VLM | **2.0 s** | Classes defined by persistence (loitering, congestion, stalled) cannot be judged |
| Median true event | 20.0 s | 10× mismatch |
| Non-normal confidences | **only 0.95 or 0.98** | `enter_conf`/`exit_conf` have never been exercised; no precision/recall knob exists |
| Motion gate rejection | 31% (designed for ~50%) | Drone ego-motion defeats raw pixel differencing |
| Stage 2 share of wall time | 83% | Any speed work must target stage 2 or nothing |
| Truth events per video | median 4, max 6 | We emit 0–2 |
| Truth event durations | 2.6 s – 125 s | No single constant can serve this range |

---

## 3. What we are NOT doing, and why

Three things I previously recommended, now retracted on evidence.

| Retracted | Why |
|---|---|
| "31% of events are unmatchable because of the 15s floor — fix first" | True about the code, irrelevant in practice. Those events had no correct-class window either. |
| "Merge gaps are too wide, we cannot emit 6 events" | True about the code, irrelevant. T025 has 6 events and **zero** accident-class windows. Nothing to split. |
| "Aggregation fixes are the top priority, 7 minutes for most of the gap" | Measured: the whole aggregation category is worth **1 event**. |

I swept every de-hardcoded aggregation configuration — 12 gap values × 6 buffer
values × 6 event caps, no floor, no cap — replayed against real ground truth:

```
  as-submitted (15s floor, 180s cap, 30/90s gaps)      D3 1/8    est 41.7
  best principled alternative (gap 20s, buffer 5s)     D3 1/8    est 40.6
```

**No principled version beats the hardcoded one today.** Not because hardcoding
is good, but because the events it would measure properly were never detected.

The lesson: I was reading code for bugs and finding real ones without checking
whether they were **binding**. They were not.

---

## 4. The plan

Ordered by measured evidence.

### Tier 1 — do these first, they are cheap and the evidence is direct

#### 1. Stop constraining the VLM's answer to the shortlist

**Change.** In `build_prompt()`, put all 11 anomaly classes plus `normal` in the
JSON schema. Keep `candidates` for selecting which ASK-HINT question banks to
include — its documented purpose. Per the review decision, include the top-5
question banks rather than all 11, to avoid a 4× longer prompt diluting
attention.

**Evidence.** 31 of 47 windows that overlapped a real event failed only because
the correct class was absent from the menu.

**Cost.** One line. No GPU. Requires one re-run to measure (~21 min).

**Expected.** Up to 20 events of headroom. The largest single item by a wide
margin, and everything else is unmeasurable until it lands, because the model
is currently answering a rigged question.

**Risk.** More classes offered means more opportunity for false positives, which
threatens our 100% D2 precision. Measure D2 FA specifically after this run.

#### 2. Train a linear probe for the shortlist

**Change.** Fit 11 one-vs-rest logistic regressions on frozen SigLIP2 embeddings
using the 3,207 labelled training clips. Use the probe, not text similarity, to
rank classes for the shortlist and for the health score.

**Is this "training"?** Yes, but narrowly. No backprop through the encoder, no
GPU, no fine-tuning. SigLIP2 stays frozen and zero-shot. ~5 minutes of CPU. What
changes is that 93 hand-written English sentences are replaced by coefficients
learned from labelled examples — strictly more information from data we already
have and currently use for exactly one number.

**Evidence.** The current shortlist hits 34% against a 27% random baseline — it
is barely informative. Separately, the text-similarity approach suffers the
CLIP-family modality gap: image and text embeddings occupy separate cones, which
compresses all similarities into a narrow band. That is the mechanism behind
both the shortlist failure and the threshold not transferring between cameras.
A probe on image embeddings alone sidesteps the gap entirely.

**Leaderboard support.** Roughly 20 of 28 entries trained a head, probe, or
LoRA. We trained nothing. The #1 team (67.8) used a *2B* VLM — half our size —
with a trained CLIP head and a temporal state machine. Capacity was never the
differentiator.

**Cost.** ~1 hour including the run.

**Open decision.** Frame-level or clip-level probe — see §6.

#### 3. Remove the hardcoded extents

**Change.**

```
1. Group windows into runs: same class, gap <= (1 / sample_fps) * 2
2. Extent = [first t0 - buffer, last t1 + buffer]   buffer = 2s, symmetric
3. No floor. No cap. No fallback prior.
4. A one-window event is as short as that window. That IS the measurement.
5. Optional: re-sample that region at higher fps and walk the health curve to
   its crossings, genuinely locating the boundary instead of inheriting the
   sampler's grid.
```

Constants removed: `FALLBACK_EVENT_SEC = 15.0`, `EXTENT_MAX_SEC = 180.0`,
`CROSS_CLASS_GAP_SEC = 30.0`, `SAME_CLASS_GAP_SEC = 90.0`.

**Evidence.** Worth ~1 event (≈5 marks, on D3) today. Becomes load-bearing once
items 1–2 land and there are correct-class windows to bound.

**Rationale.** Adopt it because it is correct, not because it scores. A system
asserting "this lasted 15 seconds" with no measurement behind it is not
measuring anything. Step 5 is the real answer to *why can't we detect the
duration* — our sampling grid is 0.5 s, so a 5-second event has 10 samples in
it. The boundary is measurable; we replaced the measurement with a constant.

**Cost.** 30 min.

#### 4. Emit `description_summary` on D3

**Change.** Populate the description field the arena's Reason bonus reads.

**Evidence.** We score `-` (zero) on Reason. The field earns +1.0 to +4.0.

**Cost.** 20 min. Free marks.

### Tier 2 — after Tier 1 has been measured

#### 5. Cosmos-Embed1-448p-anomaly-detection as stage 1

**Change.** Replace or supplement SigLIP2 with NVIDIA's anomaly-tuned embedding
model. 1B params, LoRA-tuned on VAD-Reasoning (1,755 videos, 24 anomaly
categories), 8 frames per 5-second chunk, 768-dim output. Top-1 46.4% vs 23.2%
for the base model.

**Why it matters.** It is clip-native rather than frame-native, which fixes the
temporal blindness at stage 1 where it is cheap, instead of only at stage 2
where it is expensive. And it is already tuned on our category vocabulary.

**Leaderboard support.** #23 scored 40.4 running it bare — above us. #16 scored
47.1 running it with a LoRA.

**Cost.** ~1 hour.

#### 6. Homography-stabilised motion gate

**Change.** Estimate the frame-to-frame homography (ORB features +
`cv2.findHomography`), warp to align, *then* difference. Optionally low-pass the
global trajectory to remove jitter.

**Evidence.** The gate rejects 31% where it was tuned for ~50%. On drone footage
every frame differs because the camera moved, so the gate degenerates toward a
no-op on exactly the footage where filtering matters most.

**What it buys beyond speed.** It inverts the signal correctly: a moving drone
over a calm scene yields near-zero residual, while a stationary drone over a
real incident yields a large one. That is the behaviour we want and currently
have backwards.

**Cost.** 30 min, no new model.

#### 7. Diverse frames instead of consecutive ones

**Change.** Within a window, select the N most visually distinct frames
(perceptual hash or embedding distance) rather than N consecutive ones.

**Evidence.** We currently pass 4 frames from a ~2-second span at 2 fps — in a
static scene these are near-identical. We spend four image slots showing the
model the same picture four times. Same token budget, same GPU time, several
times the information.

**Cost.** 20 min.

### Tier 3 — larger, only once Tier 1–2 are measured

#### 8. LoRA on Qwen3-VL

Real training this time. Deliberately last: attempting it while the answer menu
is rigged and windows are 2 seconds wide would confound the one measurement that
matters. The #1 team did this on a 2B model, so it works — but they also had a
trained head and temporal states.

**Cost.** 2–3 hours.

#### 9. Switch to T4 × 2

32 GB total instead of 16, and roughly 2× throughput by processing two videos in
parallel. Also the only way to run an 8B model in fp16 without quantisation.
Free, one dropdown, and we never used it.

**Cost.** 1 minute.

---

## 5. Measurement constraints

**There is no ground truth for the evaluation pack.** This shapes everything:

- All tuning happens on the **practice** set, scored locally with
  `scripts/arena_score.py`.
- The evaluation pack is **submit-and-see**. Our only feedback is the aggregate
  score, the per-video found/missed/FA counts from the arena timeline page, and
  our leaderboard row.
- Therefore: **lock every parameter on practice before submitting**, and treat
  each evaluation submission as a costly measurement, not an experiment.

**No corrected test ground truth exists.** `ground_truth_corrected_v2.csv` is
164 training videos (`TR00001`–`TR03214`) covering `wrong_way_driving` only —
zero overlap with the T0xx test set. We use `data/test/ground_truth.csv` as-is.

It remains useful for two things: 56 clean labelled positives for a class that
appears exactly once in our test set, and confirmation that short events
dominate (median duration 3.7 s, minimum 0.0 s).

**One known discrepancy, unresolved.** Our local CSV gives T028 four events, but
the arena's practice timeline skips T028 entirely. Since no corrected version
exists, we proceed with the local CSV and note that local D2 numbers may be
slightly pessimistic.

---

## 6. Decisions needed

### D-1. Probe granularity — frame-level or clip-level?

| | Frame-level | Clip-level |
|---|---|---|
| Input | one embedding per frame | 8 frames mean-pooled |
| Training data | all 3,207 clips, every frame | 3,207 examples |
| Simplicity | higher | lower |
| Persistence classes | still blind | can represent them |

**Recommendation: clip-level.** The failing classes are all temporal, and a
frame-level probe inherits exactly the blindness we are trying to remove.

### D-2. Is protecting D2 precision more valuable than finding D2 events?

The D2 marks fit has R² 0.32 with a ~19-mark intercept, which looks like credit
for correctly leaving normal videos alone. If that is right, our 100% D2
precision may be worth more than additional D2 finds — which would reorder the
plan, since item 1 puts that precision at risk.

**Testable:** score a deliberately noisy D2 submission and observe the drop.
Cheap and worth doing before item 1 if we care about the ordering.

### D-3. How far up the tiers do we go?

Tier 1 keeps the architecture. Tier 2 replaces stage 1. Tier 3 means real
training. No deadline, so this is purely appetite.

**Recommendation:** Tier 1 in full, measure, then decide. Items 1–2 convert *"is
a 4B model good enough?"* from unanswerable into answerable, which is the most
interesting question left.

---

## 7. Sequencing

```
  1. item 1  (answer menu)            5 min   -> re-run -> score locally
  2. item 4  (descriptions)          20 min   -> free marks, no risk
  3. item 3  (de-hardcode extents)   30 min   -> correctness; measure
  4. item 2  (linear probe)           1 hr    -> re-run -> score locally
  ---- review, decide on Tier 2 ----
  5. items 6, 7  (gate, frames)      50 min
  6. item 5  (Cosmos-Embed1)          1 hr
  ---- review, decide on Tier 3 ----
```

Every step is measured on the practice set before the next begins, and no
evaluation submission happens until a step has demonstrably improved the
practice score.

---

## 8. Appendix: tooling added

| File | Purpose |
|---|---|
| `scripts/arena_score.py` | Reproduces the arena's P/R/found/FA per difficulty against local ground truth, with ASCII timelines. Marks estimate is a leaderboard fit — see its docstring for the caveats. |
| `scripts/build_deck.py` | Regenerates the 2-slide submission deck from measured numbers. |

Usage:

```bash
.venv/Scripts/python.exe scripts/arena_score.py "predictions_raw (4).json" --timeline
```
