# Research notes — small VLMs for video anomaly detection

Read of the four papers the hackathon primer points at, plus the carried-over set
from the 22-Aug repo, and what they jointly imply for the build. PDFs in
`papers/`, extracted text in `papers/raw/`.

## The one that maps onto this problem statement almost exactly

### Cerberus — *Real-Time VAD via Cascaded VLMs* (arXiv 2510.16290)

Reports **57.68 fps average / 151.79× speedup at 97.2% of the best baseline's
accuracy**, i.e. −2.8% AUC for two orders of magnitude less compute. The
architecture is a direct answer to "a lightweight always-on stage paired with a
heavier verification step", which the PS lists as a blessed approach.

**Two phases.**

*Offline induction* — learn what normal looks like in this scene:
1. A VLM (Qwen2.5-VL-7B) describes normal segments, prompted with
   *"How many moving subjects (e.g. people, animals, vehicles) are in the scene,
   and what is each one doing in this specific scenario?"* — deliberately
   behavioural, not object-listing.
2. An LLM abstracts those descriptions into scene-level normality rules.
3. The rule set is augmented with **339 atomic action labels from Moments in
   Time** as *perturbed* negatives.

*Online inference* — a two-tier cascade:
- **Motion mask prompting.** Frame differencing drops static frames outright, then
  overlays red circles/squares on the moving regions. One computation does both
  filtering and visual prompting. Circles pull VLM attention harder (higher
  recall); squares include less background (higher precision) — picked adaptively
  by motion scale.
- **Stage 1**, PE-Core-L14-336 CLIP at 98.41 fps, tuned for **>95% recall** while
  discarding >50% of frames.
- **Stage 2**, Qwen2.5-VL-7B at 3.06 fps on what survives, plus a
  Qwen3-Embedding-4B classifier for the final call.

**Rule-based deviation detection** is the idea worth stealing outright. Rather
than enumerating anomalies, score each segment against a candidate pool of
*normal rules* (weight +1) and *perturbed action labels* (weight −1), take the
top-k by cosine similarity, and sum:

```
health(x) = Σ_{c ∈ topk(x)} w_c · sim(x, c)        w = +1 normal rule, −1 perturbed label
anomalous  ⟺  health(x) < threshold
```

Their motivating evidence for *not* enumerating anomalies is strong: prompting
DeepSeek-R1 to list possible anomalies and matching against that list gave
**27.13% recall on ShanghaiTech and 21.81% on NWPU Campus** despite decent AUC.
Enumeration misses most of what happens.

**Numbers that set our budget** (their L40S, 10 frames): YOLOv10-L 0.43 s / 0.86 GB;
PE-Core-L14-336 0.84 s / 3.19 GB; Qwen2.5-VL-7B **8.48 s / 17.85 GB**. The 7B VLM
is ~20× the time and memory of the detector. On a 4 GB card the 7B tier is simply
unavailable — hence the 3B backbone locally, 4B on a T4.

**Two feedback loops** worth having if time allows:
- *Fine-to-coarse*: frames stage 1 escalated but stage 2 cleared are hard
  negatives; feeding them back sharpens the normal rules so stage 1 filters more.
- *User-in-the-loop*: an operator generalises a rule from a confirmed anomaly.

### Why the cascade suits *this* dataset specifically

Cerberus leans on anomalies being rare (5.38% of frames in ShanghaiTech, 4.45% in
NWPU). Our train set is organised *per class folder including normal*, and the
test set is 34 videos of mixed content — so the same imbalance logic holds at
inference even though the training folders are balanced by construction.

## The CLIP-side papers

### Alert-CLIP (CVPR 2026)

The problem it names is one we will hit immediately: **CLIP's normal and abnormal
text embeddings are entangled**, so a video scores almost identically against "a
normal office scene" and "an anomalous office scene" — and sometimes favours the
normal caption for genuinely anomalous video. Their Figure 1 shows CLIP at
0.2026 vs 0.2182 where Alert-CLIP separates 0.5771 vs 0.1936.

Fix is three-level alignment: video-label (coarse separation), region-text
(anomaly regions ↔ detailed descriptions), region-semantic (contrast against hard
negatives). Trained on VAGTA, 4,212 re-annotated UCF-Crime + MSAD clips with box
and caption annotations. Sold as a drop-in backbone for existing VAD pipelines.

**Practical read:** don't expect raw CLIP similarity against
`"a normal street"` vs `"a traffic accident"` to separate anything. Cerberus's
health-score formulation (normal rules *minus* perturbed action labels) is
partly a workaround for the same entanglement, and it needs no training — which
is why it is the better starting point for a one-day build.

### ASK-Hint — *Fine-Grained Prompting* (WACV 2026)

Training-free, and the cheapest accuracy we can buy. Abstract prompts
("Is there any anomaly in this video?") fail where action-centric ones succeed:
asking *"Do you see punching, kicking, or wrestling on the ground?"* flips a
false negative to a correct detection on the same input. Prompts are organised
into semantically coherent groups (violence, property crime, public safety), and
classes that share action patterns share prompts — "setting fire" serves both
Explosion and Arson.

**Practical read:** our twelve labels should each expand into several concrete,
action-level questions rather than being used as prompt strings directly.
`wrong_way_driving` becomes "is a vehicle facing oncoming traffic?", "is a
vehicle moving against the direction of the other vehicles?", and so on. This is
a text file, not a training run — highest value per minute available today.

### TAU-R1 (arXiv 2603.19098)

Traffic-specific and structurally identical to Cerberus at the top level: a
**lightweight anomaly classifier** for coarse screening, then a **larger anomaly
reasoner** for event summarisation on whatever it flags. Training is
decomposed-QA SFT followed by TAU-GRPO (GRPO with task-specific rewards). Built
on Roundabout-TAU, 342 real roundabout clips with 2,000+ QA pairs.

**Practical read:** independent confirmation that the two-tier split is the right
shape for traffic anomalies specifically — which is what most of our twelve
labels are. The decomposed-QA idea also gives a fine-tuning recipe that fits the
dataset's `description_summary` column directly.

## Carried over from the 22-Aug repo

- **`2603.13306` Compact VLM for CCTV anomaly** — closest prior on small-model VAD.
- **`2404.01014` LAVAD** — training-free VAD via LLMs over captions; the
  no-training baseline to beat.
- **`2412.01095` VERA** — verbalized learning of guiding questions; ASK-Hint is
  explicitly a critique of it (its prompt search is a black box).
- **`2502.14786` SigLIP2** and **`2504.13181` Perception Encoder** — the two
  stage-1 encoder candidates. PE-Core-L14-336 is what Cerberus actually uses.

## What this implies for the build

1. **Stage 1: rule-deviation scoring over frame embeddings.** No training needed.
   Motion-gate first (free, and Cerberus measures 99.9% anomaly recall retained),
   then health-score against normal rules minus perturbed action labels.
2. **Stage 2: small VLM on escalations only,** with ASK-Hint-style fine-grained
   per-class question sets rather than bare label strings.
3. **Per-class temporal aggregation** before emitting an alert — the PS's own
   list of differing event shapes makes a single frame threshold untenable, and
   this is also where false-alarm suppression lives.
4. **Fine-tuning is the stretch goal, not the plan.** `description_summary` +
   Unsloth/ms-swift on a T4 is the path if stage 1+2 lands early.

### Local hardware reality (measured, 2026-09-05)

SigLIP2-base vision tower, batch 32, warmed up, on the GTX 1650:

| precision | throughput |
|---|---|
| fp16 | 11 frames/s |
| **fp32** | **33 frames/s** |

fp32 is 3× faster — TU117 has compute capability 7.5 but the tensor cores fused
off, so half precision has no fast path. Peak VRAM ~1.05 GB of 4 GB. **Do not
default to fp16 locally.** On Kaggle's T4s (real tensor cores) the usual fp16
advice applies again.

33 fps for the always-on stage is comfortably real-time at a 5–10 fps sampling
rate, which leaves the local machine genuinely able to run stage 1 for the demo.
