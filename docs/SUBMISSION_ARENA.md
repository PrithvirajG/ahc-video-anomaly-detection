# Submission arena — what actually gets scored

Two sources: `AHC Visual Intelligence Hackathon-submission and evaluation.pdf`
(the general rules) and the live Benchmark tab (the practice pack itself,
pasted into this project on 2026-09-05). The live page is authoritative where
they differ.

## The practice pack IS our local test set

`Practice pack: 34 videos · L1 24 · L2 6 · L3 4` — that's exactly `data/test/`,
`T001`–`T034`. This is not a private held-out set for the practice round: we
can build `arena_submission.json` and upload it today for a real score. A
later **final** round may swap in unseen videos (the general PDF's own example
uses `E001, E002, ...` ids) — don't hard-code the `T0xx` naming anywhere.

## Scoring weights — this is the priority-setting fact

**D1 = 25, D2 = 35, D3 = 40, out of 100.** 75 of 100 points come from the 10
timed videos (6 at L2, 4 at L3), not the 24 untimed L1 videos. Our first real
run scored well on L1 (8/8 true positives with zero false alarms) but **0/26
on the real temporal gate** — meaning most of the available score currently
comes from the smaller, harder-weighted slice we're failing completely.

## The file

One JSON per submission, covering every video in the manifest:

```json
{
  "schema_version": "1.0",
  "submission_id": "...",
  "model_name": "...",
  "run_metadata": {"total_wall_time_ms": ..., "hardware": "...",
                    "max_parallel_videos": 1},
  "predictions": [
    {
      "video_id": "T001",
      "events": [],
      "runtime_metadata": {
        "frames_processed": ..., "chunks_processed": ...,
        "end_to_end_internal_time_ms": ...,
        "model_runtimes": [{"model_name": "...", "call_count": ...,
                            "total_time_ms": ..., "average_time_ms": ...,
                            "p50_time_ms": ..., "p95_time_ms": ...,
                            "max_time_ms": ...}]
      }
    }
  ]
}
```

### Field rules

| Field | Rule |
|---|---|
| `video_id` | must match the manifest exactly, once each |
| `events` | array; **empty means normal — never `"class_name": "normal"`** |
| `class_name` | one of the 11 anomaly classes, never `"normal"` |
| `start_time_sec` / `end_time_sec` | **`null` at Difficulty 1**; required, `≥0`, `end > start`, **and inside the video's actual duration** at 2–3 |
| `explanation` | optional, 20–500 chars, bonus only — never costs you |
| `runtime_metadata` | **required on every video** — `frames_processed`, `chunks_processed`, `end_to_end_internal_time_ms` (excludes model load/download), `model_runtimes` |

The manifest's per-video level field has been seen called both **"level"**
(general PDF) and **"difficulty"** (live page) — the notebook's manifest
loader (cell 10) accepts either key name defensively.

### Scoring mechanics

- **Level 1**, pooled across all L1 videos: half anomaly-vs-normal accuracy,
  half class accuracy.
- **Levels 2/3**, per video then averaged: ground truth normal + you predict
  nothing = 1; you predict anything = **0, no partial credit**. Ground truth
  has events: a weighted mix of alerted/matched/timing-accuracy, timing
  weighted more at L3.
- **The real temporal gate is IoU ≥ 0.5** (intersection ÷ union), and the
  class must also be correct. **At most one predicted event can match a given
  real event — every other overlapping prediction for that same event counts
  against you.** Several fragments for one real event is worse than one
  well-merged prediction, never better.
- Latency bonus = total reported processing time ÷ total video duration.

### Silent-rejection traps

1. `"class_name": "normal"` → rejected. Use `"events": []`.
2. Non-null timestamps on a Level-1 event → rejected.
3. Omitting a video doesn't clear a previous answer — it keeps it. A video
   never answered at all scores as normal.
4. Multiple fragments for one event — only the best-overlapping one can match.
5. Any prediction on a truly-normal L2/3 video → that video scores zero.
6. Claiming the whole clip is one giant anomalous span → the 0.5 overlap gate
   makes this score far below a real, tight attempt.
7. Missing `runtime_metadata` → also where the latency bonus comes from.

### Final submission (separate from the score)

Required: code repo URL, an architecture write-up (diagram beats prose — which
models, in what order, what runs per frame vs per clip), and a **2-slide PPT**
with real weight in judging — what was built, what approach, what was learned.
Editable any time, doesn't cost a benchmark run.

## What this changes for us

Closing the recall gap on duration-dependent classes (`traffic_congestion`,
`stalled_or_broken_down_vehicle`, `vehicle_blocking_traffic`,
`loitering_or_suspicious_presence`) is now clearly the highest-leverage
remaining work — not just for recall's sake, but because 75% of the practice
score sits on exactly the level where we currently match nothing. See the
"memory between frames" discussion in the working conversation for the
architecture options being weighed.
