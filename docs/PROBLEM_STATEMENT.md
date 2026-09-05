# AHC Visual Intelligence Hackathon — Real-Time Video Anomaly Detection

Extracted from `AHC Visual Intelligence Hackathon.pdf`. Date **05 September 2026**,
09:00–19:00, FlytBase Labs or online. Build window is 11:00–18:00; demos 18:00–19:00.

## The task

Drones over a city cover highways, streets, parks, stations, terminals, gatherings
and utility sites, day and night. Most footage is routine; a small fraction needs a
response. Detect those events **while the drone is still overhead**, not in a later
review of the footage.

Why object detectors are not enough, in the PS's own framing: YOLO-class models
report the classes they were trained on, but whether something is an anomaly
depends on **context**, not the object. A stationary car is unremarkable in a
parking bay and a problem on a highway shoulder.

VLMs fit because they aren't tied to a fixed class list and can be queried in
language. The difficulty is cost — large models are too slow and too expensive to
run continuously, and inference has to stay cheap enough to run across many drone
feeds at once.

> The question the hackathon is built around: **can a small VLM do this reliably
> in real time?**

## Constraints

- Must run in real time on **limited GPU capability**.
- Larger hosted models are allowed for development, comparison, or generating
  training data — but **cannot be part of what makes the detector work at runtime**.
  This is the sharpest constraint in the document; it rules out an
  architecture that calls Gemini per frame.
- **False alarms matter as much as missed detections.** "An alerting system that
  fires regularly on ordinary activity stops being used."

Approaches the PS explicitly blesses: fine-tuning a small VLM, distilling a larger
one, pairing a lightweight always-on stage with a heavier verification step,
training something purpose-built, or implementing recent published work.

## The twelve labels

Match these strings **exactly** — scoring compares the string.

```
normal                          traffic_accident
traffic_congestion              stalled_or_broken_down_vehicle
vehicle_blocking_traffic        wrong_way_driving
road_spill_or_debris            waterlogging_or_flood
fire                            smoke
fighting_or_violence            loitering_or_suspicious_presence
```

The PS notes this is *not* a fixed list — covering other events that would matter
to a responder is welcome.

## Temporal shape is not uniform

Called out directly in the PS, and it is the part most likely to break a naive
per-frame classifier:

| Event | Shape |
|---|---|
| Accident | over in ~1 second |
| Congestion | builds gradually |
| Stalled vehicle | only anomalous *after* being stationary a while |
| Waterlogging / open drain | a static condition, not an event |

A single frame-level scorer with one threshold cannot serve all four. Whatever the
stage-1 design, it needs per-class temporal aggregation.

## Dataset

`train/<class_name>/videos/*.mp4` + `videos.csv` + `ground_truth.csv` per class
folder; `test/videos/*.mp4` + the same two CSVs. ~15–17 GB total. No synthetic
anomaly footage. Domains: CCTV, dashcam and drone; highways, streets,
intersections, campuses, open areas; day, night, poor visibility.

Public test set: **34 videos, ~56 minutes**, with `ground_truth.csv` included so
the scoring pipeline can be validated before submitting to the private evaluation.

Training sources are separated from the public test and private eval sets at the
original source-video level, so different cuts of a reserved benchmark source do
not appear in training.

### `ground_truth.csv` fields

| Column | Notes |
|---|---|
| `video_id` | repeats — one video can hold several events |
| `level` | 1, 2 or 3 — the task tier |
| `is_anomaly` | binary label |
| `class_name` | one of the twelve strings, exact match |
| `start_time_sec` / `end_time_sec` | **empty on level 1**, populated on levels 2–3 |
| `description_summary` | short natural-language description; sometimes blank |

Normal videos get one row with `class_name=normal` and empty timestamps.

The three usable framings of the same media, per the dataset doc:
1. raw video for your own preprocessing/sampling pipeline,
2. `ground_truth.csv` for anomaly / class / temporal supervision,
3. `description_summary` for **vision-language fine-tuning or distillation**.

That third one is the hook for the distillation route.

## Compute options offered

Kaggle (30 GPU-hr/week, T4 x2 — phone verification required), Colab (T4),
Lightning AI (~$30 free credits, card unlocks the bulk), Modal (serverless GPU,
$30/month credit, **card required**). Hosted APIs: AI Grants India × FlytBase
(gpt-5.6-luna, handed out on the day, ~4 RPM), NVIDIA NIM (~40 req/min free),
Gemini free tier (Flash/Flash-Lite only, accepts raw video).

See `docs/KAGGLE_REMOTE.md` for how we use Kaggle without the cold-start tax.
