# Guaranteed scan floor for long videos

**Date:** 2026-09-05
**Status:** approved, implementing

## Problem

Six of the eight anomalous long videos in the test set send **one or zero**
windows to the VLM. Those videos carry 75 of the 100 available marks.

```
T025 (congestion x6)     132/500 frames kept  ->  1 window
T027 (congestion x4)     420/500              ->  0 windows
T031 (congestion, 125s)  656/720              ->  0 windows
T032 (loitering x4)      559/577              ->  0 windows
T034 (loitering)         706/706              ->  0 windows
```

T027 has four real traffic jams in it. Not one of its 420 surviving frames
scored below the escalation threshold, so the VLM never looked at the video
at all, and we reported "normal" without examining it.

### Root cause 1: a threshold calibrated on the wrong distribution

`health_thresh` is derived by measuring 12 **known-normal training clips**
(5-30s each, one scene apiece) and cutting at their 12th percentile: -0.41.
It is then applied to test videos of 240-629s from different cameras and
scenes. It is an absolute cut taken from one distribution and applied to
another - a thermometer calibrated indoors and used outdoors.

Measured consequences:
- design intends 12% of frames to escalate; T033 escalates ~2%, four other
  long videos escalate 0%
- T008, containing dense visible traffic congestion, scores **+0.585** -
  reading as *more normal than normal footage*

### Root cause 2: the health score cannot perceive motion

The normal-rule pool contains phrases like *"traffic flowing smoothly through
an intersection"*. In a single still frame it is impossible to tell whether
those vehicles are moving or stopped, so the phrase matches any road scene and
votes +1 regardless. Five such motion-describing rules exist, and they
dominate on congestion.

This is why every duration-defined class scores zero: `traffic_congestion`
0/4, `loitering_or_suspicious_presence` 0/4, `stalled_or_broken_down_vehicle`
0/1. Every decision in the pipeline looks at one frozen instant.

## The reframe

The cascade exists to conserve GPU time. Measured: the full 56-minute test set
processes in **8.3 minutes**, with only 48% of that inside the VLM. Scanning
every long video every 20s would cost **~14 minutes of GPU**.

We are aggressively filtering to protect a budget we are not spending, and the
filtering is what costs us the marks.

## Design

A scan floor: long videos are guaranteed a VLM look at a fixed interval,
regardless of health score. The existing escalation path is untouched; the
floor only ever **adds** windows.

```
sample -> motion gate -> health score -> escalate      (unchanged)
                                             |
                                    ADD SCAN FLOOR      (new, long videos only)
                                             |
                                   VLM over the union -> aggregate
```

### Configuration

```python
scan_floor_enabled       = True
scan_floor_min_video_sec = 60.0    # only videos longer than this
scan_floor_interval_sec  = 20.0    # guarantee a look at least this often
```

**60s** cleanly separates the two populations present in the data - L1 clips
are 5-27s, L2/L3 videos are 240-629s. Nothing sits near the boundary, so the
cutoff is not delicate.

**20s** is chosen against the measured event distribution: the median real
event is 20s, so a 20s stride is guaranteed to land inside any event of median
length or longer. Events shorter than 20s can still fall between looks - a
known and accepted gap at this interval, which the knob exists to tighten.

### Mechanism

For each 20s slot not already covered by an escalated window, build a window
from the sampled frames nearest that timestamp. It then follows the identical
`pick_frames -> shortlist_classes -> vlm_verify` path. Scan windows are **not**
special-cased downstream: same code, same prompts, same aggregation.

### Measurement

Every window carries `source: "escalated" | "scan"`, and the tag rides through
to each detection.

This is the point of the exercise. After one run we can state how many
detections the scan floor produced, how many were correct, and how many false
alarms it caused - rather than observing a changed score and guessing why. It
also makes the ablation free: if scan detections prove to be mostly noise, we
filter on the tag instead of re-running.

### Safety valve, deliberately deferred

The first run ships **without** an extra confidence bar or corroboration rule.
Confidence is measurably uninformative here (0.85-0.98 whether right or wrong),
so a higher bar would be theatre, and a smarter guard should be designed
against real data rather than guessed at now.

If false alarms appear on T029/T030 (genuinely-normal long videos, currently
silent and correct), the guard to add is **corroboration**: a scan-floor
detection becomes an event only if a neighbouring window agrees or the health
score there is at least mildly depressed. Escalated windows keep their current
free pass, having evidence behind them already.

### Companion change: persist raw window verdicts

Save per-window verdicts to disk alongside aggregated events (~5 lines).

Currently any aggregation change costs a full 8-minute GPU re-run, which is
why a measured improvement to the fallback duration (10s -> 15s, converting
T026's road_spill from IoU 0.481 to a pass) is still unshipped. With raw
verdicts persisted, aggregation experiments take seconds.

## Explicitly out of scope

**Adaptive per-video thresholding.** A real bug and worth fixing, but shipping
it alongside the scan floor makes the two indistinguishable in the results.
Scan floor first, measure, then adaptive as a second run. Both flags stay
independent so either can be ablated.

## Risk

Every option here means the VLM examines more normal footage. Under arena
rules, a single false alarm on a genuinely-normal L2/L3 video scores that
video **zero**. The current record on those is perfect (0 false alarms across
6 normal videos), and this design puts that at risk in exchange for recall.

Mitigated by the tagging (we will know exactly what the floor cost us) and by
having budget for 6-8 runs, so a bad outcome can be measured and reverted
rather than discovered at submission time.

## Expected cost

~14 min of GPU per run, up from 8.3.
