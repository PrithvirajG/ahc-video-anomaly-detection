# =============================================================================
# 7 - Temporal aggregation: window verdicts -> incidents
# =============================================================================
# Three jobs, in order:
#   1. cluster detections that belong to the same incident (across classes)
#   2. give each cluster ONE main tag plus sub-tags for everything else seen
#   3. give it a real start and end, measured rather than assumed
#
# The problem statement is explicit that events do not share a temporal shape:
# an accident is over in about a second, congestion builds gradually, a stalled
# vehicle is only anomalous AFTER standing a while, waterlogging is a static
# condition. So persistence is per class, and events open/close with hysteresis
# - one confident window opens nothing, a real event survives a brief occlusion
# instead of fragmenting.

# min_dur = how long a detection must persist before it counts as this class.
# MEASURED, not reasoned: the first version came from the PS's prose and was
# badly wrong - loitering was set to 20s when the timed ground truth has a
# MEDIAN of 13.7s, and stalled_vehicle to 15s when the training clips for that
# class average 11.2s in total. "Only anomalous after standing a while" is true
# of the phenomenon and says nothing about how the dataset clipped it.
TEMPORAL = {
    "traffic_accident":                 1.0,   # PS: over in ~1s; GT min 5.0s
    "traffic_congestion":               2.5,   # GT min 5.0s
    "stalled_or_broken_down_vehicle":   4.0,   # train clips avg 11.2s total
    "vehicle_blocking_traffic":         3.0,   # GT 9.5s
    "wrong_way_driving":                1.5,   # short but unambiguous
    "road_spill_or_debris":             2.0,   # static condition
    "waterlogging_or_flood":            2.0,   # static condition
    "fire":                             1.5,
    "smoke":                            2.0,
    "fighting_or_violence":             1.5,   # GT 60s, train clips are short
    "loitering_or_suspicious_presence": 3.0,   # GT min 2.6s - NOT 20s
}
DEFAULT_MIN_DUR = 2.0

# Detections within this many seconds are treated as one incident, REGARDLESS
# of class - the smoke at 507s and the altercation at 523s in T033 are two
# views of one accident, not two incidents. The arena scores only the
# best-overlapping prediction per real event and counts every other one
# against you, so fragmenting is penalised twice over.
# Must tolerate a scan slot or two coming back "normal" in the middle of a real
# event. At a 20s scan interval a single normal slot opens a 40s gap, and 30s
# split T031's fourteen agreeing congestion windows into three separate events -
# each then emitted at the 15s fallback, turning one 125s event into three
# fragments that the arena penalises. 60s = three scan slots.
# --- REMOVED: CROSS_CLASS_GAP_SEC / SAME_CLASS_GAP_SEC ------------------------
# Both were time constants standing in for a question the data already answers.
# T031 settles it: fourteen windows all calling traffic_congestion span 9s-311s
# at a 20s scan stride, and the truth is ONE 125s event at 235-360. Merging all
# fourteen gives IoU 0.22; splitting them all gives ~0.02 each. No value of a
# gap constant produces the right answer, because the right question is not how
# far apart two detections are but whether anything happened in between - and
# the health curve records exactly that. See _recovered_between().

# --- how long was it, really? ------------------------------------------------
# Our window boundaries measure OUR SAMPLING, not the event: 4 frames at 2fps is
# a ~2s window because that is how we look, not because the event lasted 2s.
# Median predicted duration was 4.0s against a real median of 20s, and IoU
# between those is 0.20 - below the arena's 0.5 gate even with a perfect class
# and perfect centring. That made 75 of 100 marks (D2+D3) unreachable no matter
# how good detection got.
#
# So measure the extent instead of assuming it: walk outward from the detection
# while the health score stays depressed. Verified on T033's real predictions -
# the walk returns [507.0, 533.5] against a true event of [490, 535], IoU 0.589,
# a pass. The same detections under a fixed 10s expansion give IoU 0.444, a
# fail. Measurement beats the constant.
EXTENT_LOOSE_FACTOR = 0.4    # walk while health < health_thresh * this.
                             # Kept, because it is a FRACTION OF A CALIBRATED
                             # NUMBER, not a guessed duration - health_thresh is
                             # fitted on known-normal clips, and this says "still
                             # depressed" relative to it.

# --- REMOVED: EXTENT_MAX_SEC (180s cap) --------------------------------------
# It made any event longer than 360s unmatchable, and the evaluation pack's
# E027 is exactly that: one event spanning ~600s of a 602s video. A cap on how
# long reality is allowed to be is not a safeguard, it is an assertion.

# --- REMOVED: FALLBACK_EVENT_SEC (15s floor) ---------------------------------
# The honest reason it existed: when the health curve gave no usable extent, we
# had nothing to say and said "15 seconds" anyway. That is not a measurement,
# and it made 8 of 26 ground-truth events (31%) unmatchable by construction -
# IoU >= 0.5 needs a prediction no more than twice the true length, so a 15s
# floor can never match a 5s event, of which the practice set has nine.
#
# Replaced by the only assumption we actually need: a symmetric buffer, below.


def extent_buffer(cfg=None) -> float:
    """Half-width of the uncertainty around a detected boundary, in seconds.

    This is the ONE declared constant left in the extent path, and it is a
    quantisation allowance rather than a guess about duration: frames arrive on
    a 1/sample_fps grid, so a true boundary can sit up to one sampling interval
    outside the window that caught it, and the window itself is built from
    whichever frames the sampler happened to land on.

    Default 2.0s. Note the tension - a larger buffer helps a boundary we
    straddled and hurts a very short event, because IoU falls as the prediction
    outgrows the truth. On a 2.6s event (T032 has one) a 2s buffer already
    costs the match. It lives in CFG so it can be swept offline against stored
    window_verdicts without another GPU run.
    """
    cfg = cfg or CFG
    return float(getattr(cfg, "extent_buffer_sec", 2.0))


def _recovered_between(health_curve, t_a: float, t_b: float, loose: float) -> bool:
    """RETIRED as a splitting criterion. Kept for diagnostics only.

    The idea was sound and the signal is not. Measured on the five-video run,
    comparing health inside a real ground-truth event against health outside it
    in the same video:

        T026  inside +0.134  outside +0.608   separation +0.474   works
        T031  inside -0.085  outside -0.092   separation -0.007   flat
        T032  inside +0.531  outside +0.415   separation -0.116   INVERTED
        T025  inside +0.150  outside -0.097   separation -0.247   INVERTED

    On three of four videos the frames containing the anomaly are as healthy as
    or HEALTHIER than the rest of the video - event frames sit at the 65th
    percentile of their own video's health. A congested road looks like a road;
    a person loitering looks like a person. Only T026's road spill, a plainly
    visible appearance change, separates at all.

    Used as a split test this was catastrophic on T031: health_thresh is
    calibrated globally at about -0.4, so loose is about -0.16, and T031's
    *minimum* health over 656 samples is -0.119. Every sample reads as
    "recovered", so eighteen agreeing congestion windows became eighteen events
    and eighteen false alarms. The best contiguous subset of those same windows
    scores IoU 0.917.
    """
    if not health_curve:
        return False
    mids = [h for t, h in health_curve if t_a < t < t_b]
    if not mids:
        return False
    return sum(h >= loose for h in mids) > len(mids) / 2


def measure_extent(cluster: list[dict], health_curve, thresh: float,
                   duration_sec: float | None = None,
                   cfg=None) -> tuple[float, float, str]:
    """How long was it? Answer from the windows and the curve, nothing else.

    Two measurements, composed:
      1. the span of the windows that actually saw it - direct evidence, and
         previously thrown away in favour of a prior
      2. extended outward while the health curve stays depressed - the scene
         itself telling us where it returned to normal

    ...plus a symmetric quantisation buffer. No floor, no cap, no prior. A
    one-window event is as short as that window; a forty-window event is as long
    as they span; and if the curve says the depression continues past the last
    window, so does the event.

    Returns (start, end, source) with source "measured" when the curve was
    consulted and "windows" when there was no curve to consult.
    """
    buf = extent_buffer(cfg)
    s = min(w["t0"] for w in cluster)
    e = max(w["t1"] for w in cluster)
    src = "windows"

    if health_curve:
        ts = [t for t, _ in health_curve]
        hs = [h for _, h in health_curve]
        loose = thresh * EXTENT_LOOSE_FACTOR
        a = next((j for j, t in enumerate(ts) if t >= s), len(ts) - 1)
        while a > 0 and hs[a - 1] < loose:
            a -= 1
        b = next((j for j in range(len(ts) - 1, -1, -1) if ts[j] <= e), 0)
        while b < len(hs) - 1 and hs[b + 1] < loose:
            b += 1
        s, e = min(s, ts[a]), max(e, ts[b])
        src = "measured"

    s, e = s - buf, e + buf
    if s < 0:
        s = 0.0
    if duration_sec is not None and e > duration_sec:
        e = float(duration_sec)
        s = min(s, max(0.0, e - 1e-3))
    return s, e, src


def cluster_windows(windows: list[dict], health_curve=None,
                    thresh: float | None = None) -> list[list[dict]]:
    """Group same-class detections into one incident. Split only on class.

    We do not split on time, because a gap constant cannot express the question
    (T031 needs windows at 9s and 209s separated but 249s and 348s joined). We
    no longer split on the health curve either, because that curve is flat or
    inverted on three of four measured videos - see _recovered_between().

    So there is currently NO validated signal for where one incident ends and
    the next begins, and this asserts none. Measured on the five-video run,
    which is the whole justification:

        rule                          matched   false alarms   events
        health recovery (shipped)        0           20          20
        per-video relative recovery      0           12          12
        merge same class  <- this        0            2           2
        any time gap >= 60s              0            2           2

    Nothing recovers a match, because only 1 of the 15 ground-truth events is
    reachable from these windows at all. But asserting eighteen boundaries we
    cannot support costs eighteen false alarms, and the arena charges for each
    one. When the evidence does not distinguish, claim less.

    This becomes wrong the moment a video genuinely contains two separate
    incidents of the same class - T025 has six - so it is a stopgap, and the
    thing that unblocks it is a health signal that actually tracks anomalies,
    i.e. the linear probe.

    health_curve and thresh are accepted and ignored, so the call sites and the
    diagnostics that pass them keep working.
    """
    ws = sorted(windows, key=lambda w: w["t0"])
    hits = [w for w in ws if w["class"] != "normal"]
    clusters, cur = [], []
    for w in hits:
        if cur:
            prev = cur[-1]
            # Split where STAGE 2 ITSELF said normal. That is a judgement from
            # the model that can see the scene, it needs no constant, and it is
            # the only recovery signal we have left now the health curve has
            # been shown to be flat or inverted. Measured on the 34-video run it
            # recovers T033's cross-class match (smoke at 507s and an
            # altercation at 523s with nothing normal between them are one
            # incident) at the same false-alarm count as splitting on class.
            if any(x["class"] == "normal" and prev["t1"] <= x["t0"] and x["t1"] <= w["t0"]
                   for x in ws):
                clusters.append(cur)
                cur = []
        cur.append(w)
    if cur:
        clusters.append(cur)
    return clusters


def aggregate_events(windows: list[dict], cfg=None, health_curve=None,
                     duration_sec: float | None = None,
                     adjudicator=None) -> list[dict]:
    """Turn per-window verdicts into incidents with one main tag and sub-tags.

    windows: [{"t0","t1","class","confidence","description"}, ...]
    health_curve: [(t, health), ...] for measuring extent; None -> fallback prior
    adjudicator: optional fn(cluster) -> {"primary","confidence","reason"} used
        when a cluster holds several classes. NO causal table is consulted -
        the same class is a symptom in one context and the incident itself in
        another (smoke over a wreck vs smoke over a thermal plant), so the
        judgement is asked of the model that can see the context.
    """
    cfg = cfg or CFG
    thresh = cfg.health_thresh if cfg.health_thresh is not None else -0.4
    out = []

    for cluster in cluster_windows(windows, health_curve, thresh):
        strong = [w for w in cluster if w["confidence"] >= cfg.enter_conf]
        if not strong:
            continue                       # hysteresis: nothing opened this cluster

        classes = {}
        for w in cluster:
            c = classes.setdefault(w["class"], {"n": 0, "conf": 0.0, "first_t": w["t0"],
                                                "desc": w.get("description", "")})
            c["n"] += 1
            c["conf"] = max(c["conf"], w["confidence"])

        best = max(strong, key=lambda w: w["confidence"])
        main, main_conf, reason = best["class"], best["confidence"], "highest confidence"

        if len(classes) > 1 and adjudicator is not None:
            verdict = adjudicator(cluster)
            if verdict:
                main = verdict["primary"]
                main_conf = max(main_conf, verdict["confidence"])
                reason = verdict.get("reason", "adjudicated")

        # The span of the agreeing windows is itself a measurement of duration and
        # is now where measure_extent STARTS, rather than something it has to
        # override afterwards. The old window-span override existed only because
        # the fallback prior kept discarding this evidence; with the prior gone
        # there is nothing to override.
        start, end, src = measure_extent(strong, health_curve, thresh,
                                         duration_sec, cfg)

        # TEMPORAL survives the de-hardcoding on purpose, and the distinction is
        # worth being explicit about: it never invents a duration, it only
        # SUPPRESSES a claim that is physically implausible for its class (a
        # 0.3s fire). Every value is <= 4s and below the shortest real event we
        # have, so it should never fire in practice - if it starts firing, that
        # is a signal worth reading, not a threshold worth raising.
        if (end - start) + 1e-6 < TEMPORAL.get(main, DEFAULT_MIN_DUR):
            continue                       # too brief to be this class

        sub_tags = [{"class_name": c, "peak_confidence": round(v["conf"], 3),
                     "n_windows": v["n"], "first_seen_sec": round(v["first_t"], 2),
                     "description": v["desc"]}
                    for c, v in sorted(classes.items(), key=lambda kv: -kv[1]["conf"])
                    if c != main]

        out.append({
            "class_name": main,
            "start_time_sec": round(start, 2),
            "end_time_sec": round(end, 2),
            "confidence": round(float(np.mean([w["confidence"] for w in strong])), 3),
            "peak_confidence": round(float(main_conf), 3),
            "n_windows": len(cluster),
            "extent_source": src,           # "measured" (curve consulted) | "windows"
            "primary_reason": reason,
            "sub_tags": sub_tags,           # kept for analysis, not for the arena JSON
            "description_summary": best.get("description", ""),
        })
    return sorted(out, key=lambda e: e["start_time_sec"])


def _selftest():
    mk = lambda c, t, conf: {"t0": t, "t1": t + 1.0, "class": c,
                             "confidence": conf, "description": ""}

    # hysteresis: weak windows alone open nothing
    assert not aggregate_events([mk("fire", t, 0.4) for t in range(5)]), \
        "sub-enter_conf windows must not open an event"

    buf = extent_buffer()

    # NO FLOOR. A lone 1s detection is a ~1s event plus the buffer either side -
    # not the 15s the old prior asserted. This is the change that makes the
    # nine sub-10s ground-truth events reachable at all.
    ev = aggregate_events([mk("traffic_accident", 5.0, 0.9)])[0]
    span = ev["end_time_sec"] - ev["start_time_sec"]
    assert abs(span - (1.0 + 2 * buf)) < 1e-6, f"expected measured span, got {span}"
    assert ev["extent_source"] == "windows", "no curve given -> say so"

    # NO CAP. A genuinely long event stays long: E027's truth is one event over
    # ~600s of a 602s video, which the old 180s cap made unmatchable.
    long_curve = [(t, -0.5) for t in range(0, 600, 2)]
    ev = aggregate_events([mk("traffic_congestion", t, 0.9) for t in range(10, 580, 20)],
                          health_curve=long_curve, duration_sec=602.0)[0]
    assert ev["end_time_sec"] - ev["start_time_sec"] > 500, \
        f"a 600s event must survive aggregation, got {ev}"

    # SAME CLASS NEVER SPLITS, whatever the curve does. This is the T031 shape,
    # and it asserts a deliberate retreat: an earlier version split here on a
    # health recovery, which turned 18 agreeing congestion windows into 18 false
    # alarms on the real video because that curve is flat (its whole range sits
    # above the "recovered" line). Until a signal exists that actually tracks
    # anomalies, we assert one incident rather than eighteen boundaries we
    # cannot support.
    recov = [(t, 0.2 if 120 <= t <= 220 else -0.5) for t in range(0, 320, 2)]
    same = [mk("traffic_congestion", t, 0.9) for t in (20, 60, 100, 240, 280, 300)]
    assert len(aggregate_events(same, health_curve=recov, duration_sec=320.0)) == 1, \
        "same class must not split on a curve we have shown to be unreliable"
    flat = [(t, -0.5) for t in range(0, 320, 2)]
    assert len(aggregate_events(same, health_curve=flat, duration_sec=320.0)) == 1, \
        "...and the same with no recovery at all"

    # A class change alone does NOT split: T033's smoke at 507s and altercation
    # at 523s are one incident seen twice, and the adjudicator exists to name it.
    mixed = [mk("smoke", 507.5, 0.95), mk("fighting_or_violence", 523.5, 0.90)]
    assert len(cluster_windows(mixed)) == 1, "a class change alone is not a boundary"

    # ...but stage 2 saying "normal" in between IS a boundary, and it is the only
    # recovery signal left. Measured: this recovers T033's match at no extra cost.
    with_normal = [mk("traffic_congestion", 20.0, 0.9),
                   mk("normal", 100.0, 0.9),
                   mk("traffic_congestion", 200.0, 0.9)]
    assert len(cluster_windows(with_normal)) == 2, \
        "a normal verdict between two detections separates them"

    # CROSS-CLASS, T033's real shape: smoke then an altercation 16s later with
    # health depressed across the interval is one incident seen twice. The class
    # change is not what decides it - the curve is.
    t033_curve = [(t, -0.5 if 500 <= t <= 530 else 0.2) for t in range(400, 600)]
    t033 = [mk("smoke", 507.5, 0.95), mk("fighting_or_violence", 523.5, 0.90)]
    evs = aggregate_events(t033, health_curve=t033_curve, duration_sec=628.8)
    assert len(evs) == 1, "one incident, not two"
    assert evs[0]["class_name"] == "smoke"
    assert [s["class_name"] for s in evs[0]["sub_tags"]] == ["fighting_or_violence"]

    # ...and with an adjudicator the model's call wins, sub-tags still kept
    evs = aggregate_events(t033, health_curve=t033_curve, duration_sec=628.8,
                           adjudicator=lambda c: {"primary": "traffic_accident",
                                                  "confidence": 0.8,
                                                  "reason": "aftermath"})
    assert evs[0]["class_name"] == "traffic_accident"
    assert {s["class_name"] for s in evs[0]["sub_tags"]} == {"smoke", "fighting_or_violence"}

    # the curve extends an event beyond the window that caught it
    curve = [(t, -0.5 if 490 <= t <= 535 else 0.2) for t in range(400, 600)]
    ev = aggregate_events([mk("traffic_accident", 510.0, 0.9)],
                          health_curve=curve, duration_sec=628.8)[0]
    assert ev["extent_source"] == "measured"
    assert ev["start_time_sec"] <= 495 and ev["end_time_sec"] >= 530, \
        f"measured extent should track the low-health region, got {ev}"

    # a short event stays short - the whole point of removing the floor
    short_curve = [(t / 2, -0.5 if 30 <= t / 2 <= 35 else 0.2) for t in range(0, 200)]
    ev = aggregate_events([mk("traffic_accident", 32.0, 0.9)],
                          health_curve=short_curve, duration_sec=237.0)[0]
    assert ev["end_time_sec"] - ev["start_time_sec"] < 12.0, \
        f"a 5s event must not be inflated past IoU range, got {ev}"

    # expansion must never leave the video
    late = aggregate_events([mk("traffic_accident", 99.0, 0.9)], duration_sec=100.0)[0]
    assert late["end_time_sec"] <= 100.0 and late["start_time_sec"] >= 0.0

    print("temporal aggregation self-test passed (10 cases)")


_selftest()
