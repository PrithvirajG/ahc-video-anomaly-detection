# =============================================================================
# 7 - End to end: one video in, events out
# =============================================================================

def group_escalations(kept: list[dict], thresh: float, cfg=None) -> list[list[dict]]:
    """Bundle contiguous low-health frames into windows for stage 2.

    Per-frame VLM calls would be wasteful and would also throw away the temporal
    evidence the model needs - "is this vehicle stationary" is unanswerable from
    one frame. A window of a few frames spanning a couple of seconds answers it.
    """
    cfg = cfg or CFG
    gap = 2.0 / max(cfg.sample_fps, 0.1)      # allow one dropped sample inside a window
    flagged = [k for k in kept if k["health"] < thresh]
    windows, cur = [], []
    for k in flagged:
        if cur and (k["t"] - cur[-1]["t"]) > gap:
            windows.append(cur)
            cur = []
        cur.append(k)
    if cur:
        windows.append(cur)
    return windows


def add_scan_floor(windows: list[list[dict]], kept: list[dict],
                   duration: float, cfg=None) -> list[tuple[list[dict], str]]:
    """Guarantee a long video is looked at every scan_floor_interval_sec.

    Returns [(frames, source), ...] where source is "escalated" or "scan", so a
    detection can be traced back to which mechanism found it - that tag is the
    whole point of the experiment: it lets us say what the floor actually cost
    and gained, rather than watching the score move and guessing why.

    The floor only ADDS windows. Escalation is untouched, so disabling this
    returns behaviour to exactly what it was.
    """
    cfg = cfg or CFG
    out = [(w, "escalated") for w in windows]
    if not kept:
        return out

    def _never_looked():
        """Never report a video normal without the VLM having seen it once.

        Measured: 9 videos produced zero windows - no escalation, and too short
        for the interval floor - and SEVEN of those nine were genuinely
        anomalous (T007 accident, T008/T009 congestion, T010 stalled vehicle,
        T022 fighting, T023/T024 loitering). That was 41% of all our misses
        coming from videos we simply never examined. Nine extra VLM calls, ~50
        seconds, is a trivial price for removing a whole failure mode.

        The frames offered are the LOWEST-HEALTH ones available, so the single
        look gets the most suspicious moment rather than an arbitrary one.
        """
        worst = sorted(kept, key=lambda k: k["health"])[:cfg.vlm_frames]
        return [(sorted(worst, key=lambda k: k["t"]), "last-resort")]

    if not (cfg.scan_floor_enabled and duration > cfg.scan_floor_min_video_sec):
        return out or _never_looked()

    covered = [(w[0]["t"], w[-1]["t"]) for w in windows]
    step = cfg.scan_floor_interval_sec
    times = np.array([k["t"] for k in kept])

    t = 0.0
    while t < duration:
        slot_end = t + step
        # skip a slot an escalated window already covers - no point paying twice
        if any(a < slot_end and b >= t for a, b in covered):
            t = slot_end
            continue
        centre = t + step / 2
        # the frames nearest this slot's centre, in time order
        idx = np.argsort(np.abs(times - centre))[:cfg.vlm_frames]
        picked = [kept[i] for i in sorted(idx.tolist())]
        if picked and abs(picked[0]["t"] - centre) <= step:   # slot has real frames
            out.append((picked, "scan"))
        t = slot_end
    return out or _never_looked()


def pick_frames(window: list[dict], n: int) -> list[dict]:
    """Evenly spaced across the window, so the VLM sees change rather than n
    near-duplicates from the same instant."""
    if len(window) <= n:
        return window
    idx = np.linspace(0, len(window) - 1, n).round().astype(int)
    return [window[i] for i in sorted(set(idx.tolist()))]


def widen_window(kept: list[dict], centre_t: float, cfg=None) -> list[dict]:
    """cfg.vlm_frames frames spread over cfg.vlm_span_sec, centred on centre_t.

    An escalation window is ~2s wide because that is how often we sample, not
    because incidents last 2s. Handing the VLM only those frames is why it
    describes aftermath: a collision is over in about a second, and every later
    frame shows a queue. Widening to the probe's 16s gives both stages the same
    width of evidence and covers the median real event.

    Falls back to the window's own frames when the video is too short to widen,
    so a 6s L1 clip behaves exactly as before.
    """
    cfg = cfg or CFG
    span = float(getattr(cfg, "vlm_span_sec", 0.0) or 0.0)
    n = cfg.vlm_frames
    if span <= 0 or not kept:
        return []
    near = [k for k in kept if abs(k["t"] - centre_t) <= span / 2]
    if len(near) < 2:
        return []
    if len(near) <= n:
        return near
    idx = np.linspace(0, len(near) - 1, n).round().astype(int)
    return [near[i] for i in sorted(set(idx.tolist()))]


def _runtime_stats_since(model_name: str, start_idx: int) -> dict | None:
    """Slice CALL_LOG since this video started, for the arena's model_runtimes.

    Snapshotting the starting index rather than clearing CALL_LOG keeps a full
    run-long history intact (useful for our own diagnostics) while still giving
    an accurate per-video breakdown - the two uses don't conflict.
    """
    times = CALL_LOG.get(model_name, [])[start_idx:]
    if not times:
        return None
    arr = np.array(times)
    return {
        "model_name": model_name,
        "call_count": len(arr),
        "total_time_ms": round(float(arr.sum()), 1),
        "average_time_ms": round(float(arr.mean()), 1),
        "p50_time_ms": round(float(np.percentile(arr, 50)), 1),
        "p95_time_ms": round(float(np.percentile(arr, 95)), 1),
        "max_time_ms": round(float(arr.max()), 1),
    }


def process_video(path, cfg=None, verbose=False) -> dict:
    cfg = cfg or CFG
    _log_start = {k: len(v) for k, v in CALL_LOG.items()}
    t_start = time.time()
    kept, n_seen = stage1_video(path, cfg)
    t_stage1 = time.time() - t_start

    thresh = cfg.health_thresh
    if thresh is None and kept:
        # No calibration available: fall back to a within-video percentile. Worse
        # than calibrating on known-normal footage, because a video that is
        # anomalous throughout still escalates only escalate_pct of itself.
        thresh = float(np.percentile([k["health"] for k in kept], cfg.escalate_pct))

    windows = group_escalations(kept, thresh, cfg) if kept else []
    duration_full = video_duration(path)
    to_look_at = add_scan_floor(windows, kept, duration_full, cfg)

    # One capture reused across every window of this video - reopening the file
    # per frame would cost more than the seeks themselves.
    _native_cap = (cv2.VideoCapture(str(path))
                   if getattr(cfg, "vlm_crop_to_motion", False) else None)

    results = []
    t_vlm0 = time.time()
    for w, source in to_look_at:
        centre_t = (w[0]["t"] + w[-1]["t"]) / 2
        # Widen to cfg.vlm_span_sec where the video allows it, else fall back to
        # the escalation window itself. `widened` is truthy only when the VLM
        # actually judged the wider span, which is what makes the event extent
        # below an honest claim rather than an assumed one.
        widened = widen_window(kept, centre_t, cfg)
        picked = widened or pick_frames(w, cfg.vlm_frames)
        # Crop to the motion region at NATIVE resolution where we can. The
        # stored frame is already downscaled to max_side, which on 1280x720
        # source throws away 75% of the pixels - and the evidence that separates
        # a collision from the queue behind it lives in those pixels. One seek
        # per frame stage 2 actually looks at, about 39s across a five-video
        # run, and the token count does not change because the crop is
        # downscaled only if it is still larger than max_side.
        pil = []
        for r in picked:
            crop = native_crop(_native_cap, r["t"], r["box"],
                               (r["frame"].shape[1], r["frame"].shape[0]), cfg)
            if crop is not None:
                pil.append(to_pil(crop))      # already centred on the motion
            else:
                pil.append(to_pil(draw_visual_prompt(r["frame"], r["box"],
                                                     cfg.visual_prompt)))
        emb = embed_images(pil)
        # The probe sees 16s around this moment, not the ~2s the VLM sees, and
        # it costs nothing: stage 1 already encoded these frames and cell 5 now
        # keeps the vectors. Scored BEFORE the VLM so its shortlist can steer
        # the question, and kept afterwards so it can contradict the answer.
        pw = probe_window(kept, centre_t, cfg) if "probe_window" in globals() else {}
        p_anom = float(1.0 - pw.get("normal", 1.0)) if pw else 0.0
        cands = shortlist_classes(emb)
        try:
            verdict = vlm_verify(pil, cands)
        except Exception as e:
            print(f"  ! vlm failed on window @{w[0]['t']:.1f}s: "
                  f"{str(e).splitlines()[0][:120]}")
            continue

        # --- the probe may overrule a "normal" verdict, and only that ---------
        # Stage 2 answered "normal" on 79% of the windows that overlapped a real
        # event, including all 28 windows on T025's six accidents and T032's
        # four loitering events - with every class on offer. The probe reaches
        # 100% held-out recall on loitering and 85% on congestion, so where it
        # is confident and stage 2 has abstained, silence is the worse answer.
        #
        # One direction only. The probe never overrides a POSITIVE call, because
        # stage 2 looked at pixels and the probe looked at a mean of embeddings,
        # and it never fires below probe_override_p - measured at 0.95 that is
        # 353 of 484 held-out anomalies caught with zero false positives on 75
        # held-out normal clips. Held-out training clips are not 240-second test
        # videos from other cameras, so this is deliberately stricter than the
        # escalation bar and switchable from CFG.
        overrode = False
        top3 = sorted(((c, v) for c, v in pw.items() if c != "normal"),
                      key=lambda kv: -kv[1])[:3] if pw else []
        if (getattr(cfg, "probe_override_enabled", False) and top3
                and verdict["class"] == "normal"
                and p_anom >= getattr(cfg, "probe_override_p", 0.95)):
            chosen, conf = top3[0][0], float(p_anom)
            # The probe decides WHETHER; the VLM decides WHICH. Measured on
            # T025: the probe localises five of six real accidents at IoU 0.800
            # and calls every one of them wrong_way_driving. Its top-1 is 0.739
            # against 0.884 for its top-3, so re-ask with those three and no
            # "normal" - the question it failed at was never "is anything
            # happening", it was "which of these is it".
            # Widen the HINTS as well: the probe top-3 was 91% confident and wrong
            # on T025, so give the VLM the probe ranking plus stage 1's own,
            # deduplicated. The schema already offers all eleven either way.
            hints = list(dict.fromkeys([c for c, _ in top3] + list(cands)))[:6]
            pick = vlm_pick_class(pil, hints)
            if pick:
                chosen = pick["class"]
                conf = max(0.5, min(float(pick.get("confidence", conf)), conf))
            verdict = {**verdict, "class": chosen, "anomaly": True,
                       "confidence": round(conf, 3),
                       "description": (pick or {}).get("description")
                       or verdict.get("description", "")
                       or f"probe: {chosen.replace('_', ' ')}"}
            overrode = True

        results.append({
            "t0": w[0]["t"],
            "t1": w[-1]["t"] + 1.0 / cfg.sample_fps,
            "class": verdict["class"],
            "confidence": verdict["confidence"],
            "description": verdict["description"],
            "candidates": cands,
            "source": source,          # "escalated" | "scan" - the experiment's
                                       # whole point: traceable back to mechanism
            "health": float(np.mean([r["health"] for r in picked])),
            # both signals stored side by side, so the next question - which of
            # these two was right, and where - is answerable offline
            "probe_anomaly": round(p_anom, 4),
            "probe_top": top3[0][0] if top3 else None,
            "probe_top3": [[c, round(v, 4)] for c, v in top3],
            "probe_override": overrode,
            # What interval does the evidence actually cover? Stage 2 looked at
            # ~2s; the probe looked at PROBE_SPAN_SEC. When the probe is what
            # fired, the honest claim is its span, and that is worth a great
            # deal: probe-span extents score IoU 0.800 against five of T025's
            # six real 20s events, where a 2s window buffered to 6s scores 0.30
            # and fails the gate no matter how right the class is.
            # Set whenever the VLM actually judged the wider span - not only on
            # an override. If the model looked at 16s to reach its verdict, 16s
            # is the interval that verdict covers, whichever stage said it.
            "span": ([round(picked[0]["t"], 2),
                      round(picked[-1]["t"] + 1.0 / cfg.sample_fps, 2)]
                     if widened else None),
        })
    if _native_cap is not None:
        _native_cap.release()
    t_vlm = time.time() - t_vlm0
    n_scan = sum(1 for _, s in to_look_at if s == "scan")

    # The CONTAINER duration, not "wherever the last surviving frame landed".
    # Those differ by up to 2.7s on the practice pack, always short, because the
    # motion gate can drop the tail of a video - measured against the arena's own
    # manifest: T025 237.6 vs 240.0, T033 626.1 vs 628.8. It mattered little while
    # a 180s cap kept every event away from the end; with that cap gone an event
    # can legitimately run to the final frame, and clamping it to a duration 2.7s
    # short trims real overlap off exactly the long D3 events that pay 5 marks
    # each. It is also what cell 10 falls back to when no manifest file is
    # present, which on the practice pack is always.
    duration = duration_full if duration_full else (
        kept[-1]["t"] + 1.0 / cfg.sample_fps if kept else 0.0)

    # The health curve is what lets aggregation MEASURE an event's extent rather
    # than assume it - it is computed per frame in stage 1 and was previously
    # thrown away after the escalate/skip decision.
    curve = [(k["t"], k["health"]) for k in kept] if kept else None

    def _adjudicate(cluster):
        """One extra VLM call to name the primary incident when a cluster holds
        several classes. Deliberately not a causal lookup table: smoke is a
        symptom over a wrecked car and the incident itself over a thermal plant,
        so the call belongs to the model that can see which one this is."""
        frames = []
        for w in cluster:
            near = [k for k in kept if w["t0"] <= k["t"] <= w["t1"]]
            if near:
                r = near[len(near) // 2]
                frames.append(to_pil(draw_visual_prompt(r["frame"], r["box"],
                                                        cfg.visual_prompt)))
        return adjudicate_primary(frames[:cfg.vlm_frames], cluster) if frames else None

    events = aggregate_events(results, cfg, health_curve=curve,
                              duration_sec=video_duration(path),
                              adjudicator=_adjudicate)
    wall = time.time() - t_start

    # Arena schema's per-video runtime block - required on every video, and the
    # only source of the latency bonus. end_to_end_internal_time_ms starts here,
    # after models are already loaded, matching the rule to exclude load/download
    # time. chunks_processed has no exact spec meaning for our design; mapped to
    # "how many discrete windows needed the heavier model", floored at 1 for a
    # video that never escalated but still had a full stage-0/1 pass.
    model_runtimes = [s for s in (
        _runtime_stats_since("siglip2-encoder", _log_start.get("siglip2-encoder", 0)),
        _runtime_stats_since("vision-language-model",
                             _log_start.get("vision-language-model", 0)),
    ) if s is not None]

    out = {
        "video_id": Path(path).stem,
        "duration_sec": round(duration, 2),
        "frames_sampled": n_seen,
        "frames_kept": len(kept),
        "windows_escalated": len(windows),
        "windows_scan_floor": n_scan,
        # raw per-window verdicts, kept so aggregation can be re-tuned offline in
        # seconds instead of an 8-minute GPU re-run per experiment
        "window_verdicts": results,
        # ...and the curve those verdicts were measured against. Without it an
        # offline replay can reproduce the CLUSTERING but not the EXTENT, since
        # both now consult the curve - which made the last round of sweeps
        # unable to test the thing they were sweeping. ~600 floats per video,
        # rounded to keep the JSON readable.
        "health_curve": [[round(t, 2), round(h, 4)] for t, h in (curve or [])],
        "escalation_rate": round(len(windows) and sum(len(w) for w in windows)
                                 / max(len(kept), 1) or 0.0, 4),
        "events": events,
        "is_anomaly": int(bool(events)),
        "class_name": (max(events, key=lambda e: e["peak_confidence"])["class_name"]
                       if events else "normal"),
        "sec_stage1": round(t_stage1, 2),
        "sec_stage2": round(t_vlm, 2),
        "sec_total": round(wall, 2),
        "realtime_factor": round(duration / wall, 2) if wall > 0 else 0.0,
        "runtime_metadata": {
            "frames_processed": n_seen,
            "chunks_processed": max(1, len(windows)),
            "end_to_end_internal_time_ms": round(wall * 1000, 1),
            "model_runtimes": model_runtimes,
        },
    }
    if verbose:
        print(f"{out['video_id']:16s} {duration:6.1f}s  kept {len(kept):4d}/{n_seen:4d}  "
              f"esc {len(windows):3d} scan {n_scan:3d}  -> {out['class_name']:32s} "
              f"{out['realtime_factor']:5.2f}x realtime")
    return out


# --- run over the public test set --------------------------------------------
# 34 videos / ~56 min, with ground truth published, so this is the only honest
# read on whether any of the above works before the private evaluation.
def run_split(gt: pd.DataFrame, limit: int | None = None, cfg=None,
              only: list[str] | None = None) -> pd.DataFrame:
    cfg = cfg or CFG
    vids = gt.drop_duplicates("video_id")[["video_id", "path"]].dropna(subset=["path"])
    if only:
        want = list(dict.fromkeys(only))
        vids = vids[vids.video_id.isin(want)]
        missing = [v for v in want if v not in set(vids.video_id)]
        if missing:
            print(f"! requested but not found: {missing}")
        print(f"subset: {len(vids)} of {len(want)} requested videos")
    if limit:
        vids = vids.head(limit)
    rows, t0 = [], time.time()
    for i, (_, r) in enumerate(vids.iterrows(), 1):
        print(f"[{i}/{len(vids)}] ", end="")
        try:
            rows.append(process_video(r["path"], cfg, verbose=True))
        except Exception as e:
            print(f"FAILED {r['video_id']}: {str(e).splitlines()[0][:140]}")
            rows.append({"video_id": r["video_id"], "is_anomaly": 0,
                         "class_name": "normal", "events": [], "error": str(e)[:200]})
    df = pd.DataFrame(rows)
    total_video = df.get("duration_sec", pd.Series(dtype=float)).sum()
    print(f"\n{len(df)} videos, {total_video / 60:.1f} min of footage "
          f"in {(time.time() - t0) / 60:.1f} min wall "
          f"({total_video / max(time.time() - t0, 1e-6):.2f}x realtime)")
    # --- what did the scan floor actually buy? -------------------------------
    # The number this experiment turns on. Without it we would see the score move
    # and be guessing which mechanism moved it.
    if "window_verdicts" in df:
        esc = scan = esc_hit = scan_hit = 0
        for _, r in df.iterrows():
            for v in (r.get("window_verdicts") or []):
                is_scan = v.get("source") == "scan"
                scan += is_scan
                esc += not is_scan
                if v["class"] != "normal":
                    scan_hit += is_scan
                    esc_hit += not is_scan
        print(f"  windows: {esc} escalated ({esc_hit} non-normal), "
              f"{scan} scan-floor ({scan_hit} non-normal)")
        if "windows_scan_floor" in df:
            covered = int((df["windows_scan_floor"] > 0).sum())
            print(f"  scan floor active on {covered} video(s)")

    if "sec_stage1" in df:
        print(f"  stage 1: {df['sec_stage1'].sum():.0f}s    "
              f"stage 2: {df['sec_stage2'].sum():.0f}s    "
              f"({100 * df['sec_stage2'].sum() / max(df['sec_total'].sum(), 1e-6):.0f}% "
              "of wall time in the VLM)")
    return df


# --- the iteration set -------------------------------------------------------
# Five hand-picked videos, ~23 min of footage, ~8 min of GPU. A full 34-video
# run takes 21 minutes, which is too slow to think with; these five were chosen
# to make each Tier-1 change either visibly work or visibly fail.
#
#   T025  D2  238s  6x traffic_accident @20s      12 windows, ALL said "normal"
#                   -> the answer-menu bug, six independent chances to see it lift
#   T032  D3  308s  4x loitering (2.6-37.6s)      16 windows, ALL said "normal"
#                   -> the class we have never once scored, and D3 pays 5 marks
#   T031  D3  360s  1x traffic_congestion 235-360 18 windows, congestion FOUND
#                   -> the one event aggregation can win: oracle IoU 0.812, we
#                      scored 0 by emitting 9-311s. Tests the de-hardcoding.
#   T026  D2  238s  4x mixed classes              CONTROL - we already match one
#                   -> regression detector, and four different classes at once
#   T030  D2  239s  NORMAL                        CONTROL - false alarms
#                   -> guards the 100% D2 precision that item 1 puts at risk
#
# Baseline over these five: 1 of 15 ground-truth events matched (T026's spill).
# Set to None for the full set once a change looks right.
ONLY_VIDEOS = ["T025", "T026", "T030", "T031", "T032"]

LIMIT = None
PRED = run_split(GT_TEST if not GT_TEST.empty else GT_TRAIN,
                 limit=LIMIT, only=ONLY_VIDEOS)
PRED.to_json(RUNS / "predictions_raw.json", orient="records", indent=1)
print(f"\nwrote {RUNS / 'predictions_raw.json'}")
