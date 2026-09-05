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


def pick_frames(window: list[dict], n: int) -> list[dict]:
    """Evenly spaced across the window, so the VLM sees change rather than n
    near-duplicates from the same instant."""
    if len(window) <= n:
        return window
    idx = np.linspace(0, len(window) - 1, n).round().astype(int)
    return [window[i] for i in sorted(set(idx.tolist()))]


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

    results = []
    t_vlm0 = time.time()
    for w in windows:
        picked = pick_frames(w, cfg.vlm_frames)
        pil = [to_pil(draw_visual_prompt(r["frame"], r["box"], cfg.visual_prompt))
               for r in picked]
        emb = embed_images(pil)
        cands = shortlist_classes(emb)
        try:
            verdict = vlm_verify(pil, cands)
        except Exception as e:
            print(f"  ! vlm failed on window @{w[0]['t']:.1f}s: "
                  f"{str(e).splitlines()[0][:120]}")
            continue
        results.append({
            "t0": w[0]["t"],
            "t1": w[-1]["t"] + 1.0 / cfg.sample_fps,
            "class": verdict["class"],
            "confidence": verdict["confidence"],
            "description": verdict["description"],
            "candidates": cands,
        })
    t_vlm = time.time() - t_vlm0

    events = aggregate_events(results, cfg)
    duration = kept[-1]["t"] + 1.0 / cfg.sample_fps if kept else video_duration(path)
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
              f"win {len(windows):3d}  -> {out['class_name']:32s} "
              f"{out['realtime_factor']:5.2f}x realtime")
    return out


# --- run over the public test set --------------------------------------------
# 34 videos / ~56 min, with ground truth published, so this is the only honest
# read on whether any of the above works before the private evaluation.
def run_split(gt: pd.DataFrame, limit: int | None = None, cfg=None) -> pd.DataFrame:
    cfg = cfg or CFG
    vids = gt.drop_duplicates("video_id")[["video_id", "path"]].dropna(subset=["path"])
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
    if "sec_stage1" in df:
        print(f"  stage 1: {df['sec_stage1'].sum():.0f}s    "
              f"stage 2: {df['sec_stage2'].sum():.0f}s    "
              f"({100 * df['sec_stage2'].sum() / max(df['sec_total'].sum(), 1e-6):.0f}% "
              "of wall time in the VLM)")
    return df


# LIMIT=3 proved the wiring; the full 34-video run is now confirmed to finish
# in a few minutes, so this defaults to the whole test set.
LIMIT = None
PRED = run_split(GT_TEST if not GT_TEST.empty else GT_TRAIN, limit=LIMIT)
PRED.to_json(RUNS / "predictions_raw.json", orient="records", indent=1)
print(f"\nwrote {RUNS / 'predictions_raw.json'}")
