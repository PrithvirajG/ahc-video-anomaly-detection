# =============================================================================
# 11 - See it, don't just read the JSON
# =============================================================================
# Two controlled outputs, not a dump of everything:
#
#   GALLERY_VIDEO_IDS  - stills, one per video, shown individually. Edit this
#                        list to whichever videos you want to eyeball - only
#                        these get an image, nothing is auto-selected.
#
#   LIVE_CHECK_VIDEO_ID - ONE video gets replayed frame-by-frame with the
#                        motion gate, health score and any final alert overlaid
#                        over time, written out as a real mp4 and played inline.
#                        Re-samples and re-scores that one video (stage 0+1
#                        only - no extra VLM calls, the alert overlay reuses
#                        PRED's already-decided events). Costs seconds, not
#                        the several minutes a full re-run would.

import matplotlib.pyplot as plt
from IPython.display import Video, display


def grab_frame_at(path, t_sec):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(t_sec * fps)))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def put_text(frame, lines, origin=(10, 30), color=(0, 0, 255), scale=0.7):
    out = frame
    x, y = origin
    for line in lines:
        cv2.putText(out, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += int(28 * scale / 0.7)
    return out


def gt_label(video_id: str) -> str:
    # "unknown", not "normal". On the eval pack every class_name is NA, and the
    # old code's `else "normal"` would have quietly asserted that all 28 videos
    # are normal - colouring every prediction red as a false positive and
    # inverting the meaning of the whole gallery.
    if not HAS_TRUTH:
        return "unknown (no truth in this pack)"
    rows = GT_TEST[GT_TEST.video_id == video_id]
    if rows.empty:
        return "no ground truth"
    cls = sorted(set(rows.class_name.dropna()) - {"normal"})
    return ", ".join(cls) if cls else "normal"


def pred_row(video_id: str) -> dict | None:
    hits = [r for r in PRED.to_dict("records") if r["video_id"] == video_id]
    return hits[0] if hits else None


# --- controlled still gallery --------------------------------------------
# Edit freely. Nothing outside this list gets rendered.
# Practice ids are T0xx and eval ids are E0xx, so a single hard-coded list
# renders an empty gallery in one of the two modes. Both are spelled out; edit
# whichever applies. In eval mode this deliberately picks one video per level
# plus the longest, because with cell 9 unable to score anything these frames
# are the only check on the run before it is submitted.
GALLERY_VIDEO_IDS = (["E003", "E017", "E021", "E025", "E028"] if MODE == "eval"
                     else ["T005", "T012", "T016", "T009", "T033"])
GALLERY_VIDEO_IDS = [v for v in GALLERY_VIDEO_IDS if v in VIDEO_PATHS]

for vid in GALLERY_VIDEO_IDS:
    path = VIDEO_PATHS.get(vid)
    if path is None:
        print(f"! {vid} not found, skipping"); continue
    row = pred_row(vid) or {}
    events = row.get("events") or []
    if events:
        ev = max(events, key=lambda e: e["peak_confidence"])
        t = (ev["start_time_sec"] + ev["end_time_sec"]) / 2
        pred_txt = f"pred: {ev['class_name']} ({ev['confidence']:.2f})"
        # blue = "we cannot say if this is right"; green/red only where truth exists
        color = ((0, 140, 200) if not HAS_TRUTH else
                 (0, 180, 0) if ev["class_name"] in gt_label(vid) else (0, 0, 255))
    else:
        gt_rows = GT_TEST[GT_TEST.video_id == vid]
        t = (gt_rows.iloc[0].start_time_sec if not gt_rows.empty
             and pd.notna(gt_rows.iloc[0].start_time_sec) else video_duration(path) / 2)
        # This one does not crash on an all-NA column - pandas returns an empty
        # frame - but "not missed" would then be an assertion we cannot support,
        # so the truthless case short-circuits rather than relying on that.
        missed = HAS_TRUTH and vid in set(
            GT_TEST[GT_TEST.is_anomaly == True].video_id)
        pred_txt = "pred: normal" + (" (missed)" if missed else "")
        color = (0, 140, 200) if not HAS_TRUTH else (
            (0, 140, 255) if missed else (0, 180, 0))

    frame = grab_frame_at(path, t)
    if frame is None:
        print(f"! {vid} frame grab failed, skipping"); continue
    frame = put_text(frame, [vid, pred_txt, f"truth: {gt_label(vid)}"], color=color)

    plt.figure(figsize=(6, 4.2))
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title(vid)
    plt.axis("off")
    out_path = RUNS / "gallery" / f"{vid}_{RUN_ID}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.show()
    print(f"wrote {out_path}")


# --- one video, played back with the pipeline's own reasoning overlaid ------
LIVE_CHECK_VIDEO_ID = "E028" if MODE == "eval" else "T033"


def make_live_check_video(video_id: str, cfg=None) -> Path | None:
    cfg = cfg or CFG
    path = VIDEO_PATHS.get(video_id)
    if path is None:
        print(f"! {video_id} not found"); return None

    row = pred_row(video_id) or {}
    events = row.get("events") or []
    thresh = cfg.health_thresh if cfg.health_thresh is not None else -0.4

    kept, n_seen = sample_video(path, cfg)
    if not kept:
        print(f"! nothing survived the motion gate for {video_id}"); return None
    imgs = [to_pil(draw_visual_prompt(k["frame"], k["box"], cfg.visual_prompt)) for k in kept]
    health_scores = health(embed_images(imgs)).tolist()

    h, w = kept[0]["frame"].shape[:2]
    out_path = run_path(f"live_check_{video_id}.mp4")
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             max(cfg.sample_fps, 2.0), (w, h))

    for k, hscore in zip(kept, health_scores):
        t = k["t"]
        frame = draw_visual_prompt(k["frame"], k["box"], cfg.visual_prompt).copy()
        escalated = hscore < thresh
        lines = [f"t={t:6.1f}s", f"health={hscore:+.3f}" + ("  ESCALATED" if escalated else "")]
        active = next((e for e in events if e["start_time_sec"] <= t <= e["end_time_sec"]), None)
        color = (0, 0, 255) if escalated else (0, 200, 0)
        if active:
            lines.append(f"ALERT: {active['class_name']} ({active['confidence']:.2f})")
            color = (0, 0, 255)
        frame = put_text(frame, lines, color=color)
        writer.write(frame)
    writer.release()

    print(f"{video_id}: {len(kept)} frames, {len(events)} final event(s), wrote {out_path}")
    return out_path


live_path = make_live_check_video(LIVE_CHECK_VIDEO_ID)
if live_path is not None:
    display(Video(str(live_path), embed=True, html_attributes="controls width=640"))


# --- incident timeline: the main tag AND what else was seen ------------------
# The arena JSON only carries one class per event. That is the right answer for
# scoring and the wrong answer for a person on the other end of an alert, who
# wants to know an accident happened AND that there is smoke and a crowd. The
# sub-tags exist for that reader; this is where they become visible.

def plot_incident_timeline(video_id: str):
    row = pred_row(video_id)
    if not row or not row.get("events"):
        print(f"{video_id}: no incidents to plot"); return
    dur = row.get("duration_sec") or video_duration(VIDEO_PATHS[video_id])
    events = row["events"]

    fig, ax = plt.subplots(figsize=(11, 1.1 + 0.75 * len(events)))
    ax.set_xlim(0, dur); ax.set_ylim(-0.5, len(events) - 0.5)
    ax.set_xlabel("seconds"); ax.set_yticks([])
    ax.set_title(f"{video_id} - incidents, with what else was observed", fontsize=11)

    for i, e in enumerate(events):
        s, en = e["start_time_sec"], e["end_time_sec"]
        measured = e.get("extent_source", "").startswith("measured")
        ax.barh(i, en - s, left=s, height=0.42,
                color="#c44" if measured else "#c88",
                hatch=None if measured else "//", edgecolor="#822")
        ax.text(s, i + 0.30, f"{e['class_name']}  ({e['confidence']:.2f})",
                fontsize=9, weight="bold", va="bottom")
        # extent provenance matters: a measured span is evidence, the fallback
        # is a prior, and the plot should not let those look the same
        ax.text(en + dur * 0.005, i, "measured" if measured else "fallback 10s",
                fontsize=7.5, va="center", color="#666")
        for j, sub in enumerate(e.get("sub_tags") or []):
            ax.plot([sub["first_seen_sec"]], [i - 0.22 - j * 0.1], marker="v",
                    ms=6, color="#48c")
            ax.text(sub["first_seen_sec"] + dur * 0.004, i - 0.24 - j * 0.1,
                    f"also: {sub['class_name']} ({sub['peak_confidence']:.2f})",
                    fontsize=7.5, va="center", color="#26a")

    gt_rows = GT_TEST[(GT_TEST.video_id == video_id) & GT_TEST.start_time_sec.notna()]
    for _, g in gt_rows.iterrows():
        ax.axvspan(g.start_time_sec, g.end_time_sec, color="#2a2", alpha=0.13, zorder=0)
    if len(gt_rows):
        ax.text(0.99, 1.06, "green band = ground truth", transform=ax.transAxes,
                ha="right", fontsize=8, color="#2a2")

    plt.tight_layout()
    out = RUNS / "gallery" / f"timeline_{video_id}_{RUN_ID}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.show()
    print(f"wrote {out}")

    print(f"\n{video_id} incident detail:")
    for e in events:
        print(f"  [{e['start_time_sec']:7.1f} - {e['end_time_sec']:7.1f}] "
              f"{e['class_name']:32s} conf {e['confidence']:.2f}  "
              f"({e.get('extent_source','?')})")
        if e.get("primary_reason"):
            print(f"      why this tag: {e['primary_reason']}")
        for sub in (e.get("sub_tags") or []):
            print(f"      also seen: {sub['class_name']:28s} "
                  f"conf {sub['peak_confidence']:.2f} @ {sub['first_seen_sec']:.1f}s")


for _vid in [v for v in GALLERY_VIDEO_IDS if (pred_row(v) or {}).get("events")]:
    plot_incident_timeline(_vid)

# full structure, sub-tags included, for later analysis - deliberately separate
# from the arena file, which only ever carries the single main tag per event
_inc = run_path("incidents_detailed.json")
_inc.write_text(json.dumps(
    [r for r in PRED.to_dict("records") if r.get("events")], indent=1, default=str))
print(f"\nwrote {_inc} (main tags + sub-tags + provenance)")
