# =============================================================================
# 11 - See it, don't just read the JSON
# =============================================================================
# A grid of actual frames: one per detected event (predicted class + confidence
# vs ground truth), plus a few genuinely missed anomalies for honest contrast.
# This is also the fastest source of "example frames, before/after comparisons"
# the submission's architecture write-up and 2-slide PPT are asked for.

import matplotlib.pyplot as plt


def grab_frame_at(path, t_sec):
    """One seek-and-read. Fine for a dozen diagnostic grabs; the main pipeline
    avoids seeking (cell 4) because it decodes thousands of frames, not a dozen."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(t_sec * fps)))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def annotate(frame, lines, color=(0, 0, 255)):
    out = frame.copy()
    y = 30
    for line in lines:
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA)
        y += 28
    return out


def gt_label(video_id: str) -> str:
    rows = GT_TEST[GT_TEST.video_id == video_id]
    if rows.empty:
        return "no ground truth"
    cls = sorted(set(rows.class_name.dropna()) - {"normal"})
    return ", ".join(cls) if cls else "normal"


# --- what we actually predicted ----------------------------------------------
detected = [r for r in PRED.to_dict("records") if r.get("events")]
gallery = []
for row in detected:
    ev = max(row["events"], key=lambda e: e["peak_confidence"])
    path = VIDEO_PATHS.get(row["video_id"])
    if path is None:
        continue
    t = (ev["start_time_sec"] + ev["end_time_sec"]) / 2
    frame = grab_frame_at(path, t)
    if frame is None:
        continue
    lines = [row["video_id"], f"pred: {ev['class_name']} ({ev['confidence']:.2f})",
            f"truth: {gt_label(row['video_id'])}"]
    correct = ev["class_name"] == gt_label(row["video_id"]) or \
        ev["class_name"] in gt_label(row["video_id"])
    gallery.append((row["video_id"], annotate(frame, lines,
                                              (0, 180, 0) if correct else (0, 0, 255))))

# --- a few misses, for honest contrast -----------------------------------------
detected_ids = {r["video_id"] for r in detected}
missed_ids = sorted(set(GT_TEST[GT_TEST.is_anomaly == True].video_id) - detected_ids)
n_missed_total = len(missed_ids)
for vid in missed_ids[:4]:
    path = VIDEO_PATHS.get(vid)
    row = GT_TEST[GT_TEST.video_id == vid].iloc[0]
    t = row.start_time_sec if pd.notna(row.start_time_sec) else video_duration(path) / 2
    frame = grab_frame_at(path, t)
    if frame is None:
        continue
    gallery.append((vid, annotate(frame, [vid, "pred: normal (missed)",
                                          f"truth: {row.class_name}"], (0, 140, 255))))

# --- lay it out ----------------------------------------------------------------
n = len(gallery)
cols = min(4, n) or 1
rows_n = -(-n // cols)
fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 3.2 * rows_n))
axes = np.array(axes).reshape(-1)
for ax, (vid, img) in zip(axes, gallery):
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(vid, fontsize=9)
    ax.axis("off")
for ax in axes[len(gallery):]:
    ax.axis("off")
plt.tight_layout()

out_path = RUNS / "detection_gallery.png"
plt.savefig(out_path, dpi=110, bbox_inches="tight")
plt.show()
print(f"\n{len(detected)} detected shown, {min(4, n_missed_total)} of "
      f"{n_missed_total} missed anomalies shown for contrast")
print(f"wrote {out_path} - use it in the architecture write-up / 2-slide PPT")
