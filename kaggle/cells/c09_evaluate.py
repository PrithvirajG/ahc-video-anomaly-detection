# =============================================================================
# 9 - Score against the public ground truth
# =============================================================================
# The real arena submission is a different file entirely (JSON, private E00x
# video set, IoU>=0.5 gate) - see cell 10. This cell is purely local diagnostics
# against the public T00x test set, which is the only ground truth we can see.
# Reported separately by level, because they are different tasks:
#   level 1  is this video anomalous, and which class      (no timestamps)
#   level 2  ...plus when it happened                      (temporal IoU)
#   level 3  ...plus a description
#
# The false-alarm rate on normal videos is printed on its own line and not
# buried inside accuracy. The PS is blunt about it - "an alerting system that
# fires regularly on ordinary activity stops being used" - so a model that wins
# on F1 by flagging everything has failed the actual brief.

def evaluate(pred: pd.DataFrame, gt: pd.DataFrame) -> dict:
    if pred.empty or gt.empty:
        print("nothing to evaluate")
        return {}

    g = gt[gt["video_id"].isin(pred["video_id"])].copy()
    truth = (g.groupby("video_id")
              .agg(is_anomaly=("is_anomaly", "max"),
                   classes=("class_name", lambda s: sorted(set(s.dropna()) - {"normal"})))
              .reset_index())
    m = pred.merge(truth, on="video_id", suffixes=("_pred", "_true"))
    if m.empty:
        print("predictions and ground truth share no video_id")
        return {}

    yp = m["is_anomaly_pred"].astype(int).to_numpy()
    yt = m["is_anomaly_true"].astype(int).to_numpy()
    tp = int(((yp == 1) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    tn = int(((yp == 0) & (yt == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    n_normal = int((yt == 0).sum())
    far = fp / max(n_normal, 1)

    # class correct only counts where an anomaly was correctly detected at all
    hit = m[(yp == 1) & (yt == 1)]
    cls_ok = int(sum(r["class_name"] in r["classes"] for _, r in hit.iterrows()))
    cls_acc = cls_ok / max(len(hit), 1)

    print("=" * 66)
    print(f"LEVEL 1   videos={len(m)}   TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  precision {prec:.3f}   recall {rec:.3f}   F1 {f1:.3f}")
    print(f"  class accuracy on correctly-detected anomalies: "
          f"{cls_acc:.3f}  ({cls_ok}/{len(hit)})")
    print(f"  FALSE ALARM RATE on normal videos: {far:.3f}  ({fp}/{n_normal})")

    # --- level 2/3: temporal ---------------------------------------------------
    # The arena's actual gate (submission PDF): an event counts ONLY when the
    # class is right AND IoU >= 0.5 - "if your interval sits inside the real
    # event it must cover at least half of it; if it swallows the real event it
    # must be no more than twice as long." At most one predicted event can match
    # a given ground-truth event; every other overlapping prediction for the
    # SAME event counts AGAINST you, not neutrally. And predicting anything at
    # all on a video that's truly normal at level 2/3 scores that video ZERO -
    # there is no partial credit for a false alarm there. Both are much harsher
    # than the plain precision/recall above, so they're broken out separately.
    timed_ids = set(g.dropna(subset=["start_time_sec", "end_time_sec"]).video_id)
    normal_ids = set(g[g["class_name"] == "normal"].video_id) - timed_ids
    l23_normal_but_flagged = 0
    for vid in normal_ids:
        p = pred[pred["video_id"] == vid]
        if not p.empty and (p.iloc[0].get("events") or []):
            l23_normal_but_flagged += 1
    if normal_ids:
        print(f"\nLEVEL 2/3 FALSE-ALARM CHECK (real videos, not level-1 pooled)")
        print(f"  normal videos where we predicted anything (scores that video "
              f"ZERO under the real rule): {l23_normal_but_flagged}/{len(normal_ids)}")

    timed = g.dropna(subset=["start_time_sec", "end_time_sec"])
    ious_loose, ious_strict, extra_fragments = [], [], 0
    if not timed.empty:
        for _, row in timed.iterrows():
            p = pred[pred["video_id"] == row["video_id"]]
            events = (p.iloc[0].get("events") or []) if not p.empty else []
            same_class = [e for e in events if e["class_name"] == row["class_name"]]

            def iou(e):
                inter = max(0.0, min(e["end_time_sec"], row["end_time_sec"])
                            - max(e["start_time_sec"], row["start_time_sec"]))
                union = (max(e["end_time_sec"], row["end_time_sec"])
                         - min(e["start_time_sec"], row["start_time_sec"]))
                return inter / max(union, 1e-6)

            scores = [iou(e) for e in same_class]
            best = max(scores) if scores else 0.0
            ious_loose.append(best)
            ious_strict.append(best >= 0.5)
            # every same-class predicted event beyond the single best match is a
            # fragment that scores against this ground-truth event, per the rule
            extra_fragments += max(0, len(scores) - 1)

        print(f"\nLEVEL 2/3 TEMPORAL   {len(ious_loose)} timed ground-truth events")
        print(f"  real gate, IoU>=0.5 AND correct class: "
              f"{sum(ious_strict)}/{len(ious_loose)} matched")
        print(f"  mean IoU among correct-class predictions (loose diagnostic, "
              f"not the real gate): {np.mean(ious_loose):.3f}")
        print(f"  extra same-class fragments beyond the best match "
              f"(count AGAINST you): {extra_fragments}")
    else:
        print("\nLEVEL 2/3 TEMPORAL   no timed ground truth in this split")

    # --- per class -----------------------------------------------------------
    print("\nper-class detection (ground truth -> predicted):")
    for cls in ANOMALY_CLASSES:
        sub = m[m["classes"].apply(lambda cs: cls in cs)]
        if sub.empty:
            continue
        got = int(sum(r["class_name"] == cls for _, r in sub.iterrows()))
        det = int(sub["is_anomaly_pred"].sum())
        print(f"  {cls:34s} n={len(sub):3d}  detected {det:3d}  correct class {got:3d}")

    return {"precision": prec, "recall": rec, "f1": f1, "class_acc": cls_acc,
            "false_alarm_rate": far,
            "mean_iou_loose": float(np.mean(ious_loose)) if ious_loose else None,
            "level23_strict_matches": int(sum(ious_strict)) if ious_loose else None,
            "level23_timed_events": len(ious_loose) if ious_loose else None,
            "level23_extra_fragments": extra_fragments if ious_loose else None,
            "level23_normal_but_flagged": l23_normal_but_flagged if normal_ids else None,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


METRICS = evaluate(PRED, GT_TEST if not GT_TEST.empty else GT_TRAIN)
(RUNS / "metrics.json").write_text(json.dumps(METRICS, indent=2))


# The arena submission is a different, stricter schema (JSON, private video
# set, null-vs-numeric timestamps by level) - built and validated in cell 10.
print("\n(arena submission file is built in cell 10, not here - "
      "different schema, different video set)")
