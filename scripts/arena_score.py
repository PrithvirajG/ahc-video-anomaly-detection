"""Score a predictions_raw JSON the way the arena does, locally.

The arena reports, per difficulty, four numbers plus a mark: precision, recall,
found/total, and false alarms. Those definitions were recovered from our own
leaderboard row and check out exactly:

    D1  found 7/17  FA 2  -> P 78% = 7/(7+2),  R 41% = 7/17
    D2  found 2/12  FA 0  -> P 100%,           R 17% = 2/12
    D3  found 0/6   FA 6  -> P 0%,             R 0%

so  P = found / (found + FA)   and   R = found / total_truth_events.

D1 is scored per VIDEO (is it anomalous, and which class - no timings).
D2 and D3 are scored per EVENT, and an event counts only when the class is
right AND temporal IoU >= 0.5. Every other overlapping prediction for that same
event is a false alarm, not a neutral extra.

The marks estimate comes from a least-squares fit over the 28 public
leaderboard rows and is APPROXIMATE - it reproduces D1 to R2 0.85 and D3 to
0.73, but only 0.32 on D2, where a large constant (~19 marks) appears to be
credit for correctly leaving normal videos alone. Trust found/FA/P/R; treat the
marks column as a rough guide, never as a target to optimise directly.

    python scripts/arena_score.py <predictions_raw.json> [--gt path] [--timeline]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GT = ROOT / "data" / "test" / "ground_truth.csv"

# fitted on the 28 public leaderboard rows: marks ~ a + b*found + c*log1p(FA)
FIT = {1: (3.46, 1.005, +1.047), 2: (19.37, 2.135, -0.778), 3: (9.09, 4.994, -1.569)}
CAP = {1: 25.0, 2: 35.0, 3: 40.0}
FLOOR = 2.0   # observed: every 0-found D3 row bottoms out at 2.0


def iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def load_gt(path: Path) -> pd.DataFrame:
    gt = pd.read_csv(path)
    gt["is_anomaly"] = gt["is_anomaly"].astype(str).str.lower().eq("true")
    for c in ("start_time_sec", "end_time_sec"):
        gt[c] = pd.to_numeric(gt[c], errors="coerce")
    return gt


def score_d1(pred: dict[str, dict], gt: pd.DataFrame) -> dict:
    """Per-video: was the video anomalous, and did we name the right class?"""
    v = gt.groupby("video_id").agg(
        anom=("is_anomaly", "max"),
        classes=("class_name", lambda s: set(s.dropna()) - {"normal"}))
    found = fa = missed = tn = wrong_class = 0
    detail = []
    for vid, row in v.iterrows():
        p = pred.get(vid)
        if p is None:
            continue
        said_anom = bool(p.get("is_anomaly"))
        said_cls = p.get("class_name", "normal")
        if row.anom and said_anom:
            if said_cls in row.classes:
                found += 1; detail.append((vid, "found", said_cls))
            else:
                wrong_class += 1; fa += 1
                detail.append((vid, "wrong-class", f"{said_cls} != {sorted(row.classes)}"))
        elif row.anom and not said_anom:
            missed += 1; detail.append((vid, "missed", sorted(row.classes)))
        elif not row.anom and said_anom:
            fa += 1; detail.append((vid, "false-alarm", said_cls))
        else:
            tn += 1
    total = int(v.anom.sum())
    return dict(found=found, total=total, fa=fa, missed=missed, tn=tn,
                wrong_class=wrong_class, detail=detail)


def score_timed(pred: dict[str, dict], gt: pd.DataFrame, level: int) -> dict:
    """Per-event, class-correct AND IoU >= 0.5. One prediction per truth event."""
    g = gt[(gt.level == level) & gt.start_time_sec.notna()]
    normal_ids = set(gt[(gt.level == level) & (gt.class_name == "normal")].video_id)
    found = fa = wrongclass = 0
    per_video, matched_events = [], []
    for vid in sorted(set(g.video_id) | normal_ids):
        truth = g[g.video_id == vid]
        evs = list((pred.get(vid) or {}).get("events") or [])
        used = set()
        v_found = v_wrong = 0
        for _, t in truth.iterrows():
            best_i, best_iou = None, 0.0
            for i, e in enumerate(evs):
                if i in used or e["class_name"] != t.class_name:
                    continue
                s = iou(e["start_time_sec"], e["end_time_sec"],
                        t.start_time_sec, t.end_time_sec)
                if s > best_iou:
                    best_i, best_iou = i, s
            if best_i is not None and best_iou >= 0.5:
                used.add(best_i); found += 1; v_found += 1
                matched_events.append((vid, t.class_name, best_iou))
            else:
                # did anything overlap at all, just with the wrong class or extent?
                if any(iou(e["start_time_sec"], e["end_time_sec"],
                           t.start_time_sec, t.end_time_sec) > 0 for e in evs):
                    v_wrong += 1
        # The arena's timeline splits unmatched predictions into two colours,
        # and the leaderboard folds them back together. Both are reported here
        # because reconciling our numbers against the arena needs both views:
        #
        #   violet "wrong class"  - overlaps a real event, named it wrongly
        #   red    "false alarm"  - matched nothing, INCLUDING a right-class
        #                           prediction whose IoU missed the 0.5 gate
        #                           (T031: congestion 9-311s over a real
        #                           235-360s congestion is red, not violet)
        #
        # Verified against the practice timeline for arena_submission (2):
        # arena reported 2 matched, 2 wrong-class, 5 false-alarm; this scorer
        # produces 2 matched and 7 unmatched, and 2 + 5 = 7.
        v_wrongclass = v_fa = 0
        for i, e in enumerate(evs):
            if i in used:
                continue
            overlapping = [t for _, t in truth.iterrows()
                           if iou(e["start_time_sec"], e["end_time_sec"],
                                  t.start_time_sec, t.end_time_sec) > 0]
            if overlapping and all(e["class_name"] != t.class_name for t in overlapping):
                v_wrongclass += 1
            else:
                v_fa += 1
        unmatched = len(evs) - len(used)
        fa += unmatched
        wrongclass += v_wrongclass
        per_video.append(dict(video_id=vid, level=level,
                              truth=len(truth), pred=len(evs),
                              found=v_found, near_miss=v_wrong,
                              wrongclass=v_wrongclass, fa=v_fa,
                              unmatched=unmatched,
                              is_normal=vid in normal_ids))
    return dict(found=found, total=len(g), fa=fa, wrongclass=wrongclass,
                per_video=per_video, matched=matched_events)


def marks(level: int, found: int, fa: int) -> float:
    a, b, c = FIT[level]
    m = a + b * found + c * math.log1p(fa)
    return max(FLOOR, min(CAP[level], m))


def bar(truth_spans, pred_spans, duration, width=46) -> tuple[str, str]:
    """Two ASCII rows, the arena's timeline in a terminal."""
    def render(spans, ch):
        row = ["."] * width
        for s, e in spans:
            i0 = max(0, min(width - 1, int(width * s / max(duration, 1e-6))))
            i1 = max(i0, min(width - 1, int(width * e / max(duration, 1e-6))))
            for i in range(i0, i1 + 1):
                row[i] = ch
        return "".join(row)
    return render(truth_spans, "#"), render(pred_spans, "o")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--gt", default=str(DEFAULT_GT))
    ap.add_argument("--timeline", action="store_true",
                    help="print the per-video ASCII timeline")
    args = ap.parse_args()

    gt = load_gt(Path(args.gt))
    raw = json.loads(Path(args.predictions).read_text())
    pred = {r["video_id"]: r for r in raw}

    print("=" * 78)
    print(f"{Path(args.predictions).name}   vs   {Path(args.gt).name}")
    print(f"{len(pred)} predicted videos, {gt.video_id.nunique()} in ground truth, "
          f"{len(set(pred) & set(gt.video_id))} overlapping")
    print("=" * 78)

    d1 = score_d1(pred, gt[gt.level == 1])
    total = 0.0
    rows = []
    p = d1["found"] / max(d1["found"] + d1["fa"], 1)
    r = d1["found"] / max(d1["total"], 1)
    m = marks(1, d1["found"], d1["fa"])
    total += m
    rows.append(("D1", m, CAP[1], p, r, d1["found"], d1["total"], d1["fa"]))

    timed = {}
    for lv in (2, 3):
        s = score_timed(pred, gt, lv)
        timed[lv] = s
        p = s["found"] / max(s["found"] + s["fa"], 1)
        r = s["found"] / max(s["total"], 1)
        m = marks(lv, s["found"], s["fa"])
        total += m
        rows.append((f"D{lv}", m, CAP[lv], p, r, s["found"], s["total"], s["fa"]))

    print(f"{'':4} {'marks':>12}  {'P':>6} {'R':>6}  {'found':>9}  {'FA':>5}")
    for name, m, cap, p, r, f, t, fa in rows:
        print(f"{name:4} {m:6.1f} /{cap:5.0f}  {100*p:5.0f}% {100*r:5.0f}%  "
              f"{f:4d}/{t:<4d}  {fa:5d}")
    print(f"{'':4} {total:6.1f} / 100.0   <- estimated, see the docstring")
    print()
    print("  P/R use the LEADERBOARD convention: FA counts every unmatched")
    print("  prediction, wrong-class ones included. Verified against the arena's")
    print("  own practice timeline - it reported 2 matched / 2 wrong-class /")
    print("  5 false-alarm where this reports 2 matched / 7 FA, and 2+5 = 7.")
    if not timed[2]["per_video"] or any(pv["video_id"] == "T028"
                                        for pv in timed[2]["per_video"]):
        print()
        print("  ! D2 denominator differs from the arena by design: our CSV gives")
        print("    T028 four events and the arena's practice timeline omits T028")
        print("    entirely (14 D2 events there, 18 here). No corrected CSV exists,")
        print("    so D2 recall here reads ~4 events pessimistic against the arena.")

    print()
    print("-" * 78)
    print("D1 detail (video level)")
    print("-" * 78)
    print(f"  found {d1['found']}   wrong-class {d1['wrong_class']}   "
          f"missed {d1['missed']}   false alarms {d1['fa']}   correct-normal {d1['tn']}")
    for vid, kind, extra in d1["detail"]:
        if kind != "found":
            print(f"    {vid}  {kind:12} {extra}")

    for lv in (2, 3):
        s = timed[lv]
        print()
        print("-" * 78)
        print(f"D{lv} detail (event level, class + IoU>=0.5)")
        print("-" * 78)
        print("  (arena colours: matched=green  wrong-class=violet  "
              "false-alarm=red;  leaderboard FA = wrong-class + false-alarm)")
        for pv in s["per_video"]:
            tag = "NORMAL" if pv["is_normal"] else ""
            print(f"  {pv['video_id']}  truth {pv['truth']:2d}  pred {pv['pred']:2d}"
                  f"  matched {pv['found']:2d}  wrong-class {pv['wrongclass']:2d}"
                  f"  false-alarm {pv['fa']:2d}  missed {pv['truth'] - pv['found']:2d}"
                  f"  {tag}")
            if args.timeline and not pv["is_normal"]:
                tr = gt[(gt.video_id == pv["video_id"]) & gt.start_time_sec.notna()]
                dur = (pred.get(pv["video_id"]) or {}).get("duration_sec") \
                    or float(tr.end_time_sec.max())
                tspans = [(x.start_time_sec, x.end_time_sec) for x in tr.itertuples()]
                pspans = [(e["start_time_sec"], e["end_time_sec"])
                          for e in ((pred.get(pv["video_id"]) or {}).get("events") or [])]
                t_row, p_row = bar(tspans, pspans, dur)
                print(f"      truth |{t_row}|  0-{dur:.0f}s")
                print(f"      ours  |{p_row}|")
        if s["matched"]:
            print("  matched:")
            for vid, cls, i in s["matched"]:
                print(f"    {vid}  {cls:34} IoU {i:.3f}")


if __name__ == "__main__":
    main()
