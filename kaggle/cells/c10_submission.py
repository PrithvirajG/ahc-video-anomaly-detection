# =============================================================================
# 10 - Arena submission file
# =============================================================================
# The real evaluation is NOT the CSV cell 9 used to look at itself - it's a JSON
# file matching the arena's exact schema. The practice pack (checked against the
# live Benchmark page) turns out to BE our local T00x test set - 34 videos,
# L1 24 / L2 6 / L3 4 - so this is genuinely submittable today, not blocked on
# an unseen video set. A later "final" round may swap in unseen videos (the
# general PDF's own example uses E001, E002, ...), so nothing here hard-codes
# the T0xx naming. Drop a fetched manifest.json into WORK (see MANIFEST_PATH)
# to use the arena's exact video/level list instead of the local stand-in.
#
# Scoring weight, from the live page: D1=25, D2=35, D3=40 of 100. 75 of 100
# points sit on the 10 timed videos, not the 24 untimed ones - see
# docs/SUBMISSION_ARENA.md for the full rules this cell builds around.
#
# Two rules that get a file silently rejected, not just scored low:
#   - a normal video is "events": [] - NEVER {"class_name": "normal"}
#   - Level-1 events carry start_time_sec/end_time_sec = null, not omitted and
#     not 0 - Level 2/3 require real numbers, end strictly greater than start
#
# And two scoring rules worth designing the AGGREGATION around, not just
# complying with here:
#   - predicting ANYTHING on a truly-normal Level-2/3 video scores that video
#     ZERO - there is no partial credit for a false alarm there
#   - several overlapping fragments for one real event only let the BEST one
#     match; the rest count AGAINST you - our per-class merge_gap in cell 7 is
#     exactly what keeps this from happening, so don't loosen it casually

# The arena's practice manifest, embedded: {video_id: (level, duration_sec)}.
#
# It is here rather than read from a file because the file lives in the repo and
# the notebook runs on Kaggle, where nothing mounts it - so practice mode was
# silently falling through to the ground-truth CSV, which carries levels but NOT
# durations. That left the end_time_sec bounds check using our own decoded
# duration, which is short by up to 2.7s (T033 626.1 against the arena's 628.8),
# and short durations trim real overlap off exactly the long D3 events that pay
# five marks each.
#
# 34 entries is small enough to embed and large enough that a divergence would
# matter, so _check_embedded_manifest() below cross-checks it against GT_TEST at
# import time rather than trusting it to stay correct.
EMBEDDED_MANIFEST = {
    "T001": (1, 5.7), "T002": (1, 5.8), "T003": (1, 14.0),
    "T004": (1, 14.0), "T005": (1, 5.7), "T006": (1, 17.0),
    "T007": (1, 22.7), "T008": (1, 5.7), "T009": (1, 5.8),
    "T010": (1, 13.2), "T011": (1, 11.2), "T012": (1, 5.8),
    "T013": (1, 5.8), "T014": (1, 5.8), "T015": (1, 5.8),
    "T016": (1, 5.7), "T017": (1, 5.7), "T018": (1, 11.1),
    "T019": (1, 13.3), "T020": (1, 26.1), "T021": (1, 20.3),
    "T022": (1, 16.0), "T023": (1, 20.0), "T024": (1, 16.0),
    "T025": (2, 240.0), "T026": (2, 240.0), "T027": (2, 240.0),
    "T028": (2, 240.0), "T029": (2, 240.0), "T030": (2, 240.0),
    "T031": (3, 360.0), "T032": (3, 307.7), "T033": (3, 628.8),
    "T034": (3, 376.5),
}

# In eval mode the manifest ships inside the dataset itself, so there is nothing
# to fetch by hand and no chance of scoring against a stale copy. A file dropped
# into WORK still wins, which is the escape hatch if the arena reissues one.
_WORK_MANIFEST = WORK / "manifest.json"
MANIFEST_PATH = (_WORK_MANIFEST if _WORK_MANIFEST.exists()
                 else (EVAL_MANIFEST_PATH if MODE == "eval" and EVAL_MANIFEST_PATH
                       else _WORK_MANIFEST))


def _manifest_videos() -> tuple[list[dict], str]:
    """[{video_id, level, duration_sec}, ...] plus where it came from.

    One resolver for both loaders below, so the level map and the duration map
    can never disagree about which source they read.
    """
    if MANIFEST_PATH.exists():
        m = json.loads(MANIFEST_PATH.read_text())
        videos = m.get("videos", m if isinstance(m, list) else [])
        # the general PDF calls this field "level"; the live benchmark page's
        # own prose calls it "difficulty" - accept either rather than guess
        return ([{"video_id": v["video_id"],
                  "level": int(v.get("level", v.get("difficulty"))),
                  "duration_sec": (float(v["duration_sec"])
                                   if "duration_sec" in v else None)}
                 for v in videos], str(MANIFEST_PATH))
    if MODE != "eval" and EMBEDDED_MANIFEST:
        return ([{"video_id": k, "level": lv, "duration_sec": d}
                 for k, (lv, d) in sorted(EMBEDDED_MANIFEST.items())],
                "embedded practice manifest")
    return [], "none"


_MANIFEST_VIDEOS, MANIFEST_SOURCE = _manifest_videos()
print(f"manifest: {MANIFEST_SOURCE}  ({len(_MANIFEST_VIDEOS)} videos)")


def _check_embedded_manifest() -> None:
    """Cross-check the embedded copy against the mounted ground truth.

    An embedded constant is a copy, and copies drift. This is cheap and turns a
    silent wrong-level submission into a printed warning.
    """
    if MANIFEST_SOURCE != "embedded practice manifest" or GT_TEST.empty:
        return
    gt_lv = (GT_TEST.drop_duplicates("video_id")
             .set_index("video_id")["level"].astype(int).to_dict())
    emb = {v["video_id"]: v["level"] for v in _MANIFEST_VIDEOS}
    if set(emb) != set(gt_lv):
        print(f"  ! embedded manifest / ground truth disagree on which videos "
              f"exist: only-embedded={sorted(set(emb) - set(gt_lv))}, "
              f"only-truth={sorted(set(gt_lv) - set(emb))}")
    bad = {k: (emb[k], gt_lv[k]) for k in set(emb) & set(gt_lv) if emb[k] != gt_lv[k]}
    if bad:
        print(f"  ! embedded manifest / ground truth disagree on levels: {bad}")
    if not bad and set(emb) == set(gt_lv):
        print(f"  embedded manifest agrees with ground truth on all "
              f"{len(emb)} videos (ids and levels)")


_check_embedded_manifest()


def load_manifest() -> dict[str, int]:
    """{video_id: level} for every video the arena wants an answer for."""
    if _MANIFEST_VIDEOS:
        return {v["video_id"]: v["level"] for v in _MANIFEST_VIDEOS}
    print("no manifest available - using the local test set's ground truth as a "
          "stand-in so this cell is testable right now")
    return (GT_TEST.drop_duplicates("video_id")
            .set_index("video_id")["level"].astype(int).to_dict())


def load_manifest_durations() -> dict[str, float]:
    """video_id -> duration_sec, straight from the manifest when we have one -
    more authoritative than our own decoded duration for the 'end_time_sec must
    stay inside the duration' check. Falls back to PRED's measured duration."""
    return {v["video_id"]: v["duration_sec"] for v in _MANIFEST_VIDEOS
            if v.get("duration_sec") is not None}


def events_for_submission(pred_row: dict, level: int) -> list[dict]:
    """Our internal event dict -> the arena's exact per-event schema.

    Level 1 gets ONE event, never more - "One label for the whole clip" is the
    spec, not a suggestion. A second guess on a single-label task has zero
    possible upside (Level 1 scoring gives no credit for extra classes) and
    real downside (the arena counts each non-matching predicted event as its
    own false alarm, on top of the miss it doesn't fix) - measured directly: a
    tied-confidence fire+smoke double-guess on one real practice-pack video
    was two of the six false alarms in our first submission, when emitting
    only the correct one of the two would have cost nothing. Collapsing to the
    single highest-confidence event can only reduce that count, never raise it.
    """
    events = pred_row.get("events") or []
    if level == 1 and len(events) > 1:
        events = [max(events, key=lambda e: e["peak_confidence"])]
    out = []
    for e in events:
        # sub_tags stay OUT of the arena event object - the schema wants one
        # class per event - but they are real observations, so they go into the
        # explanation, which is a scored bonus field that never costs anything.
        expl = e.get("description_summary") or ""
        subs = [s["class_name"] for s in (e.get("sub_tags") or [])]
        if subs:
            also = ", ".join(s.replace("_", " ") for s in subs)
            expl = (expl + f" Also observed at this incident: {also}.").strip()
        expl = expl[:500] if len(expl) >= 20 else (expl or None)

        out.append({
            "class_name": e["class_name"],          # never "normal" - empty list instead
            "start_time_sec": None if level == 1 else float(e["start_time_sec"]),
            "end_time_sec": None if level == 1 else float(e["end_time_sec"]),
            "explanation": expl,
        })
    return out


def build_submission(pred: pd.DataFrame, manifest: dict[str, int],
                     submission_id: str, model_name: str = "ahc-cascade-v1") -> dict:
    by_id = {r["video_id"]: r for r in pred.to_dict("records")}
    predictions, total_wall_ms, max_parallel = [], 0.0, 1

    for vid, level in manifest.items():
        row = by_id.get(vid)
        if row is None:
            print(f"  ! {vid}: not in PRED - omitted. Per the rules an omitted "
                  "video KEEPS its previous answer (or scores normal if you have "
                  "never answered it) - it is not cleared.")
            continue
        events = events_for_submission(row, level)
        rt = row.get("runtime_metadata") or {
            "frames_processed": row.get("frames_sampled", 0),
            "chunks_processed": 1,
            "end_to_end_internal_time_ms": round(row.get("sec_total", 0) * 1000, 1),
            "model_runtimes": [],
        }
        total_wall_ms += rt["end_to_end_internal_time_ms"]
        predictions.append({"video_id": vid, "events": events, "runtime_metadata": rt})

    return {
        "schema_version": "1.0",
        "submission_id": submission_id,
        "model_name": model_name,
        "run_metadata": {"total_wall_time_ms": round(total_wall_ms, 1),
                         "hardware": GPU_NAME, "max_parallel_videos": max_parallel},
        "predictions": predictions,
    }


def validate_submission(sub: dict, manifest: dict[str, int],
                        durations: dict[str, float] | None = None) -> list[str]:
    """Every rule from the PDF's 'Things that catch people out', checked before
    upload. A rejected file doesn't burn a run, but there's no reason to find
    that out on the arena instead of here.

    `durations` (video_id -> seconds), when given, also checks the live
    benchmark page's rule that end_time_sec must "stay inside the duration" -
    a check the general PDF never mentions, so it's easy to miss.
    """
    durations = durations or {}
    problems, seen = [], set()
    for p in sub["predictions"]:
        vid = p["video_id"]
        if vid in seen:
            problems.append(f"{vid}: video_id appears more than once")
        seen.add(vid)
        if vid not in manifest:
            problems.append(f"{vid}: not in manifest")
            continue
        level = manifest[vid]
        if "runtime_metadata" not in p:
            problems.append(f"{vid}: missing runtime_metadata (required on "
                            "every video; also where the latency bonus comes from)")
        dur = durations.get(vid)
        for e in p["events"]:
            if e["class_name"] not in ANOMALY_CLASSES:
                problems.append(f"{vid}: class_name {e['class_name']!r} invalid - "
                                "must be one of the 11 event classes, never 'normal'")
            if level == 1:
                if e["start_time_sec"] is not None or e["end_time_sec"] is not None:
                    problems.append(f"{vid}: Level 1 events must have null timestamps")
            else:
                if e["start_time_sec"] is None or e["end_time_sec"] is None:
                    problems.append(f"{vid}: Level {level} requires real timestamps")
                elif e["end_time_sec"] <= e["start_time_sec"]:
                    problems.append(f"{vid}: end_time_sec must be greater than start_time_sec")
                elif dur is not None and e["end_time_sec"] > dur + 0.5:
                    problems.append(f"{vid}: end_time_sec {e['end_time_sec']} exceeds "
                                    f"the video's duration ({dur}s)")
    missing = set(manifest) - seen
    if missing:
        problems.append(f"{len(missing)} manifest video(s) never answered "
                        f"(scored as normal by default): {sorted(missing)[:10]}"
                        f"{' ...' if len(missing) > 10 else ''}")
    return problems


MANIFEST = load_manifest()
SUBMISSION = build_submission(PRED, MANIFEST, submission_id="ahc-run-01")
VIDEO_DURATIONS = (load_manifest_durations()
                  or dict(zip(PRED["video_id"], PRED.get("duration_sec", []))))
PROBLEMS = validate_submission(SUBMISSION, MANIFEST, VIDEO_DURATIONS)

OUT_PATH = run_path("arena_submission.json")
OUT_PATH.write_text(json.dumps(SUBMISSION, indent=1))
size_kb = OUT_PATH.stat().st_size / 1024
print(f"\nwrote {OUT_PATH}  ({size_kb:.1f} KB of the 5 MB cap, "
      f"{len(SUBMISSION['predictions'])} videos)")

if PROBLEMS:
    print(f"\n{len(PROBLEMS)} problem(s) - fix before uploading:")
    for p in PROBLEMS[:30]:
        print(f"  ! {p}")
else:
    print("no problems found by local validation - still spot-check a few "
          "entries by eye before uploading, this checks format, not judgment")
