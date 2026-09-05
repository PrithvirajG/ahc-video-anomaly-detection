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

MANIFEST_PATH = WORK / "manifest.json"     # drop the arena's file here once fetched


def load_manifest() -> dict[str, int]:
    """{video_id: level} for every video the arena wants an answer for.

    Falls back to the local public test set's own ground truth so this cell is
    fully testable before the real manifest exists. Swap in the real file and
    everything below is unchanged.
    """
    if MANIFEST_PATH.exists():
        m = json.loads(MANIFEST_PATH.read_text())
        videos = m.get("videos", m if isinstance(m, list) else [])
        # the general PDF calls this field "level"; the live benchmark page's
        # own prose calls it "difficulty" - accept either rather than guess
        return {v["video_id"]: int(v.get("level", v.get("difficulty")))
                for v in videos}
    print(f"no manifest at {MANIFEST_PATH} - using the local test set's ground "
          "truth as a stand-in so this cell is testable right now")
    return (GT_TEST.drop_duplicates("video_id")
            .set_index("video_id")["level"].astype(int).to_dict())


def load_manifest_durations() -> dict[str, float]:
    """video_id -> duration_sec, straight from the manifest when we have one -
    more authoritative than our own decoded duration for the 'end_time_sec must
    stay inside the duration' check. Falls back to PRED's measured duration."""
    if MANIFEST_PATH.exists():
        m = json.loads(MANIFEST_PATH.read_text())
        videos = m.get("videos", m if isinstance(m, list) else [])
        return {v["video_id"]: float(v["duration_sec"])
                for v in videos if "duration_sec" in v}
    return {}


def events_for_submission(pred_row: dict, level: int) -> list[dict]:
    """Our internal event dict -> the arena's exact per-event schema."""
    out = []
    for e in (pred_row.get("events") or []):
        out.append({
            "class_name": e["class_name"],          # never "normal" - empty list instead
            "start_time_sec": None if level == 1 else float(e["start_time_sec"]),
            "end_time_sec": None if level == 1 else float(e["end_time_sec"]),
            "explanation": (e.get("description_summary") or None),
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

OUT_PATH = RUNS / "arena_submission.json"
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
