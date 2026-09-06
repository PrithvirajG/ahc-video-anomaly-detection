# =============================================================================
# 3 - Libraries, config, and the ground-truth index
# =============================================================================
# Every knob lives in CFG so nothing below has a magic number. Re-run this cell
# after changing one; nothing downstream caches config.

import json
import re
import time
from dataclasses import dataclass, field

import torch

RUNS = WORK / "runs"
RUNS.mkdir(parents=True, exist_ok=True)

# Every output carries the timestamp of the run that produced it. Without this
# each run overwrites the last, and once a few are downloaded you are left
# guessing which "predictions_raw (7).json" came from which set of changes -
# which matters here, because comparing runs against each other is how nearly
# every finding in this project was made.
RUN_ID = time.strftime("%Y%m%d-%H%M%S")


def run_path(name: str) -> Path:
    """runs/<stem>_<RUN_ID><suffix>, e.g. runs/predictions_raw_20260906-011530.json"""
    p = Path(name)
    return RUNS / f"{p.stem}_{RUN_ID}{p.suffix}"


def free_cuda(*names: str) -> float:
    """Drop the named globals and hand their VRAM back. Returns GB still in use.

    Re-running a model cell in a notebook rebinds the name but does NOT free the
    old weights - the previous module is still referenced until the rebind
    completes, so for a moment two copies of a 9GB model are resident, and on a
    16GB T4 that is an out-of-memory error rather than a slow moment. Deleting
    the name first makes the load work at the cost of a reload; the callers
    below skip even that when the same model is already in memory.
    """
    import gc
    g = globals()
    for n in names:
        if n in g:
            del g[n]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return torch.cuda.memory_allocated() / 1e9
    return 0.0


@dataclass
class Config:
    # a list, not a path: an upload of several zips extracts to several sibling
    # trees on Kaggle, and the pack is their union (see cell 2)
    # In eval mode this holds BOTH packs: the eval tree supplies the videos to
    # answer for, the practice tree supplies the known-normal footage cell 5
    # calibrates on. Indexing them together is why no cell below needs to know
    # which pack a given video came from.
    data_roots: list = field(default_factory=lambda: list(DATA_ROOTS) + list(EVAL_ROOTS))

    # --- sampling -------------------------------------------------------
    sample_fps: float = 2.0        # frames/s pulled off the decoder
    max_side: int = 640            # downscale before anything expensive

    # --- stage 0: motion gate -------------------------------------------
    # 4.2 is the measured median frame-diff over the public test set, so this
    # discards ~50% of frames - the rate Cerberus reports. The first guess was
    # 1.6, which passed 12/12 frames on a normal video: a gate that gates
    # nothing. Re-measure if the encoder or sample_fps changes.
    motion_thresh: float = 4.2     # mean abs frame-diff on a 160x90 gray image
    static_keepalive_sec: float = 4.0   # force a frame through even if nothing moves
    visual_prompt: str = "circle"  # "circle" | "square" | "none"

    # --- stage 1: encoder + rule deviation -------------------------------
    encoder_id: str = "google/siglip2-base-patch16-224"
    topk: int = 5                  # rules summed per frame in health()
    escalate_pct: float = 12.0     # % lowest-health frames sent to stage 2
    health_thresh: float | None = None   # set by calibration in cell 5

    # --- scan floor: guarantee long videos are actually looked at ---------
    # health_thresh is calibrated on 5-30s training clips and then applied to
    # 240-629s test videos from different cameras - an absolute cut taken from
    # one distribution and used on another. Measured result: six of the eight
    # anomalous long videos sent 0-1 windows to the VLM, and those videos carry
    # 75 of the 100 available marks. T027 has four real traffic jams and not one
    # of its 420 surviving frames scored below the threshold, so the VLM never
    # looked at it at all.
    #
    # The floor only ever ADDS windows; escalation is untouched. Turn it off and
    # behaviour is exactly as before.
    scan_floor_enabled: bool = True
    scan_floor_min_video_sec: float = 60.0   # L1 clips are 5-27s, L2/L3 are 240s+,
                                             # so nothing sits near this boundary
    scan_floor_interval_sec: float = 20.0    # median real event is 20s, so a 20s
                                             # stride lands inside any median-or-
                                             # longer event; shorter ones can still
                                             # slip through - tighten this to close
                                             # that gap at proportional GPU cost

    # --- stage 2: small VLM ----------------------------------------------
    # Qwen3-VL-4B, not the 3B this was originally pinned to for the GTX 1650's
    # 4GB VRAM. Irrelevant on a T4 (16GB) - see the long comment in cell 6's
    # load_vlm() for the VRAM math and the organizer-referenced papers that
    # independently validate this exact model for this exact task.
    vlm_id: str = "Qwen/Qwen3-VL-4B-Instruct"
    vlm_frames: int = 4            # frames per escalated window
    vlm_max_new_tokens: int = 160

    # REVERTED to 0 (no widening) after measuring it. The theory was that four
    # consecutive samples at 2fps give the model ~2s, too little to separate a
    # collision from the queue it causes. Widening to 8 frames over 16s produced
    # word-for-word identical descriptions on T025 - "a dense queue of vehicles
    # is stopped or moving very slowly on the left side of the highway", before
    # and after - so the extra span bought nothing.
    #
    # It also cost: a 16s span on T032 pulls more scene into view, the model
    # names whatever is most salient across it, and its verdicts went from
    # loitering 10 / stalled 6 to loitering 6 / stalled 6 / accident 3, flipping
    # a correct video-level class to a wrong one. Class accuracy 0.75 -> 0.50,
    # and +3.1 minutes.
    #
    # Set to 16.0 to re-enable; the code path is kept because the idea is sound
    # for a model that can actually resolve the detail, which is the next knob.
    vlm_span_sec: float = 0.0

    # THE ACTUAL CONSTRAINT, measured after the above failed. The test videos are
    # 1280x720 and max_side downscales them to 640x360, discarding 75% of the
    # pixels. On a drone shot of a highway, two vehicles in contact occupy a few
    # dozen pixels at that scale - the model is not failing to reason about
    # damage, it is being handed frames where the damage is gone.
    #
    # Rather than send every frame at full size, crop to the motion region the
    # gate already located and send THAT at native resolution: the detail lands
    # where the evidence is, and the token cost stays flat. vlm_crop_context is
    # how much of the surroundings to keep - a tight crop of a wreck with no road
    # around it is its own kind of unanswerable.
    vlm_crop_to_motion: bool = True
    vlm_crop_context: float = 3.0   # multiple of the motion box to include
    vlm_crop_min_px: int = 320      # never crop below this, small boxes need room

    # --- temporal aggregation --------------------------------------------
    enter_conf: float = 0.55       # stage-2 confidence to open an event
    exit_conf: float = 0.35        # ...and to close it (hysteresis)
    merge_gap_sec: float = 3.0     # bridge two events of the same class

    # The only declared constant left in the extent path. Cell 7 removed the 15s
    # floor, the 180s cap and both merge-gap constants; what remains is a
    # symmetric quantisation allowance, because frames land on a 1/sample_fps
    # grid and a real boundary can sit one interval outside the window that
    # caught it. Larger helps a straddled boundary and hurts a very short event
    # (IoU falls once the prediction outgrows the truth), so it lives here to be
    # swept offline against stored window_verdicts rather than re-run on a GPU.
    extent_buffer_sec: float = 2.0

    # --- the learned prior (cell 5b) --------------------------------------
    # All three chosen by sweeping the cached training embeddings offline, which
    # costs seconds and no GPU. Held out 559 clips, 25% of the capped set:
    #
    #   C          top-1   top-5   recall@0.90   specificity@0.90
    #   1          0.673   0.969      0.849           0.893
    #   10         0.717   0.977      0.903           0.933
    #   100        0.739   0.983      0.930           0.947   <- chosen
    #   300        0.741   0.981      0.938           0.933
    #
    # Less regularisation is better here, which is what 768 well-conditioned
    # frozen features and 1,676 training rows should predict.
    probe_C: float = 100.0

    # Escalate a window when the probe puts P(anomalous) at or above this.
    # 0.90 buys recall 0.930 at specificity 0.947 - it replaces a health
    # threshold measured to be INVERTED on three of four test videos.
    probe_escalate_p: float = 0.90

    # ...and the bar for the probe to CONTRADICT stage 2. Started at 0.95, the
    # point where held-out training clips gave zero false positives. Real
    # footage then said that was needlessly strict:
    #
    #   video   highest probe score OUTSIDE any real event
    #   T030    0.582   <- a wholly normal video, the thing we must not flag
    #   T026    0.670
    #   T025    0.858
    #
    # T030 is the one that matters and it tops out at 0.582, so 0.90 keeps a
    # 0.32 margin on the only video where a false alarm is possible. Dropping
    # from 0.95 to 0.90 fires on two more of T025's six real accidents, whose
    # in-event scores run 0.748 to 0.977.
    #
    # The windows this fires on OUTSIDE an event are almost all T031 and T032,
    # where the probe believes the whole video is anomalous. Those are extent
    # errors inside videos we already flag, not false alarms on normal ones -
    # a different and much cheaper kind of wrong.
    probe_override_p: float = 0.90
    probe_override_enabled: bool = True


CFG = Config()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_NAME = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"
CAP = torch.cuda.get_device_capability(0) if DEVICE == "cuda" else (0, 0)
# T4 (sm_75, real tensor cores) wants fp16. A GTX 1650 is TU117 - capability 7.5
# but with the tensor cores fused off, so fp16 has no fast path and measures 3x
# SLOWER than fp32 (11 vs 33 fps for SigLIP2-base). Decide from the device.
USE_FP16 = DEVICE == "cuda" and not any(x in GPU_NAME for x in ("1650", "1660"))
DTYPE = torch.float16 if USE_FP16 else torch.float32

print(f"device : {DEVICE}  {GPU_NAME}  sm_{CAP[0]}{CAP[1]}")
print(f"dtype  : {DTYPE}")
print(f"torch  : {torch.__version__}")

# --- index -------------------------------------------------------------------
# video_id in the CSVs is the filename stem, so one map over every root gives the
# whole pack and no other cell has to know the layout. Unioning is what makes a
# split upload (8 sibling trees) behave identically to a single tree.
#
# DO NOT switch this to videos.csv's `filename` column. It holds a path relative
# to the CSV ("videos/T001.mp4"), and on a split upload every CSV sits in the
# first archive while the videos are spread across all eight - so 2,771 of 3,207
# rows (86%) point at files that are not there. Measured, not hypothetical. A
# stem is location-independent; a relative path is not. ground_truth.csv has no
# path column at all, which is the join we actually rely on.
VIDEO_PATHS = {}
for _root in CFG.data_roots:
    for _p in _root.rglob("*.mp4"):
        VIDEO_PATHS.setdefault(_p.stem, _p)


def load_ground_truth(split: str) -> pd.DataFrame:
    """Concatenate every ground_truth.csv under train/ or test/, across all roots.

    train/ has one per class folder; test/ has a single one. Both share the
    schema, so one loader serves either, and `path` is added for convenience.
    Rows are de-duplicated because a split upload can surface the same CSV twice.
    """
    frames = []
    for root in CFG.data_roots:
        base = root / split
        if not base.exists():
            continue
        for csv in sorted(base.rglob("ground_truth.csv")):
            df = pd.read_csv(csv)
            df["source_csv"] = str(csv)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    gt = pd.concat(frames, ignore_index=True)
    key = [c for c in ("video_id", "level", "class_name", "start_time_sec",
                       "end_time_sec") if c in gt]
    gt = gt.drop_duplicates(subset=key).reset_index(drop=True)
    for col in ("start_time_sec", "end_time_sec"):
        if col in gt:
            gt[col] = pd.to_numeric(gt[col], errors="coerce")
    gt["path"] = gt["video_id"].astype(str).map(VIDEO_PATHS)
    return gt


EVAL_MANIFEST_PATH = None
for _r in EVAL_ROOTS:
    _m = _r / "manifest.json"
    if _m.exists():
        EVAL_MANIFEST_PATH = _m
        break


def load_eval_index() -> pd.DataFrame:
    """Build a GT_TEST-shaped frame for the evaluation pack.

    This is the hinge the whole flag turns on. It returns the SAME columns
    load_ground_truth returns - video_id, level, path, and the truth columns
    class_name / start_time_sec / end_time_sec / is_anomaly - except the truth
    columns are empty, because the pack ships none.

    Keeping the shape identical is deliberate. Cells 8, 10 and 11 all consume
    GT_TEST, and every one of them keeps working with no branch of its own; the
    alternative was a MODE check in five places, each of which could drift. The
    truth columns are present-but-NaN rather than absent so that cell 9 can ask
    "is there truth here?" and get a clean answer instead of a KeyError.

    Level comes from the manifest, not the folder name, because the manifest is
    what the arena scores against - and duration_sec comes from it too, which is
    more authoritative than our own decode.
    """
    rows = []
    if EVAL_MANIFEST_PATH is not None:
        man = json.loads(EVAL_MANIFEST_PATH.read_text())
        for v in man.get("videos", man if isinstance(man, list) else []):
            rows.append({"video_id": v["video_id"],
                         "level": int(v.get("level", v.get("difficulty", 1))),
                         "duration_sec": float(v.get("duration_sec", 0)) or None})
    else:
        # No manifest: fall back to the tree itself. Levels come from the folder
        # name here, which is the best available and matches how it was shipped.
        print("  ! no manifest.json in the eval pack - falling back to the "
              "L1/L2/L3 folder names for levels")
        for r in EVAL_ROOTS:
            for lv in sorted(EVAL_LEVEL_DIRS):
                for mp4 in sorted((r / lv).rglob("*.mp4")) if (r / lv).is_dir() else []:
                    rows.append({"video_id": mp4.stem, "level": int(lv[1]),
                                 "duration_sec": None})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates("video_id").reset_index(drop=True)
    for col in ("class_name", "start_time_sec", "end_time_sec", "is_anomaly"):
        df[col] = pd.NA
    df["path"] = df["video_id"].map(VIDEO_PATHS)
    return df


GT_TRAIN = load_ground_truth("train")
GT_TEST = load_eval_index() if MODE == "eval" else load_ground_truth("test")

# Truth present or not is asked once, here, and answered everywhere else by
# reading this - cells 9 and 11 must not re-derive it from MODE, or the two
# could disagree.
HAS_TRUTH = (not GT_TEST.empty) and GT_TEST["class_name"].notna().any()

print(f"\nindexed {len(VIDEO_PATHS)} videos across {len(CFG.data_roots)} root(s)")
for name, gt in (("train", GT_TRAIN), ("test", GT_TEST)):
    if gt.empty:
        print(f"  {name}: no ground_truth.csv found")
        continue
    missing = int(gt["path"].isna().sum())
    print(f"  {name}: {len(gt)} rows, {gt['video_id'].nunique()} videos"
          f"{f', {missing} rows with no matching mp4' if missing else ''}")
    unknown = set(gt.get("class_name", pd.Series(dtype=str)).dropna()) - set(CLASSES)
    if unknown:
        print(f"  ! labels outside the twelve: {sorted(unknown)}")

if MODE == "eval":
    print()
    print(f"EVAL SET: {len(GT_TEST)} videos to answer for"
          + (f", manifest {EVAL_MANIFEST_PATH.name}" if EVAL_MANIFEST_PATH else ""))
    if not GT_TEST.empty:
        for _lv, _n in sorted(GT_TEST["level"].value_counts().items()):
            _d = GT_TEST[GT_TEST["level"] == _lv]["duration_sec"]
            print(f"  L{_lv}: {_n:2d} videos, {_d.sum() / 60:5.1f} min")
        _miss = GT_TEST[GT_TEST["path"].isna()]["video_id"].tolist()
        if _miss:
            print(f"  ! {len(_miss)} manifest video(s) with no mp4 on disk: "
                  f"{_miss[:8]}{' ...' if len(_miss) > 8 else ''}")
    print(f"  ground truth: {'present' if HAS_TRUTH else 'ABSENT (cell 9 will skip)'}")
