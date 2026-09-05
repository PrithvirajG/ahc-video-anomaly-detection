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


@dataclass
class Config:
    # a list, not a path: an upload of several zips extracts to several sibling
    # trees on Kaggle, and the pack is their union (see cell 2)
    data_roots: list = field(default_factory=lambda: list(DATA_ROOTS))

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

    # --- temporal aggregation --------------------------------------------
    enter_conf: float = 0.55       # stage-2 confidence to open an event
    exit_conf: float = 0.35        # ...and to close it (hysteresis)
    merge_gap_sec: float = 3.0     # bridge two events of the same class


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


GT_TRAIN = load_ground_truth("train")
GT_TEST = load_ground_truth("test")

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
