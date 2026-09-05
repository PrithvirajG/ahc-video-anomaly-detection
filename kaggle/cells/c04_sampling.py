# =============================================================================
# 3 - Frame sampling, motion gate, visual prompting
# =============================================================================
# Stage 0 of the cascade. One frame-differencing computation does two jobs, as
# in Cerberus: it decides whether a frame is worth encoding at all, and it
# locates the moving region so we can draw a visual prompt on it.

import cv2
from PIL import Image


def iter_sampled_frames(path, target_fps: float, max_side: int = 640):
    """Yield (t_seconds, bgr_frame) at roughly target_fps.

    grab() advances the decoder without colour-converting or copying; retrieve()
    is only paid on frames we keep. At 2 fps off 25 fps source that is ~12x less
    work than read()-ing everything. Seeking per-sample with CAP_PROP_POS_FRAMES
    would be worse still - every seek forces a keyframe jump and re-decode.
    """
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps != src_fps or src_fps <= 0:   # 0, or NaN
        src_fps = 25.0
    stride = max(1, int(round(src_fps / target_fps)))
    i = 0
    try:
        while True:
            if not cap.grab():
                break
            if i % stride == 0:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    h, w = frame.shape[:2]
                    if max(h, w) > max_side:
                        s = max_side / max(h, w)
                        frame = cv2.resize(frame, (int(w * s), int(h * s)))
                    yield i / src_fps, frame
            i += 1
    finally:
        cap.release()


def video_duration(path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return float(n / fps) if fps > 0 else 0.0


class MotionGate:
    """Frame differencing on a 160x90 grayscale thumbnail.

    The keepalive is not in the paper and matters a lot here. Cerberus gates on
    motion because its anomalies are motion events; three of our twelve labels
    are not. waterlogging_or_flood and road_spill_or_debris are static
    conditions, and stalled_or_broken_down_vehicle is *defined* by the absence of
    motion once a vehicle has been still long enough. A pure motion gate would
    discard precisely the frames that carry their evidence, so we force one
    frame through every static_keepalive_sec regardless of score.
    """

    def __init__(self, cfg):
        self.thresh = cfg.motion_thresh
        self.keepalive = cfg.static_keepalive_sec
        self.prev = None
        self.last_pass = -1e9

    def __call__(self, t: float, frame):
        small = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
        if self.prev is None:
            self.prev, self.last_pass = small, t
            return True, 0.0, None, "first"

        diff = cv2.absdiff(small, self.prev)
        self.prev = small
        score = float(diff.mean())

        moving = score >= self.thresh
        stale = (t - self.last_pass) >= self.keepalive
        if not (moving or stale):
            return False, score, None, "skipped"

        self.last_pass = t
        box = self._largest_region(diff, frame.shape) if moving else None
        return True, score, box, ("motion" if moving else "keepalive")

    @staticmethod
    def _largest_region(diff, shape):
        """Bounding box of the biggest moving blob, in full-frame coordinates."""
        _, mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        if w * h < 12:                      # noise, not a subject
            return None
        H, W = shape[:2]
        sx, sy = W / 160.0, H / 90.0
        return int(x * sx), int(y * sy), int(w * sx), int(h * sy)


def draw_visual_prompt(frame, box, style: str = "circle"):
    """Overlay a red marker on the moving region.

    Cerberus finds circles pull VLM attention harder (better recall) while
    squares admit less background (better precision), and picks between them by
    motion scale. We expose the choice and default to the recall-favouring one,
    because stage 2 is the thing that can say no - a miss here is unrecoverable.
    """
    if box is None or style == "none":
        return frame
    out = frame.copy()
    x, y, w, h = box
    if style == "square":
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 3)
    else:
        cx, cy = x + w // 2, y + h // 2
        cv2.circle(out, (cx, cy), max(18, int(0.6 * max(w, h))), (0, 0, 255), 3)
    return out


def to_pil(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def sample_video(path, cfg=None):
    """Full stage-0 pass. Returns the frames that survive the gate."""
    cfg = cfg or CFG
    gate = MotionGate(cfg)
    kept, n_seen = [], 0
    for t, frame in iter_sampled_frames(path, cfg.sample_fps, cfg.max_side):
        n_seen += 1
        passed, score, box, why = gate(t, frame)
        if passed:
            kept.append({"t": t, "frame": frame, "motion": score,
                         "box": box, "why": why})
    return kept, n_seen


# --- smoke test on one video --------------------------------------------------
_probe = next(iter(VIDEO_PATHS.values()), None)
if _probe is not None:
    _t0 = time.time()
    _kept, _seen = sample_video(_probe)
    _dt = time.time() - _t0
    _why = pd.Series([k["why"] for k in _kept]).value_counts().to_dict()
    print(f"{_probe.name}: {_seen} sampled -> {len(_kept)} kept "
          f"({100 * len(_kept) / max(_seen, 1):.0f}%) in {_dt:.1f}s   {_why}")
    print(f"stage-0 throughput: {_seen / max(_dt, 1e-6):.0f} sampled-frames/s")
else:
    print("no videos indexed yet - run cells 2 and 3 first")
