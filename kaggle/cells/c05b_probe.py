# =============================================================================
# 5b - A learned prior, because the written one is inverted
# =============================================================================
# Cell 5 scores a frame by comparing it to 30 hand-written normal rules and 63
# perturbed actions. Measured on the five-video run, comparing health INSIDE a
# real ground-truth event against health OUTSIDE it in the same video:
#
#     T026  inside +0.134  outside +0.608   separation +0.474   works
#     T031  inside -0.085  outside -0.092   separation -0.007   flat
#     T032  inside +0.531  outside +0.415   separation -0.116   INVERTED
#     T025  inside +0.150  outside -0.097   separation -0.247   INVERTED
#
# On three of four videos the anomalous frames are as healthy as or HEALTHIER
# than the rest of the video - they sit at the 65th percentile of their own
# video's health. A congested road looks like a road; a loiterer looks like a
# person. Only T026's road spill, a visible appearance change, separates.
#
# That one signal drives escalation, clustering and extent measurement, so it is
# the foundation under most of the pipeline, and it is upside down.
#
# We also have 3,173 labelled training clips that we have so far used to compute
# exactly one number (health_thresh). This cell spends them properly: embed each
# clip once, then fit a plain multinomial logistic regression over the frozen
# SigLIP2 embeddings. The encoder stays frozen and zero-shot - no backprop, no
# fine-tuning, no GPU for the fit itself. What changes is that 93 English
# sentences are replaced by coefficients learned from labelled examples.
#
# THIS CELL ONLY BUILDS AND MEASURES THE PROBE. Nothing downstream consumes it
# yet, deliberately: how it should be used depends on how good it turns out to
# be, and wiring it in before measuring it would make that unanswerable.

import numpy as np

PROBE_FRAMES = 8            # frames per training example
PROBE_SPAN_SEC = 16.0       # ...spanning this much video
PROBE_MAX_PER_CLASS = 300   # cap: normal has 973 clips and fire has 77
PROBE_CACHE_NAME = f"train_emb_{PROBE_FRAMES}x{int(PROBE_SPAN_SEC)}s.npz"

# 8 frames over 16s, not 4 over 2s, and the reason is the whole point of the
# cell. Our inference window is ~2s wide while the median real event is 20s, so
# a probe trained on wide clips and applied to 2s windows would rebuild the
# train/test mismatch it exists to remove. Both sides use this shape.


def probe_clip_frames(path, t0=None, t1=None, n=PROBE_FRAMES,
                      span=PROBE_SPAN_SEC):
    """n frames spanning `span` seconds, centred on the labelled event.

    Anomaly clips carry start/end times, so we centre on the part that is
    actually anomalous instead of averaging it away with surrounding normal
    footage. Normal clips have no timings and use the middle of the clip.
    Clips shorter than `span` just use everything they have.
    """
    dur = video_duration(path) or 0.0
    if dur <= 0:
        return []
    if t0 is not None and t1 is not None and np.isfinite(t0) and np.isfinite(t1):
        centre = (float(t0) + float(t1)) / 2
    else:
        centre = dur / 2
    half = min(span, dur) / 2
    lo = max(0.0, min(centre - half, dur - min(span, dur)))
    hi = min(dur, lo + min(span, dur))
    want = np.linspace(lo, max(lo, hi - 1e-3), n)

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out = []
    for t in want:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(t * fps)))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        if max(h, w) > CFG.max_side:
            s = CFG.max_side / max(h, w)
            frame = cv2.resize(frame, (int(w * s), int(h * s)))
        out.append(to_pil(frame))
    cap.release()
    return out


def probe_training_rows() -> list[dict]:
    """One row per training clip: path, class, and where the anomaly is.

    Capped per class. The cap is not only about time - normal has 973 clips
    against fire's 77, and an unbalanced fit would learn to say "normal".
    """
    if GT_TRAIN.empty:
        return []
    rows = []
    for cls, grp in GT_TRAIN.groupby("class_name"):
        grp = grp.dropna(subset=["path"])
        if len(grp) > PROBE_MAX_PER_CLASS:
            grp = grp.sample(PROBE_MAX_PER_CLASS, random_state=0)
        for _, r in grp.iterrows():
            rows.append({"video_id": r["video_id"], "path": r["path"],
                         "class_name": cls,
                         "t0": r.get("start_time_sec"),
                         "t1": r.get("end_time_sec")})
    return rows


def find_probe_cache():
    """WORK first, then anywhere under /kaggle/input.

    /kaggle/working does not survive a fresh session, so the cache gets
    re-attached as a Kaggle Dataset or Model and found here instead of costing
    another 40-minute embed. rglob covers both - a Dataset mounts at
    /kaggle/input/<slug>/ and a Model at
    /kaggle/input/models/<owner>/<slug>/other/default/1/ - so it does not matter
    which one it was uploaded as.

    The exact name is tried first. Failing that, any train_emb_*.npz is accepted
    with a warning, because Kaggle sometimes renames on upload and a silent
    40-minute re-embed is a worse outcome than a loud approximate match. Shape
    and classes are validated by the caller either way.
    """
    local = WORK / PROBE_CACHE_NAME
    if local.exists():
        return local
    base = Path("/kaggle/input")
    if not base.exists():
        return None
    for p in base.rglob(PROBE_CACHE_NAME):
        return p
    for p in sorted(base.rglob("train_emb_*.npz")):
        print(f"  ! no {PROBE_CACHE_NAME} attached; using {p.name} instead.")
        print(f"    Check it was built at {PROBE_FRAMES} frames over "
              f"{PROBE_SPAN_SEC:.0f}s - a cache from different settings will "
              f"fit and predict happily while meaning something else.")
        return p
    return None


def build_probe_embeddings(force: bool = False):
    """Embed every (capped) training clip once and cache the result.

    The expensive step is turning pixels into vectors; the step we actually want
    to iterate on is fitting a classifier over them. Separating the two turns
    one experiment per twenty minutes into twenty experiments per minute.
    """
    cached = None if force else find_probe_cache()
    if cached is not None:
        z = np.load(cached, allow_pickle=True)
        X, y = z["X"], z["y"]
        print(f"probe cache: {cached}")
        print(f"  {len(y)} clips, {X.shape[1]}-d embeddings, "
              f"{len(set(y.tolist()))} classes")
        # Validate rather than trust. A cache is a file someone attached, and
        # the failure modes are all silent: a truncated upload, a cache built at
        # different settings, or labels outside the twelve would each fit and
        # predict happily while meaning something else.
        ok = True
        if X.shape[0] != len(y):
            print(f"  ! X has {X.shape[0]} rows against {len(y)} labels"); ok = False
        if not np.isfinite(X).all():
            print("  ! cache contains non-finite values"); ok = False
        unknown = set(map(str, y.tolist())) - set(CLASSES)
        if unknown:
            print(f"  ! labels outside the twelve: {sorted(unknown)}"); ok = False
        if len(y) < 200:
            print(f"  ! only {len(y)} clips - too few to fit a 12-class probe"); ok = False
        if not ok:
            print("  ignoring this cache and rebuilding")
        else:
            return X, y, list(z["ids"])

    rows = probe_training_rows()
    if not rows:
        print("no training ground truth - probe unavailable")
        return None, None, None
    print(f"embedding {len(rows)} training clips at {PROBE_FRAMES} frames over "
          f"{PROBE_SPAN_SEC:.0f}s (one-off, cached to {WORK / PROBE_CACHE_NAME})")

    X, y, ids = [], [], []
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        if i % 200 == 0 or i == len(rows):
            el = time.time() - t0
            print(f"  {i}/{len(rows)}  {el/60:.1f} min elapsed, "
                  f"~{el/i*(len(rows)-i)/60:.1f} min left")
        try:
            frames = probe_clip_frames(r["path"], r["t0"], r["t1"])
        except Exception as e:
            print(f"  ! {r['video_id']}: {str(e).splitlines()[0][:90]}")
            continue
        if not frames:
            continue
        # mean-pool the clip's frames into one vector: the probe's unit is a
        # clip, which is what makes persistence classes representable at all
        emb = embed_images(frames).mean(0)
        X.append(emb.detach().float().cpu().numpy())
        y.append(r["class_name"])
        ids.append(r["video_id"])

    X = np.stack(X) if X else np.zeros((0, 768), dtype=np.float32)
    y = np.array(y)
    np.savez_compressed(WORK / PROBE_CACHE_NAME, X=X, y=y, ids=np.array(ids))
    print(f"  wrote {WORK / PROBE_CACHE_NAME}  "
          f"({X.nbytes / 1e6:.1f} MB, {time.time() - t0:.0f}s)")
    return X, y, ids


PROBE_X, PROBE_Y, PROBE_IDS = build_probe_embeddings()


def fit_probe(X, y, C: float | None = None):
    """Multinomial logistic regression over frozen embeddings.

    Balanced class weights because the cap does not fully level things (fire has
    77 clips against normal's 300), and an unbalanced fit on this data learns
    the majority answer - which is exactly the failure we are trying to remove.

    Reported on a held-out split, not on the training data, because a probe that
    memorises 3,000 embeddings would look excellent and predict nothing.
    """
    C = float(getattr(CFG, "probe_C", 100.0)) if C is None else C
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                          random_state=0, stratify=y)
    # No multi_class= argument: it was deprecated in sklearn 1.5 and REMOVED in
    # 1.9, where passing it is a TypeError rather than a warning. Multinomial is
    # the default for multiclass with lbfgs, so dropping it changes nothing and
    # works on both old and new versions - Kaggle's image moves without asking.
    clf = LogisticRegression(max_iter=3000, C=C, class_weight="balanced")
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)

    print(f"\nprobe: {len(Xtr)} train / {len(Xte)} held out, "
          f"{len(clf.classes_)} classes")
    print(classification_report(yte, pred, zero_division=0, digits=3))

    print("confusion (rows = truth, cols = predicted), held out:")
    labels = list(clf.classes_)
    cm = confusion_matrix(yte, pred, labels=labels)
    short = [c[:14] for c in labels]
    print("      " + "".join(f"{s:>6}" for s in short))
    for name, row in zip(short, cm):
        print(f"{name:>14}" + "".join(f"{v:>6d}" for v in row))

    # The number that matters for us specifically: can it tell anomalous from
    # normal at all? Class confusion is survivable, saying "normal" is not.
    anom_t = np.array([t != "normal" for t in yte])
    anom_p = np.array([p != "normal" for p in pred])
    tp = int((anom_t & anom_p).sum()); fn = int((anom_t & ~anom_p).sum())
    fp = int((~anom_t & anom_p).sum()); tn = int((~anom_t & ~anom_p).sum())
    print(f"\nanomalous vs normal (the decision stage 1 actually makes):")
    print(f"  recall {tp / max(tp + fn, 1):.3f}   "
          f"precision {tp / max(tp + fp, 1):.3f}   "
          f"TP={tp} FN={fn} FP={fp} TN={tn}")
    print(f"  for comparison, the written health score is INVERTED on 3 of the "
          f"4 test videos measured")
    return clf


PROBE = fit_probe(PROBE_X, PROBE_Y) if PROBE_X is not None and len(PROBE_X) else None


def probe_predict(emb) -> dict:
    """{class_name: probability} for one window's frames.

    Takes the same mean-pooled shape the probe was trained on, so callers must
    pass frames spanning PROBE_SPAN_SEC - handing it a 2s window would be the
    train/test mismatch this cell exists to avoid.
    """
    if PROBE is None:
        return {}
    v = emb.mean(0) if hasattr(emb, "mean") and getattr(emb, "ndim", 1) > 1 else emb
    v = v.detach().float().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
    p = PROBE.predict_proba(v.reshape(1, -1))[0]
    # str() on the keys: sklearn hands back numpy.str_, which subclasses str and
    # so passes every type check, then serialises into JSON as an object rather
    # than a string. Cast at the boundary instead of debugging it in the arena
    # submission file.
    return {str(c): float(v) for c, v in zip(PROBE.classes_, p.tolist())}


def probe_anomaly(emb) -> float:
    """P(anything is wrong) in [0, 1]. Higher is worse - note this is the
    OPPOSITE sign to health(), which is higher-is-better. Kept explicit rather
    than mimicking the health convention, because silently reusing that sign is
    how the inverted signal went unnoticed for so long."""
    p = probe_predict(emb)
    return float(1.0 - p.get("normal", 1.0)) if p else 0.0


def probe_shortlist(emb, k: int = 5) -> list[str]:
    """The k most likely ANOMALY classes, learned rather than written.

    Held out, the correct class is in the top 5 for 98.3% of anomalous clips
    (top-3: 88.4%). The written shortlist this replaces managed 34% at k=3
    against a 27% baseline for drawing three of eleven at random.
    """
    p = probe_predict(emb)
    if not p:
        return []
    ranked = sorted(((c, v) for c, v in p.items() if c != "normal"),
                    key=lambda kv: -kv[1])
    return [c for c, _ in ranked[:k]]


def probe_window(kept: list[dict], centre_t: float, cfg=None) -> dict:
    """Score the probe on PROBE_SPAN_SEC of video centred on centre_t.

    Costs no GPU. Stage 1 already encoded every kept frame and cell 5 now keeps
    those vectors on the records, so this is a mean over arrays that exist -
    which is also exactly the shape the probe was trained on (8 frames spanning
    16s, mean-pooled), rather than the ~2s window stage 2 uses.

    Returns {} when the probe is unavailable or nothing was sampled nearby, so
    every caller can treat an empty dict as "no opinion".
    """
    cfg = cfg or CFG
    if PROBE is None or not kept:
        return {}
    half = PROBE_SPAN_SEC / 2
    near = [k for k in kept
            if centre_t - half <= k["t"] <= centre_t + half and "emb" in k]
    if not near:
        return {}
    # take PROBE_FRAMES evenly across the span, matching how a training example
    # was built - not the first 8, which would bias to the start of the window
    if len(near) > PROBE_FRAMES:
        idx = np.linspace(0, len(near) - 1, PROBE_FRAMES).round().astype(int)
        near = [near[i] for i in sorted(set(idx.tolist()))]
    v = np.mean([k["emb"] for k in near], axis=0)
    return probe_predict(v)


def probe_curve(kept: list[dict], step_sec: float = 4.0, cfg=None) -> list[tuple]:
    """[(t, P(anomalous)), ...] across a whole video.

    The replacement for the health curve wherever a per-video signal is needed.
    Sampled every step_sec rather than per frame because the probe's unit is a
    16s span - scoring it at 2fps would return sixteen near-identical values.
    """
    cfg = cfg or CFG
    if PROBE is None or not kept:
        return []
    t0, t1 = kept[0]["t"], kept[-1]["t"]
    out = []
    t = t0
    while t <= t1:
        p = probe_window(kept, t, cfg)
        if p:
            out.append((round(float(t), 2), round(1.0 - p.get("normal", 1.0), 4)))
        t += step_sec
    return out
