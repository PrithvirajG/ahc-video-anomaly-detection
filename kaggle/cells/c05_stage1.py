# =============================================================================
# 4 - Stage 1: rule-deviation scoring over frame embeddings
# =============================================================================
# The always-on tier. No training. The key idea, from Cerberus: do NOT enumerate
# anomalies and match against them. Prompting an LLM for a list of possible
# anomalies and matching gave them 27.13% recall on ShanghaiTech and 21.81% on
# NWPU - enumeration misses most of what actually happens.
#
# Instead score each frame against a pool of *normal* rules (weight +1) and
# *perturbed* atomic action labels (weight -1), take the top-k by cosine
# similarity and sum:
#
#     health(x) = sum_{c in topk(x)} w_c * sim(x, c)
#     escalate  <=>  health(x) < threshold
#
# This also sidesteps CLIP's normal/abnormal text entanglement (Alert-CLIP,
# CVPR 2026): we never ask CLIP to compare "a normal street" against "an
# anomalous street", which it is measurably bad at. We ask which of many
# concrete descriptions the frame is nearest, and let the signs do the work.

import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

# --- the rule pool ------------------------------------------------------------
# Positive rules: what routine footage in these domains actually looks like.
# Deliberately behavioural and specific - "traffic flowing" not "a road".
NORMAL_RULES = [
    "vehicles driving steadily along a road in the same direction",
    "cars moving at a constant speed on a highway",
    "traffic flowing smoothly through an intersection",
    "vehicles waiting in an orderly queue at a red traffic light",
    "cars parked in marked bays in a car park",
    "a pedestrian walking along a pavement",
    "people walking calmly across a crossing",
    "a person waiting at a bus stop",
    "a cyclist riding along the side of the road",
    "an empty road with no vehicles",
    "an empty street at night lit by street lamps",
    "a quiet campus walkway with a few people walking",
    "people standing and talking in a group",
    "a delivery van stopped briefly at the kerb",
    "a motorcycle riding in its lane",
    "a bus stopping at a designated bus stay",
    "clear dry road surface with lane markings visible",
    "an aerial view of a city street with normal traffic",
    "a drone view of rooftops and roads with light traffic",
    "a roundabout with vehicles circulating normally",
    "a toll booth with cars passing through in turn",
    "a footpath beside a road with occasional pedestrians",
    "an open park area with people walking",
    "a railway platform with passengers waiting",
    "vehicles changing lanes normally in flowing traffic",
    "a construction site with normal work in progress",
    "a car indicating and turning at a junction",
    "trees and buildings beside a road",
    "an overhead view of a car park with stationary parked cars",
    "night traffic with headlights moving steadily",
]

# Negative rules: atomic action labels in the style of Moments in Time, which
# Cerberus uses as perturbed negatives. This is a working subset chosen for our
# twelve labels; the full MiT vocabulary is 339 and adding more is cheap - it is
# one forward pass of the text tower, done once.
PERTURBED_ACTIONS = [
    "crashing", "colliding", "overturning", "flipping", "derailing",
    "burning", "flaming", "smoking", "exploding", "erupting",
    "flooding", "submerging", "overflowing", "leaking", "spilling",
    "punching", "kicking", "fighting", "wrestling", "shoving",
    "falling", "collapsing", "stumbling", "tripping", "slipping",
    "running away", "fleeing", "chasing", "panicking", "scattering",
    "crowding", "stampeding", "swarming", "queueing motionless",
    "loitering", "lurking", "trespassing", "climbing a fence",
    "breaking", "smashing", "vandalising", "stealing",
    "skidding", "swerving", "reversing into traffic", "driving against traffic",
    "blocking the road", "stalling", "breaking down", "stranded",
    "towing", "rescuing", "evacuating", "carrying an injured person",
    "crying", "shouting", "screaming", "arguing",
    "collapsed on the ground", "lying motionless on the road",
    "debris scattered on the road", "smoke rising", "water covering the road",
]

RULES = NORMAL_RULES + PERTURBED_ACTIONS
RULE_W = torch.tensor([1.0] * len(NORMAL_RULES) + [-1.0] * len(PERTURBED_ACTIONS))

# --- encoder ------------------------------------------------------------------
try:
    processor = AutoProcessor.from_pretrained(CFG.encoder_id)
    encoder = AutoModel.from_pretrained(CFG.encoder_id, torch_dtype=DTYPE)
except Exception as e:
    print(f"{CFG.encoder_id} unavailable ({str(e).splitlines()[0][:120]});"
          " falling back to CLIP-B/16")
    CFG.encoder_id = "openai/clip-vit-base-patch16"
    processor = AutoProcessor.from_pretrained(CFG.encoder_id)
    encoder = AutoModel.from_pretrained(CFG.encoder_id, torch_dtype=DTYPE)

encoder = encoder.to(DEVICE).eval()
IS_SIGLIP = "siglip" in CFG.encoder_id.lower()
print(f"encoder: {CFG.encoder_id}  {sum(p.numel() for p in encoder.parameters()) / 1e6:.0f}M params")


def _as_tensor(out):
    """get_*_features returns a bare tensor on some transformers versions and a
    BaseModelOutputWithPooling on others. Kaggle's image pins its own version and
    it will not be the one this was written against, so normalise rather than
    depend on either."""
    if torch.is_tensor(out):
        return out
    for attr in ("pooler_output", "image_embeds", "text_embeds", "last_hidden_state"):
        v = getattr(out, attr, None)
        if v is not None:
            return v.mean(1) if v.ndim == 3 else v
    raise TypeError(f"cannot get embeddings from {type(out)}")


@torch.no_grad()
def embed_texts(texts, batch=64):
    """SigLIP requires padding='max_length' - it was trained with a fixed 64-token
    context and dynamic padding silently degrades the embeddings. CLIP does not
    care. Getting this wrong produces a pipeline that runs and scores noise."""
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        kw = {"padding": "max_length", "max_length": 64} if IS_SIGLIP else {"padding": True}
        inp = processor(text=chunk, return_tensors="pt", truncation=True, **kw)
        inp = {k: v.to(DEVICE) for k, v in inp.items()}
        feats = _as_tensor(encoder.get_text_features(**inp))
        out.append(F.normalize(feats.float(), dim=-1).cpu())
    return torch.cat(out)


@torch.no_grad()
def embed_images(pil_images, batch=32):
    _t0 = time.time()
    out = []
    for i in range(0, len(pil_images), batch):
        inp = processor(images=pil_images[i:i + batch], return_tensors="pt")
        pv = inp["pixel_values"].to(DEVICE, dtype=DTYPE)
        feats = _as_tensor(encoder.get_image_features(pixel_values=pv))
        out.append(F.normalize(feats.float(), dim=-1).cpu())
    _log_call("siglip2-encoder", (time.time() - _t0) * 1000)
    return torch.cat(out)


# Per-model call timings, for the arena submission's runtime_metadata.
# model_runtimes (call_count/total/avg/p50/p95/max per video). process_video()
# in cell 8 snapshots this before and after each video to get per-video stats.
CALL_LOG: dict[str, list[float]] = {}


def _log_call(model_name: str, elapsed_ms: float) -> None:
    CALL_LOG.setdefault(model_name, []).append(elapsed_ms)


RULE_EMB = embed_texts(RULES)
print(f"rule pool: {len(NORMAL_RULES)} normal (+1), {len(PERTURBED_ACTIONS)} perturbed (-1)")


def health(img_emb: torch.Tensor, topk: int | None = None) -> torch.Tensor:
    """health(x) = sum over the top-k nearest rules of w_c * sim(x, c).

    Low health means the frame's nearest neighbours in the rule pool are mostly
    perturbed actions - i.e. it does not look like anything we called normal.
    """
    topk = topk or CFG.topk
    sim = img_emb @ RULE_EMB.T                      # (N, R), both L2-normalised
    top_sim, top_idx = sim.topk(topk, dim=-1)
    return (top_sim * RULE_W[top_idx]).sum(-1)


def stage1_video(path, cfg=None):
    """Stage 0 + stage 1 over one video. Returns per-kept-frame records."""
    cfg = cfg or CFG
    kept, n_seen = sample_video(path, cfg)
    if not kept:
        return [], n_seen
    imgs = [to_pil(draw_visual_prompt(k["frame"], k["box"], cfg.visual_prompt)) for k in kept]
    emb = embed_images(imgs)
    h = health(emb)
    # Keep the embedding, not just the scalar derived from it. The probe in cell
    # 5b needs 8 frames mean-pooled over 16s, and these are exactly those frames
    # already encoded - so retaining them makes probe scoring cost ZERO extra
    # GPU rather than a second pass. About 2.7 MB for the longest test video
    # (870 kept frames x 768 floats), which is nothing against the frames
    # themselves already held in `kept`.
    _e = emb.detach().float().cpu().numpy()
    for k, hv, ev in zip(kept, h.tolist(), _e):
        k["health"] = hv
        k["emb"] = ev
    return kept, n_seen


# --- calibrate the threshold on normal training footage -----------------------
# Picking a health threshold by eye is guesswork; the distribution differs per
# encoder and per rule pool. Instead measure health on footage we KNOW is normal
# and set the cut so escalate_pct of it escalates. That makes escalate_pct a
# compute budget - "stage 2 runs on ~12% of frames" - rather than a magic number.
def calibrate(n_videos: int = 12, cfg=None):
    cfg = cfg or CFG
    if GT_TRAIN.empty:
        print("no training ground truth - leaving health_thresh unset")
        return None
    normals = (GT_TRAIN[(GT_TRAIN["class_name"] == "normal") & GT_TRAIN["path"].notna()]
               .drop_duplicates("video_id").head(n_videos))
    if normals.empty:
        print("no normal videos found - leaving health_thresh unset")
        return None

    scores = []
    t0 = time.time()
    for _, row in normals.iterrows():
        kept, _ = stage1_video(row["path"], cfg)
        scores += [k["health"] for k in kept]
    if not scores:
        return None

    arr = np.array(scores)
    thr = float(np.percentile(arr, cfg.escalate_pct))
    cfg.health_thresh = thr
    print(f"calibrated on {len(normals)} normal videos, {len(arr)} frames, "
          f"{time.time() - t0:.0f}s")
    print(f"  health: mean {arr.mean():.3f}  p1 {np.percentile(arr, 1):.3f}  "
          f"p50 {np.percentile(arr, 50):.3f}  p99 {np.percentile(arr, 99):.3f}")
    print(f"  health_thresh = {thr:.4f}  (escalates the lowest "
          f"{cfg.escalate_pct:.0f}% of normal frames)")
    return thr


calibrate()
