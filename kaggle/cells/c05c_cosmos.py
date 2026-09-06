# =============================================================================
# 5c - Does a purpose-built anomaly model read T025 differently? (DIAGNOSTIC)
# =============================================================================
# Set COSMOS_ENABLED = True to run this. It is off by default because it is a
# question, not a stage: nothing downstream consumes it yet.
#
# THE QUESTION. On T025 the pipeline now has perfect coverage - all six real
# events get a VLM window inside them - every one of eleven classes on offer, an
# explicit instruction to prefer a cause over its effect, and frames delivered
# at native resolution cropped to the motion region. Under all of that,
# Qwen3-VL-4B writes the same sentence every time:
#
#     "A dense queue of vehicles is stopped or moving very slowly on the left
#      side of the highway."   -> traffic_congestion
#
# Ground truth says traffic_accident, six times. We have eliminated coverage
# (6/6 events windowed), frame filtering (density inside events matches
# outside), duration (2s and 16s give identical text) and resolution (640px and
# native give identical text). What remains is the model's reading of the
# footage.
#
# So ask a second, independently-trained model the same question. Cosmos-Embed1
# is 1B params LoRA-tuned on VAD-Reasoning - 1,755 videos across 24 anomaly
# categories - and it has a text tower, so it can be asked zero-shot with no
# probe to fit and no 40-minute re-embed. Two outcomes, both worth having:
#
#   Cosmos says traffic_accident  -> Qwen's reading is the problem, and building
#                                    Cosmos in properly is clearly worth 2 hours
#   Cosmos says congestion too    -> two independently-trained models agree
#                                    against the label, which changes what we
#                                    should be trying to fix
#
# Leaderboard context: entrant #23 scored 40.4 running this model bare, above
# our 37.5, and #16 got 47.1 with a LoRA on top.

COSMOS_ENABLED = False
COSMOS_ID = "nvidia/Cosmos-Embed1-448p-anomaly-detection"
COSMOS_FRAMES = 8          # the shape it was trained at
COSMOS_DIAG_VIDEOS = ["T025", "T032", "T026"]


def _shim_transformers_for_cosmos() -> list[str]:
    """Put back the helpers Cosmos's vendored QFormer imports from
    transformers.modeling_utils, which no longer exports them.

    The failure is a hard one and worth naming precisely:

        cannot import name 'apply_chunking_to_forward'
        from 'transformers.modeling_utils'

    Cosmos ships its own modeling code via trust_remote_code, written against a
    transformers where the BERT-era helpers still lived in modeling_utils. They
    moved to pytorch_utils and activations, and the compatibility re-exports
    were dropped - Kaggle's image is transformers 5.x, so the import fails at
    load time.

    Pinning an older transformers is not an option: Qwen3-VL needs a recent one,
    and downgrading to satisfy Cosmos would break stage 2. Pinning a model
    revision does not help either, since the vendored code is the same at every
    revision. So re-export what still exists, reimplement the one small pure
    function that does not, and leave loud stubs for the rest - all three prune
    helpers are used only by prune_heads(), which inference never calls, so a
    stub that raises is strictly better than a wrong implementation.
    """
    import transformers.modeling_utils as mu
    patched = []

    def _adopt(name, module_path):
        if hasattr(mu, name):
            return
        try:
            import importlib
            src = importlib.import_module(module_path)
            setattr(mu, name, getattr(src, name))
            patched.append(f"{name} <- {module_path}")
        except Exception:
            pass

    for n in ("apply_chunking_to_forward", "prune_linear_layer", "Conv1D",
              "meshgrid"):
        _adopt(n, "transformers.pytorch_utils")
    _adopt("get_activation", "transformers.activations")

    if not hasattr(mu, "find_pruneable_heads_and_indices"):
        def find_pruneable_heads_and_indices(heads, n_heads, head_size,
                                             already_pruned_heads):
            """Verbatim behaviour of the removed transformers helper."""
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head = head - sum(1 if h < head else 0
                                  for h in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index

        mu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
        patched.append("find_pruneable_heads_and_indices (reimplemented)")

    for n in ("prune_conv1d_layer", "prune_layer"):
        if not hasattr(mu, n):
            def _gone(*a, _n=n, **k):
                raise NotImplementedError(
                    f"{_n} was removed from transformers and is only reachable "
                    "through prune_heads(), which inference does not call. If "
                    "you are seeing this, something is pruning attention heads "
                    "and that needs a real implementation, not this stub.")
            setattr(mu, n, _gone)
            patched.append(f"{n} (stub)")

    return patched


def load_cosmos(model_id: str = COSMOS_ID):
    """Load Cosmos-Embed1, picking a dtype the GPU can actually run.

    The model card says bfloat16. bf16 needs sm_80 (Ampere) and Kaggle's T4 is
    sm_75, where it is emulated rather than native - so try fp16 first, which
    the T4 does have hardware for, and fall back to fp32. Getting this wrong is
    not an error message, it is a silently slow run.

    trust_remote_code=True is required: the architecture ships as custom code on
    the Hub rather than living in transformers, so the notebook needs internet
    enabled. That is a real precondition, not a detail.
    """
    from transformers import AutoModel, AutoProcessor
    import transformers
    _p = _shim_transformers_for_cosmos()
    print(f"cosmos: transformers {transformers.__version__}"
          + (f", shimmed {len(_p)} helpers: {', '.join(_p)}" if _p else ""))
    last = None
    for dt in (torch.float16, torch.float32):
        try:
            m = AutoModel.from_pretrained(model_id, trust_remote_code=True,
                                          torch_dtype=dt).to(DEVICE).eval()
            p = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            print(f"cosmos: {model_id} loaded as {dt}")
            return m, p, dt
        except Exception as e:
            last = e
            print(f"  {dt} failed: {str(e).splitlines()[0][:140]}")
    raise RuntimeError(f"could not load {model_id}: {last}")


@torch.no_grad()
def cosmos_video_embedding(frames_bgr: list, model, proc, dtype):
    """One 768-d L2-normalised embedding for a clip of BGR frames.

    Input to the processor is (B, T, C, H, W) with values in 0..1 - the model
    card's example transposes (1, T, H, W, 3) by (0, 1, 4, 2, 3) to get there.
    The processor handles the resize to 448.
    """
    if not frames_bgr:
        return None
    idx = np.linspace(0, len(frames_bgr) - 1, COSMOS_FRAMES).round().astype(int)
    picked = [frames_bgr[i] for i in idx]
    rgb = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in picked])  # T,H,W,3
    batch = np.transpose(rgb[None, ...], (0, 1, 4, 2, 3))                 # 1,T,3,H,W
    inputs = proc(videos=batch)
    inputs = {k: (v.to(DEVICE, dtype=dtype) if hasattr(v, "to") else v)
              for k, v in inputs.items()}
    out = model.get_video_embeddings(**inputs)
    v = getattr(out, "visual_proj", out)
    return v.float().cpu().numpy().reshape(-1)


@torch.no_grad()
def cosmos_class_embeddings(model, proc, dtype):
    """Text embeddings for the eleven anomaly classes plus normal.

    Phrased as short scene descriptions rather than bare label strings, because
    the model was trained against captions - "a traffic accident with collided
    vehicles" sits closer to its training distribution than
    "traffic_accident" does.
    """
    prompts = {
        "normal": "ordinary traffic flowing normally with nothing unusual",
        "traffic_accident": "a traffic accident, vehicles collided or overturned",
        "traffic_congestion": "heavy traffic congestion, a long queue of slow vehicles",
        "stalled_or_broken_down_vehicle": "a broken down vehicle stopped at the roadside",
        "vehicle_blocking_traffic": "a vehicle blocking the road so others cannot pass",
        "wrong_way_driving": "a vehicle driving the wrong way against the traffic",
        "road_spill_or_debris": "spilled cargo or debris scattered on the road surface",
        "waterlogging_or_flood": "a flooded road covered in standing water",
        "fire": "flames and fire burning",
        "smoke": "thick smoke rising",
        "fighting_or_violence": "people fighting, punching or brawling",
        "loitering_or_suspicious_presence": "a person loitering, standing around suspiciously",
    }
    names = list(prompts)
    inputs = proc(text=[prompts[n] for n in names])
    inputs = {k: (v.to(DEVICE) if hasattr(v, "to") else v) for k, v in inputs.items()}
    out = model.get_text_embeddings(**inputs)
    t = getattr(out, "text_proj", out)
    return names, t.float().cpu().numpy()


def cosmos_zeroshot(frames_bgr: list, model, proc, dtype, names, T):
    """{class: probability} by video-text similarity. No probe, no training."""
    v = cosmos_video_embedding(frames_bgr, model, proc, dtype)
    if v is None:
        return {}
    sims = T @ v
    e = np.exp((sims - sims.max()) * 100.0)      # 100 ~ the model's logit scale
    return dict(zip(names, (e / e.sum()).tolist()))


if not COSMOS_ENABLED:
    print("cell 5c: COSMOS_ENABLED is False - set it True to run the diagnostic")
else:
    # Free the VLM first if it is resident. Qwen-4B is ~9.5GB and SigLIP2 ~1.5GB;
    # adding a 1B model on a 16GB T4 is close enough to the edge that the answer
    # would be an OOM rather than a verdict. This cell is a diagnostic, so it may
    # cost a Qwen reload afterwards - cell 6 will notice and reload on its own.
    if "vlm" in globals():
        print(f"cosmos: freeing the VLM first "
              f"({free_cuda('vlm', 'vlm_proc', '_VLM_ID'):.2f} GB in use)")

    _cm, _cp, _cd = load_cosmos()
    _names, _T = cosmos_class_embeddings(_cm, _cp, _cd)
    print(f"cosmos: {len(_names)} class prompts embedded, "
          f"{_T.shape[1]}-d\n")

    for _vid in COSMOS_DIAG_VIDEOS:
        _path = VIDEO_PATHS.get(_vid)
        if _path is None:
            print(f"  ! {_vid} not found")
            continue
        _truth = GT_TEST[(GT_TEST.video_id == _vid)
                         & GT_TEST.start_time_sec.notna()]
        if _truth.empty:
            continue
        print(f"{_vid}  truth: {sorted(set(_truth.class_name))}")
        _kept, _ = sample_video(_path, CFG)
        for _, _t in _truth.iterrows():
            # the frames inside this real event, which is the fair test - we
            # already know coverage is not the problem here
            _win = [k["frame"] for k in _kept
                    if _t.start_time_sec <= k["t"] <= _t.end_time_sec]
            if len(_win) < 2:
                print(f"   {_t.start_time_sec:6.0f}-{_t.end_time_sec:<6.0f} "
                      f"only {len(_win)} frames, skipped")
                continue
            _p = cosmos_zeroshot(_win, _cm, _cp, _cd, _names, _T)
            _top = sorted(_p.items(), key=lambda kv: -kv[1])[:3]
            _hit = "CORRECT" if _top[0][0] == _t.class_name else ""
            print(f"   {_t.start_time_sec:6.0f}-{_t.end_time_sec:<6.0f} "
                  + "  ".join(f"{c[:22]}:{v:.2f}" for c, v in _top) + f"   {_hit}")
        print()
    print("If Cosmos names these correctly, building it in is worth the time.")
    print("If it agrees with Qwen, two independent models disagree with the label")
    print("and the thing to fix is somewhere else entirely.")
