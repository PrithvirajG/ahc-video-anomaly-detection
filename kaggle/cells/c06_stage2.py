# =============================================================================
# 5 - Stage 2: small VLM verification on escalated windows only
# =============================================================================
# Runs on the ~12% of frames stage 1 could not clear. Two things make this
# cheaper and more accurate than "show the VLM a frame and ask if it is weird":
#
# 1. SHORTLISTING. Stage 1's frame embedding already ranks the twelve labels.
#    We only ask about the top few, so the prompt stays short and the model is
#    not invited to hallucinate its way through nine irrelevant options.
#
# 2. ASK-HINT PROMPTING (WACV 2026). Abstract prompts fail where action-centric
#    ones succeed - "Is there any anomaly?" misses what "Do you see punching,
#    kicking, or wrestling on the ground?" catches on the same input. So every
#    label expands into concrete, observable questions rather than being handed
#    to the model as a bare class string. This is a text file, not a training
#    run: the highest accuracy-per-minute available today.

import subprocess
import sys

from transformers import AutoProcessor as VLMProcessor

# Descriptions used for the stage-1 shortlist (embedding space, not the VLM).
CLASS_DESCRIPTIONS = {
    "traffic_accident": [
        "a car crash with damaged vehicles on the road",
        "two vehicles collided at an intersection",
        "an overturned vehicle on its side after a crash",
    ],
    "traffic_congestion": [
        "a long queue of stationary vehicles filling the road",
        "heavy traffic jam with cars bumper to bumper",
    ],
    "stalled_or_broken_down_vehicle": [
        "a single vehicle stopped on the hard shoulder with hazard lights",
        "a broken down car stationary in a live traffic lane",
    ],
    "vehicle_blocking_traffic": [
        "a vehicle parked across the road obstructing other cars",
        "a truck blocking a junction so traffic cannot pass",
    ],
    "wrong_way_driving": [
        "a vehicle driving towards oncoming traffic",
        "a car travelling the wrong way down a one way road",
    ],
    "road_spill_or_debris": [
        "debris and scattered objects lying across the road surface",
        "a spilled load of cargo covering the carriageway",
    ],
    "waterlogging_or_flood": [
        "a road submerged under standing flood water",
        "vehicles driving through deep water on a flooded street",
    ],
    "fire": [
        "an open flame burning on a vehicle or building",
        "a fire with visible orange flames in the scene",
    ],
    "smoke": [
        "thick smoke rising and spreading across the scene",
        "a plume of grey smoke obscuring the view",
    ],
    "fighting_or_violence": [
        "two people physically fighting and throwing punches",
        "a violent altercation between people in the street",
    ],
    "loitering_or_suspicious_presence": [
        "a person lingering in a restricted area for a long time",
        "someone loitering near parked vehicles at night",
    ],
}

# ASK-Hint question banks. Concrete and observable - each one should be
# answerable by looking, without inference about intent.
ASK_HINT = {
    "traffic_accident": [
        "Do you see two or more vehicles in contact, or a vehicle that has struck something?",
        "Is any vehicle visibly damaged, overturned, or off its wheels?",
        "Are people gathered around a stopped vehicle in the roadway?",
    ],
    "traffic_congestion": [
        "Is there a dense queue of vehicles that are stopped or barely moving?",
        "Does the queue extend across most of the visible road?",
    ],
    "stalled_or_broken_down_vehicle": [
        "Is a single vehicle stationary while other traffic moves past it?",
        "Is it stopped on a shoulder, in a live lane, or somewhere vehicles do not normally park?",
        "Are hazard lights on, a bonnet open, or a warning triangle placed?",
    ],
    "vehicle_blocking_traffic": [
        "Is a vehicle positioned so that other vehicles cannot get past?",
        "Is a vehicle stopped across a junction, crossing, or lane?",
    ],
    "wrong_way_driving": [
        "Is any vehicle facing or moving opposite to the other vehicles around it?",
        "Is a vehicle on the wrong side of a divided road or driving against arrows and markings?",
    ],
    "road_spill_or_debris": [
        "Are there objects, rubble, cargo, or scattered material on the road surface?",
        "Are vehicles swerving or slowing to avoid something lying on the road?",
    ],
    "waterlogging_or_flood": [
        "Is part of the road covered by standing water?",
        "Are vehicle wheels partly submerged, or is water rippling across the surface?",
    ],
    "fire": [
        "Do you see open flames anywhere in the scene?",
        "Is a vehicle, building, or pile of material actively burning?",
    ],
    "smoke": [
        "Do you see smoke rising or drifting across the scene?",
        "Is visibility reduced by a plume of smoke rather than by fog or rain?",
    ],
    "fighting_or_violence": [
        "Do you see punching, kicking, grappling, or pushing between people?",
        "Is anyone on the ground while others stand over them?",
        "Is a crowd reacting to or surrounding a physical confrontation?",
    ],
    "loitering_or_suspicious_presence": [
        "Is a person remaining in one place for an unusually long time?",
        "Is someone lingering near vehicles, doors, or fences without an obvious purpose?",
        "Is a person in an area that is otherwise empty of people?",
    ],
}

CLASS_EMB_TEXTS, CLASS_EMB_OWNER = [], []
for cls, descs in CLASS_DESCRIPTIONS.items():
    CLASS_EMB_TEXTS += descs
    CLASS_EMB_OWNER += [cls] * len(descs)
CLASS_EMB = embed_texts(CLASS_EMB_TEXTS)
CLASS_OWNER = np.array(CLASS_EMB_OWNER)


def shortlist_classes(img_emb: torch.Tensor, k: int = 5) -> list[str]:
    """Rank the eleven anomaly labels for a window by max similarity.

    Note this is NOT used as a detector - Alert-CLIP shows CLIP-family text
    embeddings for normal vs abnormal are entangled enough that raw similarity
    is a poor yes/no. It is used only to decide which questions to ask, where
    being roughly right is sufficient and being wrong just wastes a question.

    That last sentence was false for the whole first run, and it cost us most of
    our score. See build_prompt() below: the shortlist was also injected into
    the required JSON schema, so being wrong did not waste a question - it
    deleted the correct answer. The shortlist is now a hint and nothing else,
    which is what this docstring always claimed.

    k is 5 rather than 3 because the shortlist no longer restricts anything, so
    a wider hint costs only prompt length. It was measured at 34% hit rate for
    k=3 against a 27% random baseline, i.e. very nearly uninformative; widening
    it is a stopgap until the linear probe replaces this ranking entirely.
    """
    # Prefer the learned ranking when cell 5b produced one. Held out, the correct
    # class is in the probe's top 5 for 98.3% of anomalous clips against 34% for
    # this text-similarity ranking at k=3 - which is barely above the 27% you get
    # by drawing three of eleven at random. The text version stays as the
    # fallback for a run where the probe could not be fitted.
    if "probe_shortlist" in globals() and PROBE is not None:
        learned = probe_shortlist(img_emb, k=k)
        if learned:
            return learned
    sim = (img_emb.mean(0, keepdim=True) @ CLASS_EMB.T).squeeze(0)
    best = {}
    for s, owner in zip(sim.tolist(), CLASS_OWNER):
        best[owner] = max(best.get(owner, -9.9), s)
    return [c for c, _ in sorted(best.items(), key=lambda kv: -kv[1])[:k]]


# --- load the VLM -------------------------------------------------------------
# sdpa, not flash-attention-2: FA2 needs sm_80+ and Kaggle's T4 is sm_75. Asking
# for it fails at load, not at generate, which is at least an honest error.
def load_vlm(model_id=None):
    """Qwen3-VL-4B, not Qwen2.5-VL-3B and not Qwen3-VL-8B.

    3B was a compromise for the GTX 1650's 4GB VRAM ceiling - irrelevant on a
    T4 (16GB). Measured VRAM: Qwen3-VL-4B ~9-10GB fp16 (comfortable alongside
    SigLIP2's ~1.5GB), Qwen3-VL-8B ~19GB fp16 / ~12GB in 4-bit ("on the edge"
    per multiple sources - not worth the OOM risk on a live run). Qwen3-VL adds
    video-specific architecture (interleaved MRoPE, textual timestamps,
    temporally dense captions) that Qwen2.5-VL lacks, and beats Qwen2.5-VL-7B on
    11/12 shared benchmarks including the video ones (CharadesSTA, LVBench).

    Independently validated for THIS exact task: QVAD (arXiv:2604.03040), a
    training-free VAD paper in the organizers' own SOTA deck, uses
    Qwen3-VL-4B-Instruct for captioning. AI City Challenge 2026 Track 3
    (traffic anomalies) had 2 of the top 3 accepted-paper teams on Qwen3-VL-8B.

    Needs transformers>=4.57.0 (Qwen3-VL shipped Oct 2025); cell 5 may have
    already imported an older version, so upgrade defensively and fall back to
    Qwen2.5-VL-3B (known-good) rather than leave the notebook dead mid-session.
    """
    model_id = model_id or CFG.vlm_id
    try:
        from transformers import Qwen3VLForConditionalGeneration as VLMClass
    except ImportError:
        print("transformers too old for Qwen3-VL - upgrading (needs >=4.57.0)...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                        "transformers>=4.57.0"], check=False)
        try:
            from transformers import Qwen3VLForConditionalGeneration as VLMClass
        except ImportError as e:
            print(f"still unavailable after upgrade ({e}); "
                  "falling back to Qwen2.5-VL-3B-Instruct")
            model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
            from transformers import Qwen2_5_VLForConditionalGeneration as VLMClass

    proc = VLMProcessor.from_pretrained(model_id)
    # Cap the vision token count. This family is resolution-native, so an
    # uncapped 640px frame can cost >1500 tokens per image; at 4 images per
    # window that alone decides whether this fits on a T4 and whether it is
    # 3 fps or 0.5 fps.
    if hasattr(proc, "image_processor"):
        proc.image_processor.min_pixels = 256 * 28 * 28
        proc.image_processor.max_pixels = 768 * 28 * 28
    m = VLMClass.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if USE_FP16 else torch.float32,
        attn_implementation="sdpa",
        device_map="auto" if DEVICE == "cuda" else None,
        low_cpu_mem_usage=True,
    ).eval()
    return m, proc


vlm, vlm_proc = load_vlm()
print(f"vlm: {CFG.vlm_id} loaded on {DEVICE}")
if DEVICE == "cuda":
    print(f"     VRAM in use: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


SYSTEM = (
    "You are a video surveillance analyst reviewing a few consecutive frames from "
    "one camera. Answer only from what is visible. If the scene looks like ordinary "
    "activity, say so - false alarms are as costly as missed events."
)


# A queue of stopped cars is what an accident LOOKS like from above once the
# first second is over, and the label is the accident. Measured on T025, where
# the truth is six traffic_accident events and, given all eleven classes and no
# constraint, the model wrote:
#
#   "A dense queue of vehicles is stopped or moving very slowly on the highway."
#   "A long queue of trucks and cars is stationary at the service area entrance,
#    indicating traffic congestion."
#
# Those are accurate descriptions and the wrong answer. The ASK-HINT questions
# for traffic_accident are good ones - vehicles in contact, visible damage,
# people gathered - and the model answered them honestly with "no", then picked
# the class whose questions it could answer "yes" to. Nothing about that is a
# failure of prompting the individual classes; what was missing is an ordering
# between them.
#
# Note what this deliberately is NOT: a lookup table mapping consequence to
# cause. We rejected that earlier and the reasoning still holds - smoke over a
# wrecked car is a symptom, smoke over a thermal plant is the incident, and no
# table can tell those apart. This asks the model to LOOK for a cause before
# settling on an effect, and leaves the judgement where the eyes are.
CAUSE_BEFORE_EFFECT = (
    "One ordering rule. Stopped traffic, a queue, a blocked lane and a crowd are "
    "usually consequences of something else. If you see one, look at the head of "
    "the queue or the centre of the crowd before you answer: a collision, a "
    "damaged or overturned vehicle, debris, water or fire there is the incident, "
    "and the queue is only its effect. Report the cause when you can see one, and "
    "the effect only when you cannot."
)


def build_prompt(hint_classes: list[str]) -> str:
    """Ask about the shortlisted classes; accept an answer from all eleven.

    hint_classes chooses which ASK-HINT question banks to spell out. It does NOT
    restrict what the model may answer - the schema below always offers every
    anomaly class plus "normal".

    That separation is the single highest-value fix in this project, because the
    previous version collapsed it. `candidates` went into the required JSON
    schema, so the model had to reply with one of three classes or "normal".
    Measured on the practice pack, over the 47 windows that actually overlapped
    a real ground-truth event:

        correct class present in the 3-way shortlist   34%   (random 3-of-11: 27%)
        model answered "normal"                        79%
        right class never on the menu at all           31 of 47   (66%)

    So two thirds of our misses were a multiple-choice question with the correct
    answer removed, and "normal" was the only remaining option that was not
    definitionally wrong. Of 26 ground-truth timed events, 23 had a window
    overlapping them and only 3 had a window of the right class - we were
    looking at 88% of real events and recognising 12%.

    The question banks stay shortlisted (top 5, not all 11) so the prompt does
    not quadruple in length and dilute attention across sixty-odd questions.
    """
    lines = [
        "These frames were flagged by an automatic filter. Decide whether they show "
        "a genuine incident that a responder should be sent to.",
        "",
        "These checks are the most likely possibilities, not the only ones - if "
        "what you see is a different kind of incident, name that instead:",
    ]
    for cls in hint_classes:
        lines.append(f"\n[{cls}]")
        lines += [f"  - {q}" for q in ASK_HINT.get(cls, [])]
    lines += [
        "",
        "If a red circle or square is drawn on a frame, it marks where motion was "
        "detected - look there first, but judge the whole frame.",
        "",
        CAUSE_BEFORE_EFFECT,
        "",
        "Reply with JSON only, no other text:",
        '{"anomaly": true|false, "class": "<one of: '
        + ", ".join(ANOMALY_CLASSES + ["normal"]) + '>", '
        '"confidence": <0.0-1.0>, "description": "<one short sentence>"}',
    ]
    return "\n".join(lines)


def resolve_class(raw: str) -> str:
    """Map a model's class string onto one of the twelve, tolerantly.

    Exact-match-or-normal was safe while the schema offered three options the
    model could copy verbatim. Now that it chooses freely from eleven, a reply
    of "traffic accident" or "Traffic_Accident" would be silently scored as
    normal - reintroducing the same failure this change exists to remove, just
    one layer further down. Normalise separators and case before giving up.
    """
    s = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    s = re.sub(r"_+", "_", s).strip("_")
    if s in CLASSES:
        return s
    squashed = {c.replace("_", ""): c for c in CLASSES}
    return squashed.get(s.replace("_", ""), "normal")


@torch.no_grad()
def vlm_pick_class(pil_frames: list, options: list[str]) -> dict | None:
    """Forced choice among `options`. No "normal", because the caller already
    decided something IS happening.

    This exists for exactly one measured failure. On T025 the probe correctly
    localises five of six real accidents - probe-span extents score IoU 0.800
    against the ground truth - and then names every one of them
    wrong_way_driving. Extent right, class wrong, five events lost.

    The division of labour that fixes it: the PROBE decides WHETHER (98% recall,
    from 16s of temporal context), the VLM decides WHICH (it can actually see).
    Asking the probe to do both wastes the VLM, and the probe's top-1 is 0.739
    against 0.884 for its top-3 - so handing the VLM those three and making it
    choose is worth about fifteen points of class accuracy if the VLM can pick
    at all.

    Returns None on any failure, so the caller keeps the probe's own answer.
    """
    if not options:
        return None
    # `options` narrows WHICH QUESTIONS get spelled out. It does NOT narrow the
    # answer - the schema below offers all eleven.
    #
    # The first version of this function did narrow the answer, and it rebuilt
    # the exact bug cell 6 exists to fix, one layer down. Measured on T025:
    # traffic_accident was in the probe's top 3 for 1 window out of 12, so the
    # re-ask could not answer it however clearly the frames showed one. The VLM
    # dutifully picked stalled_or_broken_down_vehicle off the menu it was given,
    # and five events that pass the IoU gate at 0.82-0.98 stayed wrong.
    #
    # A shortlist is a hint about where to look. The moment it reaches the reply
    # schema it stops being a hint and starts deleting correct answers.
    lines = [
        "Something in these frames has been flagged as an incident by an "
        "automatic system, and you should assume it is right about that.",
        "",
        "Your job is to say WHICH incident it is. These are the most likely "
        "candidates, but you may answer with any class in the list at the end:",
    ]
    for c in options:
        lines.append(f"\n[{c}]")
        lines += [f"  - {q}" for q in ASK_HINT.get(c, [])]
    lines += [
        "",
        CAUSE_BEFORE_EFFECT,
        "",
        "Pick the single best fit even if you are unsure. Reply with JSON only:",
        '{"class": "<one of: ' + ", ".join(ANOMALY_CLASSES) + '>", '
        '"confidence": <0.0-1.0>, "description": "<one short sentence>"}',
    ]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user", "content": [{"type": "image"} for _ in pil_frames]
         + [{"type": "text", "text": "\n".join(lines)}]},
    ]
    try:
        text = vlm_proc.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)
        inputs = vlm_proc(text=[text], images=pil_frames, return_tensors="pt",
                          padding=True)
        inputs = {k: (v.to(vlm.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        out = vlm.generate(**inputs, max_new_tokens=CFG.vlm_max_new_tokens,
                           do_sample=False, temperature=None, top_p=None, top_k=None)
        gen = out[0][inputs["input_ids"].shape[1]:]
        d = parse_json_reply(vlm_proc.decode(gen, skip_special_tokens=True))
    except Exception:
        return None
    # It may still answer "normal" despite not being offered it - that is the
    # model declining the premise, and the caller's probe evidence outranks a
    # refusal to choose, so treat it as no opinion rather than as a veto.
    return d if d.get("class") in ANOMALY_CLASSES else None


def parse_json_reply(text: str) -> dict:
    """Small models wrap JSON in prose or fences often enough that a bare
    json.loads is a reliability bug, not a shortcut."""
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            cls = resolve_class(d.get("class", "normal"))
            conf = float(d.get("confidence", 0.0))
            return {
                "anomaly": bool(d.get("anomaly", False)) and cls != "normal",
                "class": cls,
                "confidence": max(0.0, min(1.0, conf)),
                "description": str(d.get("description", ""))[:300],
                "raw": text,
            }
        except Exception:
            pass
    # Fall back to keyword rescue rather than dropping the window entirely.
    low = text.lower()
    hit = next((c for c in ANOMALY_CLASSES if c.replace("_", " ") in low), None)
    return {"anomaly": hit is not None, "class": hit or "normal",
            "confidence": 0.4 if hit else 0.0, "description": text.strip()[:300],
            "raw": text}


@torch.no_grad()
def vlm_verify(pil_frames: list, candidates: list[str]) -> dict:
    _t0 = time.time()
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user", "content": [{"type": "image"} for _ in pil_frames]
         + [{"type": "text", "text": build_prompt(candidates)}]},
    ]
    text = vlm_proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = vlm_proc(text=[text], images=pil_frames, return_tensors="pt", padding=True)
    inputs = {k: (v.to(vlm.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    out = vlm.generate(**inputs, max_new_tokens=CFG.vlm_max_new_tokens,
                       do_sample=False, temperature=None, top_p=None, top_k=None)
    gen = out[0][inputs["input_ids"].shape[1]:]
    result = parse_json_reply(vlm_proc.decode(gen, skip_special_tokens=True))
    # "vision-language-model" matches the PDF's own model_runtimes example name.
    _log_call("vision-language-model", (time.time() - _t0) * 1000)
    return result


ADJUDICATE_SYSTEM = (
    "You are a video surveillance analyst. Several separate observations were made "
    "at the same location within a short time. Decide what single incident best "
    "explains them together, judging only from the frames."
)


@torch.no_grad()
def adjudicate_primary(pil_frames: list, observations: list[dict]) -> dict | None:
    """Given several class verdicts inside one temporal cluster, pick the ONE
    primary incident - by asking the model, not by consulting a causal table.

    Why not a table: a static cause->consequence map has to decide once and for
    all what smoke "means", and smoke is a symptom over a wrecked car but the
    primary event over a thermal plant or a hillside. The same class changes
    role with context, so any fixed tree is wrong in whichever context it did
    not anticipate. The VLM already sees the context, so it is the right thing
    to ask - one extra call per multi-class cluster, a handful per video.

    Returns None on any failure; the caller then falls back to the highest
    confidence observation, so this can only improve on that baseline.
    """
    seen = []
    for o in sorted(observations, key=lambda o: o["t0"]):
        seen.append(f"  - at {o['t0']:.0f}s: {o['class']} ({o['confidence']:.2f}) "
                    f"- {o.get('description', '')[:110]}")
    prompt = "\n".join([
        "These observations were made at one location, in this order:",
        *seen,
        "",
        "They may be several views of ONE incident, or genuinely separate things.",
        "Pick the single class that best describes the primary incident here. If "
        "the observations are consequences of something else visible in the frames "
        "(for example smoke and a gathered crowd around damaged vehicles), name "
        "that underlying incident instead.",
        "",
        "Reply with JSON only:",
        '{"primary": "<one of: ' + ", ".join(ANOMALY_CLASSES) + '>", '
        '"confidence": <0.0-1.0>, "reason": "<one short sentence>"}',
    ])
    try:
        _t0 = time.time()
        messages = [
            {"role": "system", "content": [{"type": "text", "text": ADJUDICATE_SYSTEM}]},
            {"role": "user", "content": [{"type": "image"} for _ in pil_frames]
             + [{"type": "text", "text": prompt}]},
        ]
        text = vlm_proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = vlm_proc(text=[text], images=pil_frames, return_tensors="pt", padding=True)
        inputs = {k: (v.to(vlm.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        out = vlm.generate(**inputs, max_new_tokens=120, do_sample=False,
                           temperature=None, top_p=None, top_k=None)
        gen = out[0][inputs["input_ids"].shape[1]:]
        raw = vlm_proc.decode(gen, skip_special_tokens=True)
        _log_call("vision-language-model", (time.time() - _t0) * 1000)

        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return None
        d = json.loads(m.group(0))
        cls = str(d.get("primary", "")).strip()
        if cls not in ANOMALY_CLASSES:
            return None
        return {"primary": cls,
                "confidence": max(0.0, min(1.0, float(d.get("confidence", 0.5)))),
                "reason": str(d.get("reason", ""))[:200]}
    except Exception as e:
        print(f"  ! adjudication failed: {str(e).splitlines()[0][:100]}")
        return None


# --- smoke test ---------------------------------------------------------------
if _probe is not None:
    _k, _ = stage1_video(_probe)
    if _k:
        _worst = sorted(_k, key=lambda r: r["health"])[:CFG.vlm_frames]
        _worst = sorted(_worst, key=lambda r: r["t"])
        _pil = [to_pil(draw_visual_prompt(r["frame"], r["box"], CFG.visual_prompt)) for r in _worst]
        _emb = embed_images(_pil)
        _cands = shortlist_classes(_emb)
        _t0 = time.time()
        _res = vlm_verify(_pil, _cands)
        print(f"\n{_probe.name}  candidates={_cands}  ({time.time() - _t0:.1f}s)")
        print(json.dumps({k: v for k, v in _res.items() if k != "raw"}, indent=2))
