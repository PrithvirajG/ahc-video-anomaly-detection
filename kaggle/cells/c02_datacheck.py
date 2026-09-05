# =============================================================================
# 2 - Is the dataset actually there?
# =============================================================================
# Verification only. Nothing here downloads anything - the pack is attached as a
# Kaggle Dataset and mounts read-only under /kaggle/input.
#
# This cell exists because "the dataset is attached" and "the dataset has videos
# in it" are different claims, and the gap between them is silent: an empty or
# wrong mount indexes to zero videos and the failure only surfaces several cells
# later as a confusing error in the encoder.

from pathlib import Path

# =============================================================================
# WHICH DATASET DOES THIS RUN USE?
# =============================================================================
#   "practice" - the public train/test pack. 34 test videos named T0xx, with
#                ground truth, so cell 9 can score the run and we can iterate.
#   "eval"     - the private evaluation pack. 28 videos named E0xx under
#                L1/L2/L3, NO ground truth (withheld on purpose), so cell 9
#                cannot score anything and cell 10's JSON is the only output.
#
# Only this line changes between the two. Everything downstream reads GT_TEST,
# which cell 3 builds from whichever source this selects - so the pipeline
# itself has no idea which mode it is in, and practice mode behaves exactly as
# it did before this flag existed.
#
# In "eval", attach BOTH datasets on Kaggle:
#     prithvirajgotepatil/ahc-visual-intelligence-eval         <- the videos
#     prithvirajgotepatil/ahc-visual-intelligence-train-test   <- calibration
# Cell 5 measures health_thresh on known-normal TRAINING footage. Without the
# second dataset that threshold stays unset and cells 7/8/11 silently fall back
# to a hard-coded -0.4 that was never measured on anything.
MODE = "practice"    # "practice" | "eval"

assert MODE in ("practice", "eval"), f"MODE must be 'practice' or 'eval', not {MODE!r}"

# The twelve label strings. Scoring compares the string, so these are copied
# exactly from the problem statement and must never be "tidied up".
CLASSES = [
    "normal",
    "traffic_accident",
    "traffic_congestion",
    "stalled_or_broken_down_vehicle",
    "vehicle_blocking_traffic",
    "wrong_way_driving",
    "road_spill_or_debris",
    "waterlogging_or_flood",
    "fire",
    "smoke",
    "fighting_or_violence",
    "loitering_or_suspicious_presence",
]
ANOMALY_CLASSES = [c for c in CLASSES if c != "normal"]

ON_KAGGLE = Path("/kaggle").exists()
WORK = Path("/kaggle/working") if ON_KAGGLE else Path.cwd()


def find_data_roots(max_depth: int = 8) -> list[Path]:
    """Find EVERY directory holding a train/ or test/ - there may be several.

    Two mount shapes have to work, and one of them is not a single tree:

      /kaggle/input/<slug>/                      "Add data"
      /kaggle/input/datasets/<owner>/<slug>/     kagglehub

    ...and when the pack was uploaded as multiple zips, Kaggle extracts each
    archive into its OWN directory named after that archive, so the real mount
    can be five levels down:

      /kaggle/input/datasets/<owner>/<slug>/ahc_part06/Train and Test/train/...

    Google Drive split the pack arbitrarily, so one class folder can have its
    ground_truth.csv in one part and some of its videos in another. There is
    therefore no single "correct root"; the pack is the UNION of these trees,
    which is why this returns a list.

    Depth is not hard-coded, because guessing it kept being wrong - kagglehub
    costs three levels, the archive a fourth, Drive's own folder a fifth. Walk
    and prune instead: a directory containing train/ or test/ IS a root, and
    there is never a reason to descend past one.
    """
    roots: list[Path] = []

    def walk(d: Path, depth: int):
        if depth > max_depth:
            return
        try:
            subdirs = [p for p in d.iterdir() if p.is_dir()]
        except OSError:
            return
        names = {p.name for p in subdirs}
        if "train" in names or "test" in names:
            # resolve() before the dedupe: the bases overlap ("data" and
            # WORK/"data" are one directory), and two Path objects for the same
            # directory are not equal, so an unresolved check counts it twice
            # and every audit number silently doubles.
            r = d.resolve()
            if r not in roots:
                roots.append(r)
            return
        for p in subdirs:
            if p.name in ("videos", "__MACOSX", ".ipynb_checkpoints"):
                continue
            walk(p, depth + 1)

    for base in (Path("/kaggle/input"), WORK / "data", Path("data")):
        if base.exists():
            walk(base, 0)
    return roots


EVAL_LEVEL_DIRS = {"L1", "L2", "L3"}


def find_eval_roots(max_depth: int = 8) -> list[Path]:
    """Find every directory holding the evaluation pack's L1/L2/L3 levels.

    The evaluation pack has a different shape from the train/test pack, and the
    difference is silent rather than loud: no train/, no test/, and no
    ground_truth.csv anywhere - its README says truth is withheld on purpose -
    just L1/, L2/, L3/, each containing videos/ and videos.csv.

    find_data_roots() above matches on a train/ or test/ sibling, so it returns
    [] for this tree. That is not an error anyone sees: it indexes zero videos
    and the run completes having processed nothing. Hence a second finder rather
    than a looser first one - the two shapes stay distinguishable, which is what
    lets both packs be attached at once in eval mode.
    """
    roots: list[Path] = []

    def walk(d: Path, depth: int):
        if depth > max_depth:
            return
        try:
            subdirs = [p for p in d.iterdir() if p.is_dir()]
        except OSError:
            return
        names = {p.name for p in subdirs}
        # every level need not be present - a partial mount is still a root,
        # and saying so beats reporting "no data" for a tree that has videos
        if names & EVAL_LEVEL_DIRS:
            r = d.resolve()
            if r not in roots:
                roots.append(r)
            return
        for p in subdirs:
            if p.name in ("videos", "__MACOSX", ".ipynb_checkpoints"):
                continue
            walk(p, depth + 1)

    for base in (Path("/kaggle/input"), WORK / "eval", Path("eval")):
        if base.exists():
            walk(base, 0)
    return roots


def looks_like_drive_html(p: Path) -> bool:
    """Kaggle's 'link a Google Drive URL' importer cannot authenticate and cannot
    walk a folder - it does a plain GET and stores whatever comes back. Pointed
    at a Drive folder it yields one ~360KB file named after the folder id whose
    content is the Drive *web page*. It mounts perfectly happily and contains no
    data, so name it rather than letting it look like an empty dataset."""
    try:
        if p.is_file() and p.stat().st_size < 2_000_000:
            head = p.read_bytes()[:400].lower()
            return b"<!doctype html" in head and b"google drive" in head
    except OSError:
        pass
    return False


DATA_ROOTS = find_data_roots()
DATA_ROOT = DATA_ROOTS[0] if DATA_ROOTS else (WORK / "data")   # first, for messages
EVAL_ROOTS = find_eval_roots()

print("=" * 74)
print(f"  MODE: {MODE.upper()}" + ("   - private evaluation pack, no ground truth"
                                   if MODE == "eval" else
                                   "   - public train/test pack, scored by cell 9"))
print("=" * 74)

print("attached under /kaggle/input:")
root = Path("/kaggle/input")
if root.exists():
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        files = [f for f in d.rglob("*") if f.is_file()]
        vids = [f for f in files if f.suffix.lower() == ".mp4"]
        size = sum(f.stat().st_size for f in files)
        print(f"  {d.name:38s} {len(vids):5d} mp4   {size / 1e6:9.1f} MB")
        for f in files:
            if looks_like_drive_html(f):
                print(f"     ! {f.name}")
                print("       This is a Google Drive WEB PAGE, not the dataset.")
                print("       Kaggle's URL importer cannot authenticate or walk a")
                print("       folder, so it saved the HTML it was served.")
else:
    print("  (not running on Kaggle)")

# --- audit, unioned across every root -------------------------------------
videos, gt_files, found_classes = [], [], set()
n_test = 0
for r in DATA_ROOTS:
    videos += list(r.rglob("*.mp4"))
    gt_files += list(r.rglob("ground_truth.csv"))
    if (r / "train").is_dir():
        found_classes |= {p.name for p in (r / "train").iterdir() if p.is_dir()}
    if (r / "test").is_dir():
        n_test += len(list((r / "test").rglob("*.mp4")))
missing, extra = set(CLASSES) - found_classes, found_classes - set(CLASSES)

print(f"\ndata roots: {len(DATA_ROOTS)}")
for r in DATA_ROOTS:
    n = len(list(r.rglob("*.mp4")))
    try:
        shown = r.relative_to("/kaggle/input")
    except ValueError:
        shown = r
    print(f"  {str(shown):58s} {n:5d} mp4")
print(f"\nvideos    : {len(videos)}  ({sum(p.stat().st_size for p in videos) / 1e9:.2f} GB)")
print(f"train     : {len(found_classes)}/12 class folders")
print(f"test      : {n_test} clips")
print(f"ground_truth.csv files: {len(gt_files)}")

# --- the evaluation pack, audited separately -------------------------------
# Counted per level, not in total, because a silently missing level is the
# failure that costs whole marks: L2 and L3 carry 75 of the 100 points between
# just eight videos, so eight absent files is not a rounding error.
n_eval = 0
if EVAL_ROOTS:
    print()
    print(f"eval roots: {len(EVAL_ROOTS)}")
    for r in EVAL_ROOTS:
        try:
            shown = r.relative_to("/kaggle/input")
        except ValueError:
            shown = r
        per = {lv: len(list((r / lv).rglob("*.mp4")))
               for lv in sorted(EVAL_LEVEL_DIRS) if (r / lv).is_dir()}
        n_eval += sum(per.values())
        print(f"  {str(shown):46s} " + "  ".join(f"{k}:{v}" for k, v in per.items()))
    print(f"eval videos: {n_eval}")

if MODE == "eval":
    # An absent ground_truth.csv is the EXPECTED state here, not a fault, so
    # requiring one would reject a perfectly good mount. The practice pack is
    # still wanted - for calibration only - hence a warning, not a stop.
    DATA_OK = n_eval > 0
    if not gt_files:
        print()
        print("  ! the train/test pack is NOT attached. Cell 5 calibrates")
        print("    health_thresh on known-normal TRAINING footage, so without it")
        print("    the threshold stays unset and cells 7/8/11 fall back to a")
        print("    hard-coded -0.4 that was never measured on this data.")
        print("    Add data -> prithvirajgotepatil/ahc-visual-intelligence-train-test")
else:
    DATA_OK = bool(videos) and not missing and n_test > 0 and len(gt_files) > 0

if missing and MODE != "eval":
    print(f"  ! missing class folders: {sorted(missing)}")
if extra and MODE != "eval":
    print(f"  ! unexpected folders: {sorted(extra)}")

if MODE == "eval" and DATA_OK:
    print()
    print(f"DATA OK - {n_eval} evaluation videos."
          + ("" if gt_files else "  (no calibration pack - see the warning above)"))
    print("      Cell 9 cannot score this run: the pack ships no ground truth.")
    print("      Cell 10's arena_submission.json is the output that matters.")
elif MODE == "eval":
    print("""
DATA NOT USABLE. MODE is "eval" but no L1/L2/L3 tree was found.

Attach this one:  Add data -> Your Datasets ->
                  prithvirajgotepatil/ahc-visual-intelligence-eval

It holds the 28-video private evaluation pack (E001-E028, L1 20 / L2 4 / L3 4,
1.33 GB, 47 minutes of footage) and its manifest.json. Attach the train-test
pack alongside it so cell 5 can still calibrate the threshold.
""")
elif DATA_OK:
    print(f"\nDATA OK - all twelve classes, {n_test} test clips, ground truth present.")
    if len(DATA_ROOTS) > 1:
        print(f"      (assembled from {len(DATA_ROOTS)} extracted archives - the pack was")
        print("       uploaded as multiple zips, so Kaggle extracted each into its own")
        print("       directory. Everything below reads the union, so this is fine.)")
else:
    print("""
DATA NOT USABLE. Nothing below will work until a dataset with videos is attached.

Attach this one:  Add data -> Your Datasets ->
                  prithvirajgotepatil/ahc-visual-intelligence-train-test

It holds the full 16.0 GB pack (3,207 clips, all twelve classes, the 34-video
test set and the ground truth). Note it will NOT be found by the Drive-URL
import - that stores a web page, not data, and the earlier
`flytbase-ahc-vis-int` dataset is exactly that failure.
""")
