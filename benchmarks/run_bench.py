"""Benchmarks: DP vs greedy vs graph cut. Owner: P3.

Run from the repo root:
    python benchmarks/run_bench.py                    # all three finders
    python benchmarks/run_bench.py --algos dp greedy  # leave out slow graph cut

Everything is written to --output (default out/bench/). Six experiments,
plus the checks behind the timing method:

1. Finder time and memory vs image size (brief Item 5b)
   The input image is resized to several widths. For each size we build the
   energy map once, then time each seam finder on it. This keeps the rest of
   the carve loop out of the numbers, so they show how each algorithm scales:
       DP:     fills a full H x W cost table     -> O(HW) time per seam
       greedy: scans the top row, then 3 cells   -> O(W + H) time per seam
               per row below
       graph cut: Edmonds-Karp max-flow on a     -> O(V E^2) worst case,
               graph with one node per pixel        with V and E ~ HW
   The Sobel energy map is timed too, because the carve loop rebuilds it
   for every seam whichever finder is used. Peak extra memory for one seam
   is measured separately with tracemalloc (space complexity):
       DP:     one int8 backpointer per pixel, plus a few rows -> O(HW)
       greedy: the seam itself                                 -> O(H)
       graph cut: the whole graph, a dict of edges per pixel   -> O(HW),
               with a far larger constant than DP
   -> finder_timing.csv, finder_timing.png, memory.png

2. Height vs width (brief Item 5b)
   Random energy maps where only the height grows, then only the width.
   Separating the two shows where the time really goes. dp.py works on one
   row at a time with a few NumPy calls, so it has a fixed cost per row on
   top of its per-pixel work, and that hides some of its O(HW) growth when
   only the image size is varied (experiment 1).
   -> shape_scaling.csv, shape_scaling.png
   With a limited finder (graph cut), also the same small sizes on three
   kinds of map: the image's own energy, uniform random and smooth random.
   DP and greedy do the same steps whatever the values, but graph cut needs
   more augmenting paths on some maps than others, so its time depends on
   the content as well as the size. Its searches are counted to show why.
   -> map_dependence.csv, map_dependence.png

3. Seam quality on one energy map (brief Items 3, 4 and 5c)
   Every finder gets the SAME energy map and its seam is compared with the
   DP seam. DP is optimal, so ratio = seam cost / DP seam cost is 1.00 at
   best. Uses the real image at each size, plus uniform random energy maps
   and smooth ones (blurred noise, closer to real images).
   -> seam_quality.csv, seam_quality.png

4. Full carve (brief Items 3, 4 and 5c)
   Carves the image to the target size with each finder. Records the total
   time, the time spent inside the finder, the total energy of all removed
   seams, the share of the original energy kept, and how many removed
   pixels it shares with DP. Saves the results and seam pictures for the
   visual comparison. Plain scaling and the best possible crop are run on
   the same input as baselines, as in the brief's figure.
   -> carve.csv, carve_<algo>.png, seams_<algo>.png, compare.png,
      time_breakdown.png (one panel per carved input), baselines.csv,
      baselines.png

5. Multithreading (brief Item 5d)
   OpenCV on 1 thread vs its default number of threads, for the energy map
   and for whole carves. OpenCV is the only part here that uses several
   threads: NumPy's element-wise maths runs on one thread, and DP's rows
   must be computed in order because each row needs the one above (the
   cells within a row are independent, but NumPy already does a whole row
   in compiled code, and one row is too little work to share).
   -> threading.csv, threading.png

6. Divide and conquer for the Sobel filter (brief Item 2b)
   The energy map's Sobel filter computed five ways on the same images:
   OpenCV (what energy.py uses), one NumPy pass that makes new arrays for
   each step, the same pass writing into arrays it reuses, divide and
   conquer over tiles, and FFT convolution. Each is checked against
   OpenCV's map. The two one-pass versions do the same arithmetic, so the
   gap between them is the cost of making fresh arrays. Then divide and
   conquer alone with different tile sizes at the largest width: the work
   is the same, so any change in time comes from memory or from the cost
   of each tile.
   -> sobel_methods.csv, sobel_methods.png, sobel_tiles.csv

7. Method checks (summary.md section 7)
   Run first, while the process is still cold: the cold-start cost, one call
   per sample vs batched samples, and how much tracemalloc slows code down.
   Section 7 lists every timing choice, what was tried instead, and these
   checks.

A second test image, a photo (--photo; by default matplotlib's sample photo
grace_hopper.jpg, which every matplotlib install has), goes through the
experiments whose results depend on the picture: the energy maps of
experiment 2, seam quality (3), and full carves with baselines (4). Its files
have _photo in their names. Full carves also report how much each carve looks
like a plain crop ("Like a crop"), to show when a finder just cuts a strip off.

Slow finders (graph cut): PIXEL_LIMITS caps the input size for them. When
one is selected, every finder also runs on a smaller set of inputs that fits
the cap: image widths up to the cap, a smaller height/width ladder, smaller
random maps, and a carve of a smaller copy of the image. That way all
finders are compared on the same inputs. Those files end in _small.

summary.md has every table in Markdown, ready to paste into the report,
with the settings and the command that reproduce it.

How the timing is done, and why (checked on a Windows laptop before choosing):
- One warm-up call before timing anything. The first call in a process is
  far slower (the first Sobel energy map took ~500x longer than later ones,
  the first DP seam ~1.8x).
- Fast calls are batched so each sample lasts at least MIN_SAMPLE_SECONDS,
  using Python's timeit, which also switches the garbage collector off.
  Timing single calls was 1.6-2.7x noisier between runs.
- One measurement is the median of --repeats samples, which ignores the
  odd sample slowed down by another program.
- Every timing is measured in PASSES passes that take turns over all the
  inputs of its experiment, and the median pass is reported with the lowest
  and highest in brackets. One pass was not enough: the first measurement
  of the energy map at 200 x 133 kept coming out at ~0.6 ms where later
  ones gave ~0.1-0.2 ms, with a tight spread around the wrong value.
- Full carves are too slow to batch: one untimed warm-up run, then --repeats
  timed runs that take turns between finders, so a slow patch on the
  machine affects every finder alike.
- Scaling slopes are fitted on the SLOPE_POINTS largest sizes each finder
  ran on, where the fixed cost of each call matters least. Timings whose
  passes differ by more than UNSTEADY times are left out of the fit.
"""

import argparse
import csv
import math
import os
import platform
import statistics
import sys
import time
import timeit
import tracemalloc
from pathlib import Path

import cv2
import numpy as np

# Let "python benchmarks/run_bench.py" find the seam_carving package
# (python -m benchmarks.run_bench and pytest already can).
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seam_carving.core import carve, load_image, save_image  # noqa: E402
from seam_carving.energy import compute_energy  # noqa: E402
from seam_carving.finders import FINDERS, get_finder  # noqa: E402
from seam_carving.seam_utils import seam_cost  # noqa: E402
from seam_carving.visualize import SeamRecorder, all_seams_on_original, pad_to  # noqa: E402

DEFAULT_ALGOS = ["dp", "greedy", "graphcut"]  # unfinished finders are skipped
DEFAULT_SIZES = [50, 100, 200, 400, 800]
REFERENCE = "dp"  # optimal, so every other finder is measured against it

# Graph cut builds a max-flow graph with one node per pixel, so it is far
# slower. It only runs on inputs up to this many pixels (same limit as the app).
PIXEL_LIMITS = {"graphcut": 60 * 60}
SMALL_STEPS = 5  # number of image widths in the small set for limited finders

MIN_SAMPLE_SECONDS = 0.02  # each timing sample batches calls until it lasts this long
PASSES = 3                 # every timing is measured in this many passes over its inputs
SLOPE_POINTS = 3           # slopes are fitted on this many of the largest sizes
UNSTEADY = 1.5             # flag a timing whose highest pass is this many times its lowest

# Experiment 2: one side stays at SHAPE_BASE while the other takes each step.
SHAPE_BASE = 400
SHAPE_STEPS = [50, 100, 200, 400, 800]

RANDOM_SHAPE = (100, 100)  # size of the random energy maps in experiment 3
SMOOTH_SIGMA = 3           # blur (pixels) for the smooth random maps in experiment 3

# Chart colours. Each algorithm keeps the same colour in every chart.
ALGO_COLOURS = {"dp": "#2a78d6", "greedy": "#eb6834", "graphcut": "#1baf7a", "dummy": "#eda100"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"         # titles
INK_2 = "#52514e"       # labels and tick text
MUTED = "#898781"       # tick marks, the energy reference line
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


# ---------- helpers ----------

def available_finders(names, lookup=get_finder):
    """Return ({name: finder} for the finders that work, [names skipped]).

    Unfinished finders still raise NotImplementedError, so they are skipped
    instead of stopping the whole benchmark.
    """
    finders, skipped = {}, []
    tiny = np.random.default_rng(0).random((3, 3))
    for name in names:
        finder = lookup(name)
        try:
            finder(tiny)
        except NotImplementedError:
            skipped.append(name)
            continue
        finders[name] = finder
    return finders, skipped


def too_big(name, height, width):
    """True if this finder is too slow to run on an input this size."""
    return height * width > PIXEL_LIMITS.get(name, float("inf"))


def smallest_limit(names):
    """The tightest pixel limit among these finders, or None if none has one."""
    limits = [PIXEL_LIMITS[name] for name in names if name in PIXEL_LIMITS]
    return min(limits) if limits else None


def resized_height(img, width):
    """Height of img resized to `width` with the same aspect ratio."""
    return max(2, round(img.shape[0] * width / img.shape[1]))


def resize_to_width(img, width):
    """Resize img to the given width, keeping its aspect ratio."""
    # INTER_AREA is the usual choice for shrinking, INTER_LINEAR for enlarging.
    interpolation = cv2.INTER_AREA if width < img.shape[1] else cv2.INTER_LINEAR
    return cv2.resize(img, (width, resized_height(img, width)), interpolation=interpolation)


def fit_width(img, max_pixels):
    """Largest width (at least 3) whose resized copy has at most max_pixels pixels."""
    width = max(3, math.isqrt(int(max_pixels * img.shape[1] / img.shape[0])))
    # The square root is only a first guess, because the height is rounded.
    while width > 3 and width * resized_height(img, width) > max_pixels:
        width -= 1
    while (width + 1) * resized_height(img, width + 1) <= max_pixels:
        width += 1
    return width


def small_widths(img, max_pixels):
    """Up to SMALL_STEPS widths from 10 to the largest that fits, evenly spaced on a log scale."""
    top = fit_width(img, max_pixels)
    return sorted({int(round(w)) for w in np.geomspace(min(10, top), top, SMALL_STEPS)})


def crop_to_shape(img, height, width):
    """Centre-crop img to the aspect ratio of height x width, then resize it to
    exactly that size, so a photo can stand in for another image's map."""
    h, w = img.shape[:2]
    if w * height > h * width:  # too wide: keep the middle columns
        keep = max(1, round(h * width / height))
        img = img[:, (w - keep) // 2:(w - keep) // 2 + keep]
    else:  # too tall: keep the middle rows
        keep = max(1, round(w * height / width))
        img = img[(h - keep) // 2:(h - keep) // 2 + keep]
    interpolation = cv2.INTER_AREA if width < img.shape[1] else cv2.INTER_LINEAR
    return cv2.resize(np.ascontiguousarray(img), (width, height), interpolation=interpolation)


def bundled_photo():
    """Path of matplotlib's sample photo (Grace Hopper, public domain), or None.

    matplotlib is already in requirements.txt, so every copy of the project
    has this real photo without downloading anything. It is the second test
    image, so the conclusions do not rest on one drawing.
    """
    try:
        from matplotlib import cbook
        path = Path(cbook.get_sample_data("grace_hopper.jpg", asfileobj=False))
    except Exception:  # older or trimmed matplotlib without sample data
        return None
    return path if path.exists() else None


def smooth_random(rng, shape):
    """Random energy map where neighbouring pixels are alike, as in real images.

    Uniform noise blurred with a Gaussian (SMOOTH_SIGMA pixels), then
    stretched back to [0, 1] so it has the same range as the uniform maps.
    """
    blurred = cv2.GaussianBlur(rng.random(shape), (0, 0), SMOOTH_SIGMA)
    low, high = blurred.min(), blurred.max()
    return (blurred - low) / (high - low) if high > low else np.zeros(shape)


def best_crop(energy, new_height, new_width):
    """(top, left) of the new_height x new_width window holding the most energy.

    A summed-area table gives every window's total in O(1), so the whole
    search is O(H x W).
    """
    height, width = energy.shape
    table = np.zeros((height + 1, width + 1))
    table[1:, 1:] = energy.cumsum(axis=0).cumsum(axis=1)
    rows, cols = height - new_height + 1, width - new_width + 1
    window = (table[new_height:, new_width:] - table[:rows, new_width:]
              - table[new_height:, :cols] + table[:rows, :cols])
    top, left = np.unravel_index(np.argmax(window), window.shape)
    return int(top), int(left)


def random_shapes(names):
    """Shapes of the random maps in experiment 3: the usual one, plus a small
    one that fits the tightest pixel limit if the usual one does not."""
    shapes = [RANDOM_SHAPE]
    limit = smallest_limit(names)
    if limit is not None and RANDOM_SHAPE[0] * RANDOM_SHAPE[1] > limit:
        side = math.isqrt(limit // 4)  # a quarter of the limit: graph cut is slow
        shapes.append((side, side))
    return shapes


def shape_ladders(names):
    """[(name, base, steps)] for experiment 2, plus a small ladder for limited finders."""
    ladders = [("main", SHAPE_BASE, SHAPE_STEPS)]
    limit = smallest_limit(names)
    if limit is not None and SHAPE_BASE * max(SHAPE_STEPS) > limit:
        base = math.isqrt(limit // 2)  # so base x (2 x base) still fits the limit
        steps = sorted({max(3, base // 4), max(3, base // 2), base, 2 * base})
        ladders.append(("small", base, steps))
    return ladders


def carve_inputs(img, names, new_width, new_height):
    """[(label, image, target width, target height)] for experiment 4.

    Always the image itself. If a limited finder cannot carve it, also a
    smaller copy that fits, with the same share of seams removed.
    """
    inputs = [("full", img, new_width, new_height)]
    limit = smallest_limit(names)
    height, width = img.shape[:2]
    if limit is not None and height * width > limit:
        small = resize_to_width(img, fit_width(img, limit))
        small_h, small_w = small.shape[:2]
        target_w = max(2, round(small_w * new_width / width))
        target_h = max(2, round(small_h * new_height / height))
        if (target_w, target_h) != (small_w, small_h):
            inputs.append(("small", small, target_w, target_h))
    return inputs


def median_mad(values):
    """(median, median absolute deviation) of some numbers."""
    values = list(values)
    median = statistics.median(values)
    return median, statistics.median(abs(v - median) for v in values)


def time_call(fn, repeats, min_time=None):
    """Time fn() and return (median seconds per call, spread).

    1. One warm-up call, not timed.
    2. Find how many calls make a sample last at least min_time (doubling
       1, 2, 4, ... like pyperf's calibration). A single greedy call takes
       ~0.02 ms, too short to time reliably on its own.
    3. Take `repeats` samples with timeit, which turns the garbage
       collector off while timing, and divide each by the number of calls.

    spread = median absolute deviation / median: 0.05 means the samples
    were typically within 5% of the median.
    """
    if min_time is None:
        min_time = MIN_SAMPLE_SECONDS
    timer = timeit.Timer(fn)
    fn()
    number = 1
    while timer.timeit(number) < min_time:
        number *= 2
    median, mad = median_mad(t / number for t in timer.repeat(repeats, number))
    return median, (mad / median if median > 0 else 0.0)


def peak_memory(fn):
    """Extra memory in bytes that fn() needs at its peak, measured with tracemalloc.

    NumPy reports its arrays to tracemalloc, so DP's tables are included.
    Kept apart from the timing because tracing slows the code down.
    """
    started_here = not tracemalloc.is_tracing()
    if started_here:
        tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        fn()
        return tracemalloc.get_traced_memory()[1] - before
    finally:
        if started_here:
            tracemalloc.stop()


def measure_passes(jobs, repeats):
    """Time every job in PASSES passes. Returns {key: (median, lowest, highest)} in ms.

    jobs: list of (key, fn). Each pass times every job once with time_call,
    so a slow stretch of the run, or a cold first measurement at one size,
    lands in one pass of each job instead of in all of one job's numbers.
    """
    times = {key: [] for key, _fn in jobs}
    for _ in range(PASSES):
        for key, fn in jobs:
            times[key].append(1000 * time_call(fn, repeats)[0])
    return {key: (statistics.median(t), min(t), max(t)) for key, t in times.items()}


def _store(row, prefix, measured):
    """Put (median, lowest, highest) in row as <prefix>_ms, _min_ms, _max_ms (None if skipped)."""
    median, low, high = measured if measured else (None, None, None)
    row[f"{prefix}_ms"] = median
    row[f"{prefix}_min_ms"] = low
    row[f"{prefix}_max_ms"] = high


def compare_to_reference(energy, seam, ref_seam):
    """Compare a seam with the DP seam found on the same energy map."""
    cost = seam_cost(energy, seam)
    ref_cost = seam_cost(energy, ref_seam)
    if ref_cost > 0:
        ratio = cost / ref_cost
    else:
        # A zero-energy DP seam (a perfectly flat strip) makes the ratio
        # undefined: count it as 1 if the other finder also found zero.
        ratio = 1.0 if cost == 0 else float("inf")
    return {
        "cost": cost,
        "dp_cost": ref_cost,
        "ratio": ratio,
        "optimal": bool(np.isclose(cost, ref_cost)),
        # Share of rows where both seams remove the same column.
        "rows_same": float(np.mean(np.asarray(seam) == np.asarray(ref_seam))),
    }


def removed_mask(shape, removed):
    """H x W bool mask of every removed pixel, in original-image coordinates.

    removed: SeamRecorder.removed, a list of (direction, orig_rows, orig_cols)
    """
    mask = np.zeros(shape, dtype=bool)
    for _direction, rows, cols in removed:
        mask[rows, cols] = True
    return mask


def overlap(a, b):
    """Pixels removed by both / pixels removed by either (1 = identical)."""
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def crop_likeness(mask, new_height, new_width):
    """How much a carve looks like a crop: its overlap with the closest crop.

    mask: the pixels the carve removed. A crop to new_height x new_width
    removes as many pixels as the carve, so the overlap is highest for the
    crop window holding the fewest removed pixels, which best_crop finds on
    the kept pixels. 1 means the carve removed exactly a crop's pixels.
    """
    top, left = best_crop((~mask).astype(np.float64), new_height, new_width)
    crop_removed = np.ones_like(mask, dtype=bool)
    crop_removed[top:top + new_height, left:left + new_width] = False
    return overlap(mask, crop_removed)


def loglog_slope(sizes, times, points=None):
    """Slope of log(time) against log(size), fitted on the `points` largest sizes.

    1 means time is proportional to the size, 0.5 to its square root,
    0 means the size makes no difference. Small sizes are left out by
    default because there the fixed cost of each call hides the growth.
    """
    if points is None:
        points = SLOPE_POINTS
    data = sorted((s, t) for s, t in zip(sizes, times) if t is not None and t > 0)
    data = data[-points:]
    if len(data) < 2:
        return None
    x, y = zip(*data)
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


# ---------- experiment 1: finder time and memory vs image size ----------

def finder_timing(img, finders, widths, repeats):
    """One row per image width: energy map and finder times (ms) and finder peak memory (KB)."""
    inputs, jobs = {}, []
    for width in widths:
        resized = resize_to_width(img, width)
        energy = compute_energy(resized)
        inputs[width] = (resized, energy)
        jobs.append((("energy", width), lambda r=resized: compute_energy(r)))
        for name, finder in finders.items():
            if not too_big(name, *energy.shape):
                jobs.append(((name, width), lambda f=finder, e=energy: f(e)))
    measured = measure_passes(jobs, repeats)

    rows = []
    for width in widths:
        resized, energy = inputs[width]
        height = resized.shape[0]
        row = {"width": width, "height": height, "pixels": width * height}
        _store(row, "energy", measured[("energy", width)])
        for name in finders:
            _store(row, name, measured.get((name, width)))
        row["energy_peak_kb"] = peak_memory(lambda: compute_energy(resized)) / 1024
        for name, finder in finders.items():
            fits = (name, width) in measured
            row[f"{name}_peak_kb"] = peak_memory(lambda f=finder: f(energy)) / 1024 if fits else None
        rows.append(row)
    return rows


# ---------- experiment 2: height vs width ----------

def shape_scaling(finders, repeats, seed=0):
    """Time each finder on random maps where only the height, then only the width, grows."""
    rng = np.random.default_rng(seed)
    shapes, jobs = [], []
    for ladder, base, steps in shape_ladders(finders):
        for vary in ("height", "width"):
            for step in steps:
                height, width = (step, base) if vary == "height" else (base, step)
                energy = rng.random((height, width))
                shape = (ladder, vary, height, width)
                shapes.append(shape)
                for name, finder in finders.items():
                    if not too_big(name, height, width):
                        jobs.append(((name, shape), lambda f=finder, e=energy: f(e)))
    measured = measure_passes(jobs, repeats)

    rows = []
    for shape in shapes:
        ladder, vary, height, width = shape
        row = {"ladder": ladder, "vary": vary, "height": height, "width": width}
        for name in finders:
            _store(row, name, measured.get((name, shape)))
        rows.append(row)
    return rows


MAP_KINDS = ("image", "random", "smooth")  # the image's own map, uniform and smooth random
ALL_MAP_KINDS = ("image", "photo", "random", "smooth")  # plus the photo's map, if there is one
NEAR_ZERO = 0.01  # "near zero" = below this share of the map's largest energy


def bfs_runs(energy):
    """Breadth-first searches graph cut makes to find one seam on this map.

    About one per augmenting path, so it counts the work that makes graph
    cut's time depend on the values in the map. Wraps graphcut._bfs for one
    call and puts it back; returns None if graphcut.py has no _bfs to count.
    """
    from seam_carving import graphcut
    original = getattr(graphcut, "_bfs", None)
    if original is None:
        return None
    count = 0

    def counting(*args, **kwargs):
        nonlocal count
        count += 1
        return original(*args, **kwargs)

    graphcut._bfs = counting
    try:
        graphcut.find_vertical_seam(energy)
    finally:
        graphcut._bfs = original
    return count


def edge_share(seam, width):
    """Share of rows where the seam is in the first or last column. There
    greedy has only two neighbours to check instead of three."""
    seam = np.asarray(seam)
    return float(np.mean((seam == 0) | (seam == width - 1)))


def map_dependence(img, finders, widths, repeats, seed=0, photo=None):
    """Same sizes, different energy maps: does the content change the time?

    At each width, the image's own energy map (as in experiment 1), the
    photo's map if given (centre-cropped to the same shape), one uniform
    random map and one smooth random map (as in experiment 3), all the same
    shape. DP and greedy do the same steps whatever the values;
    graph cut's max-flow needs a number of augmenting paths that depends on
    them. Rows: width, height, pixels, map, near_zero, <name>_ms (+ range),
    graphcut_searches, greedy_edge (share of rows where greedy's seam runs
    along the first or last column, where it checks one cell fewer).
    """
    rng = np.random.default_rng(seed)
    maps = []
    for width in widths:
        energy = compute_energy(resize_to_width(img, width))
        shape = energy.shape
        maps.append((width, "image", energy))
        if photo is not None:  # takes nothing from rng, so the random maps stay the same
            maps.append((width, "photo", compute_energy(crop_to_shape(photo, *shape))))
        maps += [(width, "random", rng.random(shape)), (width, "smooth", smooth_random(rng, shape))]
    jobs = [((name, width, kind), lambda f=finder, e=energy: f(e))
            for width, kind, energy in maps
            for name, finder in finders.items() if not too_big(name, *energy.shape)]
    measured = measure_passes(jobs, repeats)

    rows = []
    for width, kind, energy in maps:
        height = energy.shape[0]
        row = {"width": width, "height": height, "pixels": width * height, "map": kind,
               "near_zero": float(np.mean(energy < NEAR_ZERO * energy.max())) if energy.max() > 0 else 1.0}
        for name in finders:
            _store(row, name, measured.get((name, width, kind)))
        counted = "graphcut" in finders and not too_big("graphcut", height, width)
        row["graphcut_searches"] = bfs_runs(energy) if counted else None
        greedy = finders.get("greedy")
        row["greedy_edge"] = edge_share(greedy(energy), width) if greedy else None
        rows.append(row)
    return rows


# ---------- experiment 3: seam quality on one energy map ----------

def seam_quality(img, finders, widths, trials, seed=0, small=(), photo=None, photo_small=()):
    """Compare every finder's seam with the DP seam on the same energy maps.

    Energy maps: the real image at each width (and at each `small` width),
    the photo too if given (never enlarged, so only widths up to its own),
    then `trials` uniform random maps and `trials` smooth random maps of each
    shape from random_shapes().
    Returns one row per (energy map, finder).
    """
    dp = get_finder(REFERENCE)

    pictures = [("image", img, widths), ("image (small)", img, small)]
    if photo is not None:
        pictures += [("photo", photo, [w for w in widths if w <= photo.shape[1]]),
                     ("photo (small)", photo, photo_small)]
    cases = []
    for kind, picture, kind_widths in pictures:
        for width in kind_widths:
            resized = resize_to_width(picture, width)
            cases.append((kind, f"{resized.shape[1]}x{resized.shape[0]}", compute_energy(resized)))
    rng = np.random.default_rng(seed)
    shapes = random_shapes(finders)
    for shape in shapes:
        for trial in range(trials):
            cases.append((f"random {shape[1]}x{shape[0]}", f"#{trial}", rng.random(shape)))
    # Smooth maps come after the uniform ones, so the uniform maps for a
    # given seed stay the same as in earlier runs.
    for shape in shapes:
        for trial in range(trials):
            cases.append((f"smooth {shape[1]}x{shape[0]}", f"#{trial}", smooth_random(rng, shape)))

    rows = []
    for kind, source, energy in cases:
        ref_seam = dp(energy)
        for name, finder in finders.items():
            if too_big(name, *energy.shape):
                continue
            seam = np.asarray(finder(energy))
            rows.append({"kind": kind, "source": source, "algo": name,
                         **compare_to_reference(energy, seam, ref_seam)})
    return rows


# ---------- experiment 4: full carve ----------

def carve_run(img, finder, new_width, new_height):
    """Carve once. Returns (result, total s, s inside the finder, energy removed)."""
    finder_seconds = 0.0
    costs = []

    def timed_finder(energy):
        # Wrap the finder so we can time it and record each seam's energy
        # without changing any finder code. Measured overhead: within noise.
        nonlocal finder_seconds
        start = time.perf_counter()
        seam = finder(energy)
        finder_seconds += time.perf_counter() - start
        costs.append(seam_cost(energy, np.asarray(seam)))
        return seam

    start = time.perf_counter()
    result = carve(img, new_width, new_height, timed_finder)
    total_seconds = time.perf_counter() - start
    return result, total_seconds, finder_seconds, float(sum(costs))


def carve_comparison(img, finders, new_width, new_height, repeats):
    """Carve img with each finder that fits. Returns (rows, pictures).

    pictures[name] = {"result": carved image, "seams": seams on the original}
    """
    names = [name for name in finders if not too_big(name, *img.shape[:2])]
    rows, pictures, masks = [], {}, {}

    # Untimed run first, with the recorder for the seam pictures. It is
    # also the warm-up, so cold-start time never counts as carving time.
    for name in names:
        recorder = SeamRecorder(keep_frames=False)
        result = carve(img, new_width, new_height, finders[name], on_seam=recorder)
        masks[name] = removed_mask(img.shape[:2], recorder.removed)
        pictures[name] = {"result": result,
                          "seams": all_seams_on_original(img, recorder.removed)}

    # Timed runs take turns between finders (dp, greedy, dp, greedy, ...), so
    # a slow patch on the machine (another program, CPU clock changes) hits
    # every finder alike instead of only the one running at that moment.
    runs = {name: [] for name in names}
    for _ in range(repeats):
        for name in names:
            runs[name].append(carve_run(img, finders[name], new_width, new_height))

    seams = (img.shape[1] - new_width) + (img.shape[0] - new_height)
    # Measured on the ORIGINAL energy map, which no finder optimises directly
    # (each seam is chosen on the current, already carved image).
    original_energy = compute_energy(img)
    total_energy = original_energy.sum()
    for name in names:
        totals = [run[1] for run in runs[name]]
        inside = [run[2] for run in runs[name]]
        rows.append({
            "algo": name,
            "seams": seams,
            "total_s": statistics.median(totals),
            "total_min_s": min(totals),
            "total_max_s": max(totals),
            "finder_s": statistics.median(inside),
            "finder_min_s": min(inside),
            "finder_max_s": max(inside),
            # Share per run, then the median, so both parts come from one run.
            "finder_share": statistics.median(f / t if t > 0 else 0.0
                                              for f, t in zip(inside, totals)),
            # Energy map + seam removal for one seam, inside the real carve loop.
            "outside_per_seam_ms": 1000 * statistics.median(t - f for f, t in zip(inside, totals))
                                   / seams if seams else None,
            "energy_removed": runs[name][0][3],  # the same in every run
            # Share of the original image's energy whose pixels survive.
            "energy_kept": (1 - original_energy[masks[name]].sum() / total_energy
                            if total_energy > 0 else None),
            "overlap_with_dp": None,
            # How much the carve looks like a crop: 1 = it removed exactly the
            # pixels some crop of the same size cuts off. "Is it just cropping?"
            "crop_likeness": crop_likeness(masks[name], new_height, new_width),
        })

    if REFERENCE in masks:
        for row in rows:
            row["overlap_with_dp"] = overlap(masks[row["algo"]], masks[REFERENCE])
    return rows, pictures


def baselines(img, new_width, new_height, repeats):
    """Plain scaling and cropping, the two simple ways to resize (the brief's figure).

    The crop is the window that keeps the most original energy: the best
    possible crop, so seam carving is compared with a strong baseline.
    Returns (rows, pictures).
    """
    energy = compute_energy(img)
    total = energy.sum()

    def scale():
        return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    def crop():
        # Timed with the energy map, because the crop needs it to choose.
        # .copy(): a slice is only a view, so without it nothing would be timed.
        top, left = best_crop(compute_energy(img), new_height, new_width)
        return img[top:top + new_height, left:left + new_width].copy()

    # The obvious alternative to the best crop: cut the same amount off every side.
    centre_top = (img.shape[0] - new_height) // 2
    centre_left = (img.shape[1] - new_width) // 2

    def centre_crop():
        return img[centre_top:centre_top + new_height, centre_left:centre_left + new_width].copy()

    measured = measure_passes([("scaling", scale), ("cropping", crop),
                               ("centre crop", centre_crop)], repeats)
    top, left = best_crop(energy, new_height, new_width)

    def kept(t, l):
        return energy[t:t + new_height, l:l + new_width].sum() / total if total > 0 else None

    rows = []
    # Scaling blends pixels instead of keeping some, so "energy kept" does not apply.
    for method, share in (("scaling", None), ("cropping", kept(top, left)),
                          ("centre crop", kept(centre_top, centre_left))):
        row = {"method": method}
        _store(row, "time", measured[method])
        row["energy_kept"] = share
        rows.append(row)
    return rows, {"scaling": scale(), "cropping": crop()}


# ---------- experiment 5: multithreading ----------

def threading_check(img, finders, widths, new_width, new_height, repeats):
    """Energy map and full carve times with OpenCV on 1 thread vs its default.

    The energy map is measured in PASSES passes that take turns between the
    two settings; full carves in --repeats runs that do the same. Rows give
    the median and the lowest and highest pass or run.
    Restores OpenCV's thread count afterwards, even if something fails.
    """
    default = cv2.getNumThreads()
    settings = (1, default)
    resized = {width: resize_to_width(img, width) for width in widths}
    energy_times = {(width, threads): [] for width in widths for threads in settings}
    names = [name for name in finders if not too_big(name, *img.shape[:2])]
    carve_times = {(name, threads): [] for name in names for threads in settings}
    try:
        for _ in range(PASSES):
            for threads in settings:
                cv2.setNumThreads(threads)
                for width in widths:
                    seconds, _spread = time_call(lambda: compute_energy(resized[width]), repeats)
                    energy_times[(width, threads)].append(seconds)
        for _ in range(repeats):  # take turns, as in experiment 4
            for threads in settings:
                cv2.setNumThreads(threads)
                for name in names:
                    carve_times[(name, threads)].append(
                        carve_run(img, finders[name], new_width, new_height)[1])
    finally:
        cv2.setNumThreads(default)

    rows = [_threading_row(f"energy map {width} x {resized[width].shape[0]}",
                           energy_times[(width, 1)], energy_times[(width, default)])
            for width in widths]
    rows += [_threading_row(f"full carve, {name}",
                            carve_times[(name, 1)], carve_times[(name, default)])
             for name in names]
    return rows


def _threading_row(part, one, many):
    """one, many: times in seconds on 1 thread and on the default threads."""
    return {
        "part": part,
        "one_thread_ms": 1000 * statistics.median(one),
        "one_thread_min_ms": 1000 * min(one),
        "one_thread_max_ms": 1000 * max(one),
        "default_threads_ms": 1000 * statistics.median(many),
        "default_threads_min_ms": 1000 * min(many),
        "default_threads_max_ms": 1000 * max(many),
        "speedup": statistics.median(one) / statistics.median(many),
    }


# ---------- experiment 6: Item 2b, divide and conquer for the Sobel filter ----------

SOBEL_LEAF = 64  # divide and conquer stops splitting at tiles this size or smaller
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = SOBEL_X.T


def gray_of(img):
    """The float64 greyscale image energy.py filters."""
    return cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2GRAY).astype(np.float64)


def sobel_opencv(gray):
    """|gx| + |gy| with OpenCV, exactly as energy.py computes it."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.abs(gx) + np.abs(gy)


def _sobel_inside(padded):
    """|gx| + |gy| for the inside of `padded`, which has a one-pixel border:
    the 3 x 3 Sobel filter written out as sums of shifted slices."""
    p = padded
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
    return np.abs(gx) + np.abs(gy)


def sobel_direct(gray):
    """The whole image in one pass. np.pad's "reflect" matches OpenCV's default
    border (BORDER_REFLECT_101), so the result is the same map."""
    return _sobel_inside(np.pad(gray, 1, mode="reflect"))


_SOBEL_BUFFERS = {}  # (height, width) -> arrays that sobel_reused makes once and reuses


def sobel_reused(gray):
    """The same one pass as sobel_direct, step for step, but every step writes
    into arrays made once per image size and reused on later calls (NumPy's
    out=), so a call makes no new arrays. The arithmetic and its order are
    the same, so the result equals sobel_direct's exactly, and any gap in
    time between the two is the cost of making fresh arrays.

    Returns one of its reused arrays: copy it to keep it past the next call.
    Images under 2 x 2 have no reflected border, so they go to sobel_direct.
    """
    height, width = gray.shape
    if height < 2 or width < 2:
        return sobel_direct(gray)
    buffers = _SOBEL_BUFFERS.get((height, width))
    if buffers is None:
        buffers = _SOBEL_BUFFERS[(height, width)] = (
            np.empty((height + 2, width + 2)),) + tuple(np.empty((height, width)) for _ in range(4))
    p, gx, gy, part, twice = buffers
    # The border np.pad(mode="reflect") would add: row 1 above row 0, and so on.
    p[1:-1, 1:-1] = gray
    p[0, 1:-1], p[-1, 1:-1] = gray[1], gray[-2]
    p[:, 0], p[:, -1] = p[:, 2], p[:, -3]

    def weighted_sum(first, middle, last, out):
        """out = first + 2 * middle + last, in the same order as _sobel_inside."""
        np.multiply(middle, 2, out=twice)
        np.add(first, twice, out=out)
        out += last

    weighted_sum(p[:-2, 2:], p[1:-1, 2:], p[2:, 2:], gx)
    weighted_sum(p[:-2, :-2], p[1:-1, :-2], p[2:, :-2], part)
    gx -= part
    weighted_sum(p[2:, :-2], p[2:, 1:-1], p[2:, 2:], gy)
    weighted_sum(p[:-2, :-2], p[:-2, 1:-1], p[:-2, 2:], part)
    gy -= part
    np.abs(gx, out=gx)
    np.abs(gy, out=gy)
    gx += gy
    return gx


def sobel_divide_conquer(gray, leaf=SOBEL_LEAF):
    """Divide and conquer: split the image in half along its longer side until
    each tile is at most leaf x leaf, filter each tile with a one-pixel border
    taken from its neighbours, and write it into place (the combine step).

    The tiles never overlap in the output, so the work stays O(H x W); the
    recursion adds O(H x W / leaf^2) calls on top.
    """
    padded = np.pad(gray, 1, mode="reflect")
    out = np.empty_like(gray)

    def solve(top, left, height, width):
        if height <= leaf and width <= leaf:
            tile = padded[top:top + height + 2, left:left + width + 2]
            out[top:top + height, left:left + width] = _sobel_inside(tile)
        elif height >= width:
            half = height // 2
            solve(top, left, half, width)
            solve(top + half, left, height - half, width)
        else:
            half = width // 2
            solve(top, left, height, half)
            solve(top, left + half, height, width - half)

    solve(0, 0, *gray.shape)
    return out


def sobel_fft(gray):
    """Convolution by FFT: O(H x W log(H x W)), the usual fast method for big kernels."""
    padded = np.pad(gray, 1, mode="reflect")
    height, width = padded.shape
    size = (height + 2, width + 2)  # full convolution with a 3 x 3 kernel
    spectrum = np.fft.rfft2(padded, size)
    out = 0.0
    for kernel in (SOBEL_X, SOBEL_Y):
        full = np.fft.irfft2(spectrum * np.fft.rfft2(kernel, size), size)
        # Convolution flips the kernel, which only flips the sign; abs() undoes it.
        out = out + np.abs(full[2:height, 2:width])
    return out


SOBEL_METHODS = {"opencv": sobel_opencv, "direct": sobel_direct, "reused": sobel_reused,
                 "divide": sobel_divide_conquer, "fft": sobel_fft}
SOBEL_LABELS = {"opencv": "OpenCV (cv2.Sobel)", "direct": "one pass, new arrays",
                "reused": "one pass, reused arrays", "divide": "divide and conquer",
                "fft": "FFT convolution"}


def sobel_methods(img, widths, repeats):
    """Time the five ways of computing the energy map's Sobel filter on the same
    greyscale images, and check each gives OpenCV's map. Rows: width, height,
    pixels, <method>_ms (+ range), <method>_diff (largest difference)."""
    grays = {width: gray_of(resize_to_width(img, width)) for width in widths}
    jobs = [((method, width), lambda f=fn, g=gray: f(g))
            for width, gray in grays.items() for method, fn in SOBEL_METHODS.items()]
    measured = measure_passes(jobs, repeats)
    rows = []
    for width, gray in grays.items():
        reference = sobel_opencv(gray)
        row = {"width": width, "height": gray.shape[0], "pixels": gray.size}
        for method, fn in SOBEL_METHODS.items():
            _store(row, method, measured[(method, width)])
            row[f"{method}_diff"] = float(np.max(np.abs(fn(gray) - reference)))
        rows.append(row)
    return rows


SOBEL_TILES = (16, 32, 64, 128, 256)  # tile sizes for the tile-size check, plus one whole-image tile


def sobel_tile_sizes(img, width, repeats):
    """Divide and conquer at one image size with different largest tiles, up
    to one tile for the whole image. Every tile size does the same work, so
    a time that changes with tile size points to memory (the size of the
    arrays each tile makes) or to the cost of each tile, not to less work.
    Rows: leaf, tiles, label, divide_ms (+ range)."""
    gray = gray_of(resize_to_width(img, width))
    whole = max(gray.shape)
    leaves = [leaf for leaf in SOBEL_TILES if leaf < whole] + [whole]
    jobs = [(leaf, lambda g=gray, n=leaf: sobel_divide_conquer(g, leaf=n)) for leaf in leaves]
    measured = measure_passes(jobs, repeats)
    rows = []
    for leaf in leaves:
        row = {"leaf": leaf, "label": "whole image" if leaf == whole else f"at most {leaf} x {leaf}",
               "tiles": tile_count(*gray.shape, leaf), "width": gray.shape[1], "height": gray.shape[0]}
        _store(row, "divide", measured[leaf])
        rows.append(row)
    return rows


def tile_count(height, width, leaf):
    """How many tiles sobel_divide_conquer filters, splitting exactly as it does."""
    if height <= leaf and width <= leaf:
        return 1
    if height >= width:
        return tile_count(height // 2, width, leaf) + tile_count(height - height // 2, width, leaf)
    return tile_count(height, width // 2, leaf) + tile_count(height, width - width // 2, leaf)


def _tile_effect(tile_rows):
    """(best small-tile row, whole-image row, ratio) from the tile-size check,
    or None. ratio = whole-image time / best small-tile time."""
    if not tile_rows:
        return None
    whole = tile_rows[-1]
    best = min(tile_rows[:-1], key=lambda r: r["divide_ms"], default=None)
    if best is None or not best["divide_ms"]:
        return None
    return best, whole, whole["divide_ms"] / best["divide_ms"]


# ---------- method checks: why the timing is done this way (section 7) ----------

def prefer_fast_cores():
    """Ask Windows not to run this process in its power-saving mode (EcoQoS).

    On a CPU with fast and efficient cores, Windows can move a background
    process onto the slower efficient cores while it runs, so one pass of a
    timing can come out much slower than another for no reason in the code.
    Turning off execution-speed throttling for the process (the same switch
    as "powercfg /powerthrottling disable") asks the scheduler to treat it as
    full-speed work. Returns True if Windows accepted, False elsewhere.
    """
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    class PowerThrottlingState(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG),
                    ("StateMask", wintypes.ULONG)]

    # Version 1; control execution speed (0x1); state 0 = never throttle it.
    state = PowerThrottlingState(1, 0x1, 0)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                               wintypes.DWORD]
    kernel32.SetProcessInformation.restype = wintypes.BOOL
    process_power_throttling = 4  # PROCESS_INFORMATION_CLASS value
    return bool(kernel32.SetProcessInformation(kernel32.GetCurrentProcess(), process_power_throttling,
                                               ctypes.byref(state), ctypes.sizeof(state)))


def method_checks(img, finders, repeats):
    """Measure, in this run, the reasons behind the timing method.

    Call it first, while the process is still cold. Returns a dict with:
      cold_energy_ms, warm_energy_ms: the first energy map vs later ones
      cold_finder_ms, warm_finder_ms, finder: the same for DP (or the first finder)
      single_us, batched_us, single_drift, batched_drift, fast_finder, tiny_size:
        five measurements of the fastest finder on a tiny map, with one call
        per sample vs batched: their median and how far they drift apart
      resolution_us: the resolution of time.perf_counter on this machine
      traced_ratio: DP's time with tracemalloc running / without (None if
        tracemalloc was already running)
      size: the width x height used
    """
    small = resize_to_width(img, min(200, img.shape[1]))
    checks = {"size": f"{small.shape[1]} x {small.shape[0]}"}

    def first_then_warm(fn):
        start = time.perf_counter()
        fn()
        first = 1000 * (time.perf_counter() - start)
        warm = []
        for _ in range(5):
            start = time.perf_counter()
            fn()
            warm.append(1000 * (time.perf_counter() - start))
        return first, statistics.median(warm)

    checks["cold_energy_ms"], checks["warm_energy_ms"] = first_then_warm(lambda: compute_energy(small))
    energy = compute_energy(small)
    name = REFERENCE if REFERENCE in finders else next(iter(finders))
    checks["finder"] = name
    finder = finders[name]
    checks["cold_finder_ms"], checks["warm_finder_ms"] = first_then_warm(lambda: finder(energy))

    # Batching matters most for the shortest calls, so check it on a tiny map.
    fast = "greedy" if "greedy" in finders else name
    checks["fast_finder"] = fast
    tiny = compute_energy(resize_to_width(img, min(20, img.shape[1])))
    checks["tiny_size"] = f"{tiny.shape[1]} x {tiny.shape[0]}"
    single, batched = [], []
    for _ in range(5):  # take turns, so both see the same machine
        single.append(time_call(lambda: finders[fast](tiny), repeats, min_time=0)[0])
        batched.append(time_call(lambda: finders[fast](tiny), repeats)[0])
    checks["single_us"] = 1e6 * statistics.median(single)
    checks["batched_us"] = 1e6 * statistics.median(batched)
    # How far five measurements of the same call drift apart: (highest - lowest) / median.
    checks["single_drift"] = (max(single) - min(single)) / statistics.median(single)
    checks["batched_drift"] = (max(batched) - min(batched)) / statistics.median(batched)
    checks["resolution_us"] = 1e6 * time.get_clock_info("perf_counter").resolution

    checks["traced_ratio"] = None
    if not tracemalloc.is_tracing():
        plain = time_call(lambda: finder(energy), repeats)[0]
        tracemalloc.start()
        try:
            traced = time_call(lambda: finder(energy), repeats)[0]
        finally:
            tracemalloc.stop()
        checks["traced_ratio"] = traced / plain if plain > 0 else None
    return checks


def pass_spread(rows, prefixes, label):
    """(median, largest, where) of highest / lowest pass over every timing."""
    found = []
    for row in rows:
        for prefix in prefixes:
            low, high = row.get(f"{prefix}_min_ms"), row.get(f"{prefix}_max_ms")
            if low and high:
                found.append((high / low, f"{prefix} at {label(row)}"))
    if not found:
        return None
    worst = max(found)
    return statistics.median(r for r, _ in found), worst[0], worst[1]


# ---------- output: CSV, charts, Markdown ----------

def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)  # None is written as an empty cell


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")  # draw straight to a file, no window
    import matplotlib.pyplot as plt
    return plt


def _style_axes(fig, ax):
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(which="both", colors=MUTED, labelcolor=INK_2, labelsize=9)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _label_line_ends(ax, ends, gap_points=11):
    """Name each line at its right end, nudging a label up when it would sit
    within gap_points of the one below it. ends: [(x, y, text)].
    Call after the axes' scales and margins are set."""
    ax.autoscale_view()
    to_points = 72 / ax.figure.dpi
    placed = None
    for x_end, y_end, text in sorted(ends, key=lambda end: end[1]):
        height = ax.transData.transform((x_end, y_end))[1] * to_points
        shift = 0.0
        if placed is not None and height < placed + gap_points:
            shift = placed + gap_points - height
        placed = height + shift
        ax.annotate(text, (x_end, y_end), xytext=(8, shift), textcoords="offset points",
                    va="center", color=INK_2, fontsize=9,
                    bbox={"boxstyle": "square,pad=0.15", "fc": SURFACE, "ec": "none"})


def _plot_finder_lines(ax, rows, x_key, names):
    """One line per finder (fixed colour), named at its right end."""
    for name in names:
        points = [(r[x_key], r[f"{name}_ms"]) for r in rows if r[f"{name}_ms"] is not None]
        if not points:
            continue
        x, y = zip(*points)
        ax.plot(x, y, color=ALGO_COLOURS.get(name, INK_2), linewidth=2, marker="o",
                markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
        ax.annotate(name, (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", color=INK_2, fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.margins(x=0.12)


def plot_finder_timing(rows, names, path):
    """Log-log line chart: time to find one seam vs pixels, one line per finder."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    _style_axes(fig, ax)

    pixels = [r["pixels"] for r in rows]
    # Reference line: the energy map every finder pays for, once per seam.
    ax.plot(pixels, [r["energy_ms"] for r in rows], color=MUTED, linewidth=1.5,
            linestyle="--", label="Sobel energy map (needed before every seam)")
    ax.annotate("energy map", (pixels[-1], rows[-1]["energy_ms"]), xytext=(8, 0),
                textcoords="offset points", va="center", color=INK_2, fontsize=9)
    _plot_finder_lines(ax, rows, "pixels", names)

    ax.set_xlabel("Pixels in the energy map (H x W, log scale)", color=INK_2)
    ax.set_ylabel("Time for one seam (ms, log scale)", color=INK_2)
    ax.set_title("Seam finder time vs image size", loc="left", color=INK, fontsize=11)
    # Below the plot: a slow finder's line (graph cut) can end in any corner.
    ax.legend(frameon=False, labelcolor=INK_2, fontsize=9, ncol=2,
              loc="upper left", bbox_to_anchor=(0, -0.16))
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_memory(rows, names, path):
    """Two log-log panels: peak extra memory per seam, and the same per pixel.

    In the per-pixel panel a flat line means memory grows like H x W, and a
    falling line means it grows more slowly (greedy's O(H)).
    """
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9), dpi=150)
    series = ([(n, f"{n}_peak_kb", ALGO_COLOURS.get(n, INK_2), True) for n in names]
              + [("energy map", "energy_peak_kb", MUTED, False)])
    for ax, per_pixel in zip(axes, (False, True)):
        _style_axes(fig, ax)
        for label, key, colour, is_finder in series:
            points = [(r["pixels"], 1024 * r[key] / r["pixels"] if per_pixel else r[key])
                      for r in rows if r[key] is not None]
            if not points:
                continue
            x, y = zip(*points)
            if is_finder:
                ax.plot(x, y, color=colour, linewidth=2, marker="o", markersize=6,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
            else:  # reference line, like the energy map in finder_timing.png
                ax.plot(x, y, color=colour, linewidth=1.5, linestyle="--", label=label)
            ax.annotate(label, (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points",
                        va="center", color=INK_2, fontsize=9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.margins(x=0.2)
        ax.set_xlabel("Pixels (H x W, log scale)", color=INK_2)
    axes[1].axhline(1.0, color=AXIS, linewidth=1, linestyle=":")
    axes[1].annotate("1 byte (DP's int8 table)", (axes[1].get_xlim()[0], 1.0), xytext=(4, -10),
                     textcoords="offset points", color=INK_2, fontsize=8)
    axes[0].set_title("Peak extra memory for one seam", loc="left", color=INK, fontsize=10)
    axes[0].set_ylabel("KB (log scale)", color=INK_2)
    axes[1].set_title("Per pixel: flat means O(H x W)", loc="left", color=INK, fontsize=10)
    axes[1].set_ylabel("Bytes per pixel (log scale)", color=INK_2)
    axes[0].legend(frameon=False, labelcolor=INK_2, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_threading(rows, threads, path):
    """Speed-up from OpenCV's default threads over 1 thread, one row per part.

    Dot = median speed-up. Line = from the least to the most favourable
    combination of passes (or runs), so a line crossing 1 means the
    measurements cannot tell the two settings apart.
    """
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7, 1.5 + 0.42 * len(rows)), dpi=150)
    _style_axes(fig, ax)
    ax.yaxis.grid(False)
    for y, r in enumerate(rows):
        low = r["one_thread_min_ms"] / r["default_threads_max_ms"]
        high = r["one_thread_max_ms"] / r["default_threads_min_ms"]
        ax.plot([low, high], [y, y], color=AXIS, linewidth=3, solid_capstyle="round")
        ax.plot([r["speedup"]], [y], "o", color=INK_2, markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.annotate(f"{r['speedup']:.2f}x", (high, y), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK_2, fontsize=9)
    ax.axvline(1.0, color=MUTED, linestyle="--", linewidth=1.2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["part"] for r in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.margins(x=0.15)
    ax.set_xlabel(f"Speed-up with {threads} OpenCV threads instead of 1 (1 = no change)",
                  color=INK_2)
    ax.set_title("Does multithreading help?\nDot = median; line = least to most favourable "
                 "pass or run", loc="left", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_shape_scaling(rows, names, base, path):
    """Two panels: only the height grows, then only the width grows."""
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9), dpi=150, sharey=True)
    for ax, vary in zip(axes, ("height", "width")):
        _style_axes(fig, ax)
        fixed = "width" if vary == "height" else "height"
        _plot_finder_lines(ax, [r for r in rows if r["vary"] == vary], vary, names)
        ax.set_title(f"Only the {vary} grows ({fixed} = {base})",
                     loc="left", color=INK, fontsize=10)
        ax.set_xlabel(f"{vary.capitalize()} in pixels (log scale)", color=INK_2)
    axes[0].set_ylabel("Time for one seam (ms, log scale)", color=INK_2)
    axes[0].legend(frameon=False, labelcolor=INK_2, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def map_ratios(rows, name, kind):
    """[(pixels, time on `kind` map / time on the image's map)] for one finder."""
    image = {r["width"]: r[f"{name}_ms"] for r in rows if r["map"] == "image"}
    return [(r["pixels"], r[f"{name}_ms"] / image[r["width"]]) for r in rows
            if r["map"] == kind and r[f"{name}_ms"] and image.get(r["width"])]


def plot_map_dependence(rows, names, path):
    """Log-log panels (the photo's map if any, uniform random, smooth random):
    time on that map / time on the image's own map at the same size, one line
    per finder. 1 = the map makes no difference."""
    plt = _pyplot()
    titles = {"photo": "Photo's map", "random": "Uniform random map", "smooth": "Smooth random map"}
    kinds = [kind for kind in titles if any(r["map"] == kind for r in rows)]
    fig, axes = plt.subplots(1, len(kinds), figsize=(4.5 * len(kinds), 3.9), dpi=150,
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, kind in zip(axes, kinds):
        title = titles[kind]
        _style_axes(fig, ax)
        ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1.2)
        ends = []
        for name in names:
            points = map_ratios(rows, name, kind)
            if not points:
                continue
            x, y = zip(*points)
            ax.plot(x, y, color=ALGO_COLOURS.get(name, INK_2), linewidth=2, marker="o",
                    markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
            ends.append((x[-1], y[-1], f"{name} {y[-1]:.1f}x"))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.margins(x=0.35)
        _label_line_ends(ax, ends)
        ax.set_title(f"{title} vs the image's own map", loc="left", color=INK, fontsize=10)
        ax.set_xlabel("Pixels (H x W, log scale)", color=INK_2)
    axes[0].set_ylabel("Time on this map / on the image's map\n(log scale; 1 = no difference)",
                       color=INK_2)
    fig.tight_layout()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, labelcolor=INK_2, fontsize=9, ncol=len(labels),
               loc="upper left", bbox_to_anchor=(axes[0].get_position().x0, 0))
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


SOBEL_STYLES = {"opencv": ("-", "o", INK), "direct": ("--", "s", INK_2),
                "reused": ("-", "v", MUTED), "divide": ("-.", "D", INK_2),
                "fft": (":", "^", MUTED)}


def plot_sobel_methods(rows, path):
    """Log-log: time for the Sobel energy filter vs pixels, one line per method.

    The methods are not finders, so they use greys with different line styles
    and markers, plus a label at each line's end, leaving the finder colours
    to mean only the finders.
    """
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    _style_axes(fig, ax)
    ends = []
    for method, (style, marker, colour) in SOBEL_STYLES.items():
        x = [r["pixels"] for r in rows]
        y = [r[f"{method}_ms"] for r in rows]
        ax.plot(x, y, color=colour, linestyle=style, linewidth=2, marker=marker, markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=SOBEL_LABELS[method])
        ends.append((x[-1], y[-1], SOBEL_LABELS[method]))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.margins(x=0.45)
    _label_line_ends(ax, ends)
    ax.set_xlabel("Pixels (H x W, log scale)", color=INK_2)
    ax.set_ylabel("Time for one energy map (ms, log scale)", color=INK_2)
    ax.set_title("Item 2b: the Sobel filter computed five ways", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, labelcolor=INK_2, fontsize=9, ncol=2,
              loc="upper left", bbox_to_anchor=(0, -0.16))
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_carve_comparison(img, pictures, path):
    """One row per finder: its removed seams on the original, and its result."""
    plt = _pyplot()
    height, width = img.shape[:2]
    names = list(pictures)
    fig, axes = plt.subplots(len(names), 2, figsize=(8, 0.4 + 2.9 * len(names)),
                             dpi=150, squeeze=False)
    fig.patch.set_facecolor(SURFACE)

    for row_axes, name in zip(axes, names):
        result = pictures[name]["result"]
        panels = [
            (pictures[name]["seams"], f"{name}: removed seams"),
            # Pad the result back to the original size so both panels share
            # one scale and the result is visibly smaller.
            (pad_to(result, height, width, fill=252),
             f"{name}: result {result.shape[1]} x {result.shape[0]}"),
        ]
        for ax, (picture, title) in zip(row_axes, panels):
            ax.imshow(picture, interpolation="nearest")
            ax.set_title(title, loc="left", color=INK, fontsize=10)
            ax.axis("off")

    fig.suptitle("Red = vertical seams, blue = horizontal seams",
                 x=0.01, ha="left", color=INK_2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_seam_quality(rows, path):
    """One band per kind of energy map: a dot per map at seam cost / DP seam cost.

    Returns False (and draws nothing) if only DP was compared.
    """
    names = [n for n in dict.fromkeys(r["algo"] for r in rows) if n != REFERENCE]
    if not names:
        return False
    plt = _pyplot()
    kinds = list(dict.fromkeys(r["kind"] for r in rows))
    bands = [(kind, name) for kind in kinds for name in names
             if any(r["kind"] == kind and r["algo"] == name for r in rows)]
    fig, ax = plt.subplots(figsize=(7, 1.2 + 0.5 * len(bands)), dpi=150)
    _style_axes(fig, ax)
    ax.yaxis.grid(False)
    jitter = np.random.default_rng(0)  # only spreads the dots so they do not hide each other
    labelled = set()
    for y, (kind, name) in enumerate(bands):
        ratios = [r["ratio"] for r in rows
                  if r["kind"] == kind and r["algo"] == name and np.isfinite(r["ratio"])]
        ax.scatter(ratios, y + jitter.uniform(-0.18, 0.18, len(ratios)), s=18, alpha=0.55,
                   color=ALGO_COLOURS.get(name, INK_2), edgecolors="none",
                   label=None if name in labelled else name)
        labelled.add(name)
        median = float(np.median(ratios))
        ax.plot([median, median], [y - 0.32, y + 0.32], color=INK, linewidth=2)
    ax.axvline(1.0, color=MUTED, linestyle="--", linewidth=1.2)
    ax.annotate("DP (optimal)", (1.0, -0.75), xytext=(4, 0), textcoords="offset points",
                va="center", color=INK_2, fontsize=9)
    ax.set_yticks(range(len(bands)))
    ax.set_yticklabels([kind if len(names) == 1 else f"{kind}, {name}" for kind, name in bands])
    ax.set_ylim(len(bands) - 0.5, -1.1)
    ax.set_xlabel("Seam cost / DP seam cost on the same map (1 = optimal)", color=INK_2)
    title = "Seam cost compared with DP" + (f": {names[0]}" if len(names) == 1 else "")
    ax.set_title(f"{title}\nOne dot per energy map; black line = median",
                 loc="left", color=INK, fontsize=10)
    if len(names) > 1:
        ax.legend(frameon=False, labelcolor=INK_2, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return True


def plot_time_breakdown(rows, path):
    """Stacked bars per finder: time per seam finding the seam vs everything else.

    rows: carve rows with an "input" key. One panel per carved input, each on
    its own scale: graph cut only carves the small copy and is far slower, so
    one shared scale would flatten every other bar.
    """
    plt = _pyplot()
    keys = list(dict.fromkeys((r.get("picture", "image"), r["input"]) for r in rows))
    groups = [[r for r in rows if (r.get("picture", "image"), r["input"]) == k] for k in keys]
    inputs = [f"{picture}, {carve_input}" if any(p != "image" for p, _ in keys) else carve_input
              for picture, carve_input in keys]
    fig, axes = plt.subplots(len(groups), 1, squeeze=False, dpi=150,
                             figsize=(7, 0.9 + sum(1.0 + 0.55 * len(g) for g in groups)),
                             gridspec_kw={"height_ratios": [len(g) for g in groups]})
    axes = axes[:, 0]
    for ax, carve_input, group in zip(axes, inputs, groups):
        _style_axes(fig, ax)
        ax.yaxis.grid(False)
        y = np.arange(len(group))
        finding = [1000 * r["finder_s"] / r["seams"] for r in group]
        rest = [r["outside_per_seam_ms"] for r in group]
        ax.barh(y, finding, height=0.5, color=INK_2, edgecolor=SURFACE, linewidth=2,
                label="finding the seam")
        ax.barh(y, rest, left=finding, height=0.5, color=AXIS, edgecolor=SURFACE, linewidth=2,
                label="energy map + removing the seam")
        for yi, f, r in zip(y, finding, rest):
            ax.annotate(f"{f:.2f} + {r:.2f} ms", (f + r, yi), xytext=(6, 0),
                        textcoords="offset points", va="center", color=INK_2, fontsize=9)
        ax.set_yticks(y)
        ax.set_yticklabels([r["algo"] for r in group])
        ax.set_ylim(len(group) - 0.5, -0.5)
        ax.margins(x=0.3)
        ax.set_title(f"Input {carve_input}", loc="left", color=INK_2, fontsize=9)
    axes[0].set_title(f"Where a full carve spends its time\nInput {inputs[0]}",
                      loc="left", color=INK, fontsize=10)
    axes[-1].set_xlabel("Time per seam in a full carve (ms, median run"
                        + ("; each panel has its own scale)" if len(groups) > 1 else ")"),
                        color=INK_2)
    fig.tight_layout()
    # Added after the layout, just under the figure, so it never squeezes the panels;
    # bbox_inches="tight" then grows the saved image to include it.
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, labelcolor=INK_2, fontsize=9, ncol=2,
               loc="upper left", bbox_to_anchor=(axes[-1].get_position().x0, 0))
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_baselines(img, panels, path):
    """The original, scaling, cropping and each finder's result, on one scale."""
    plt = _pyplot()
    height, width = img.shape[:2]
    cols = 3
    grid_rows = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(grid_rows, cols, figsize=(3.1 * cols, 0.2 + 2.3 * grid_rows),
                             dpi=150, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (title, picture) in zip(axes.flat, panels):
        ax.imshow(pad_to(picture, height, width, fill=252), interpolation="nearest")
        ax.set_title(f"{title} ({picture.shape[1]} x {picture.shape[0]})",
                     loc="left", color=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + " --- |" * len(headers)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def _range(row, prefix, unit="ms", digits=3):
    """'median (lowest–highest)' from <prefix>_<unit>, _min_<unit>, _max_<unit>; 'skipped' for None."""
    median = row[f"{prefix}_{unit}"]
    if median is None:
        return "skipped"
    low, high = row[f"{prefix}_min_{unit}"], row[f"{prefix}_max_{unit}"]
    return f"{median:.{digits}f} ({low:.{digits}f}–{high:.{digits}f})"


def _unsteady(rows, label, prefixes, unit="ms"):
    """'name at <label> (2.1x)' for each timing whose highest pass or run is
    more than UNSTEADY times its lowest."""
    found = []
    for row in rows:
        for prefix in prefixes:
            if _is_unsteady(row, prefix, unit):
                low, high = row[f"{prefix}_min_{unit}"], row[f"{prefix}_max_{unit}"]
                found.append(f"{prefix} at {label(row)} ({high / low:.1f}x)")
    return found


def _steadiness(found):
    """A warning line for the summary if any timing was not steady."""
    if not found:
        return []
    return ["", f"Not steady (highest more than {UNSTEADY}x the lowest): {'; '.join(found)}. "
            "Treat these as rough."]


def _times(ratio):
    """'4.2x faster' or '1.5x slower'."""
    return f"{ratio:.1f}x faster" if ratio >= 1 else f"{1 / ratio:.1f}x slower"


def _is_unsteady(row, prefix, unit="ms"):
    """True if this timing's highest pass or run is more than UNSTEADY times its lowest."""
    low, high = row.get(f"{prefix}_min_{unit}"), row.get(f"{prefix}_max_{unit}")
    return bool(low and high and high > UNSTEADY * low)


def _slopes(rows, x_key, keys):
    """'label slope' for each (label, key) that has at least two points.

    Timings flagged as not steady are left out of the fit: one wild point
    among the few largest sizes can swing the slope a long way (the energy
    map once read 1.39 because of a single unsteady point).
    """
    out = []
    for label, key in keys:
        slope, dropped, rough = _slope_value(rows, x_key, key)
        if slope is None:
            continue
        if rough:
            note = " (rough: too few steady points, so unsteady ones were used)"
        elif dropped:
            note = f" ({dropped} unsteady point{'s' if dropped > 1 else ''} left out)"
        else:
            note = ""
        out.append(f"{label} {slope:.2f}{note}")
    return ", ".join(out)


def _slope_value(rows, x_key, key):
    """(slope or None, unsteady points left out, rough) for one column.

    If leaving out the unsteady timings would leave fewer than two points,
    every point is used instead and rough is True: a rough slope is more
    use than none, and the summary says it is rough.
    """
    prefix = key[:-len("_ms")] if key.endswith("_ms") else None
    measured = [r for r in rows if r[key] is not None]
    steady = [r for r in measured if not (prefix and _is_unsteady(r, prefix))]
    slope = loglog_slope([r[x_key] for r in steady], [r[key] for r in steady])
    if slope is None and len(steady) < len(measured):
        slope = loglog_slope([r[x_key] for r in measured], [r[key] for r in measured])
        return slope, 0, slope is not None
    return slope, len(measured) - len(steady), False


SENTENCE_NAMES = {"dp": "DP", "greedy": "Greedy", "graphcut": "Graph cut"}


def _start(name):
    """A finder's name where it starts a sentence: 'DP', 'Greedy', 'Graph cut'.
    Tables and mid-sentence mentions keep the code names (dp, greedy, graphcut)."""
    return SENTENCE_NAMES.get(name, name[:1].upper() + name[1:])


def _fit_rows(rows, x_key, key):
    """The rows a slope is fitted on, as in _slope_value: the SLOPE_POINTS
    largest steady sizes, or every measured one if that leaves fewer than two."""
    prefix = key[:-len("_ms")] if key.endswith("_ms") else None
    measured = [r for r in rows if r[key] is not None and r[key] > 0]
    steady = [r for r in measured if not (prefix and _is_unsteady(r, prefix))]
    use = steady if len(steady) >= 2 else measured
    return sorted(use, key=lambda r: r[x_key])[-SLOPE_POINTS:]


def _and_list(items):
    """['a', 'b', 'c'] -> 'a, b and c'."""
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _more_or_less(ratio):
    """1.45 -> '45% more', 0.8 -> '20% less'."""
    return f"{100 * (ratio - 1):.0f}% more" if ratio >= 1 else f"{100 * (1 - ratio):.0f}% less"


def _energy_vs_dp(ratio):
    """1.45 -> '45% more energy than DP's', 1.0 -> 'the same energy as DP's'."""
    if abs(ratio - 1) < 0.0005:
        return "the same energy as DP's"
    return f"{_more_or_less(ratio)} energy than DP's"


def _share(fraction):
    """0.873 -> '87%', 0.997 -> 'over 99%', so a rounded 100% or 0% never misleads."""
    if fraction >= 0.995:
        return "over 99%"
    if fraction <= 0.005:
        return "under 1%"
    return f"{100 * fraction:.0f}%"


def _map_effect(map_rows, name):
    """At the largest size where `name` ran on every kind of map:
    (that size's image-map row, {kind: time / time on the image's map},
    {kind: graph cut searches}). None if there is no such size."""
    for width in sorted({r["width"] for r in map_rows}, reverse=True):
        at = {r["map"]: r for r in map_rows if r["width"] == width}
        if all(at.get(kind) and at[kind][f"{name}_ms"] for kind in MAP_KINDS):
            base = at["image"][f"{name}_ms"]
            kinds = [kind for kind in ALL_MAP_KINDS if at.get(kind) and at[kind][f"{name}_ms"]]
            return (at["image"], {kind: at[kind][f"{name}_ms"] / base for kind in kinds},
                    {kind: at[kind]["graphcut_searches"] for kind in kinds})
    return None


def _other_ratios(map_rows, name):
    """Every (time on another map / time on the image's map) for one finder."""
    return [ratio for kind in ALL_MAP_KINDS[1:] for _pixels, ratio in map_ratios(map_rows, name, kind)]


def _map_independent(map_rows, name, low=0.75, high=4 / 3):
    """True if every size's time on every other map is within low..high of the image's."""
    ratios = _other_ratios(map_rows, name)
    return bool(ratios) and all(low <= ratio <= high for ratio in ratios)


def _map_factor(map_rows, name):
    """Largest factor (1 or more) between a finder's time on another map and on
    the image's map, over every size and kind; None if not measured."""
    ratios = _other_ratios(map_rows, name)
    return max(max(r, 1 / r) for r in ratios) if ratios else None


def _greedy_edge_note(map_rows):
    """Why greedy's time moves with the map, if its seam explains it: at the
    largest size, how often it runs along the edge on each map. None if the
    edge does not differ enough between the maps to explain anything."""
    width = max(r["width"] for r in map_rows)
    edge = {r["map"]: r["greedy_edge"] for r in map_rows if r["width"] == width}
    if any(edge.get(kind) is None for kind in MAP_KINDS) or edge["image"] - edge["random"] < 0.25:
        return None
    return (f"on the image's map its seam runs along the edge in {100 * edge['image']:.0f}% of "
            f"rows, against {100 * edge['random']:.0f}% and {100 * edge['smooth']:.0f}% on the "
            "random maps, and on the edge it checks two neighbours instead of three")


def _carve_speed(name, row, ref):
    """How a finder's whole carve compares with DP's, and where its time goes.

    No full stop at the end, so callers can add a section reference.
    """
    finder = _times(ref["finder_s"] / row["finder_s"])
    whole = _times(ref["total_s"] / row["total_s"])
    if row["finder_s"] < ref["finder_s"]:
        return (f"{name} finds its seams {finder} than DP, but the whole carve is only {whole}: "
                f"{_share(1 - row['finder_share'])} of its time goes on rebuilding the energy map "
                "and removing seams, which every finder has to do")
    return (f"{name} finds its seams {finder} than DP, and the whole carve is {whole}: "
            f"{_share(row['finder_share'])} of its time goes on finding seams, so the energy map "
            "and seam removal hardly matter")


def conclusions(setup, timing, shapes, quality, carve_rows, threads, base_rows, map_rows=(),
                sobel_rows=(), tile_rows=()):
    """Short answers to the brief's questions, worded from this run's numbers.

    Returns Markdown lines, or [] when there is nothing to compare with DP.
    Every sentence points to the section with the detail; the visual
    judgement is left to the pictures, because no number captures it.
    """
    names = setup["finders"]
    others = [n for n in names if n != REFERENCE]
    if REFERENCE not in names or not others:
        return []
    full_input = f"{setup['width']} x {setup['height']} -> {setup['new_width']} x {setup['new_height']}"
    by_input = {}  # (picture, carve input) -> {finder: row}, each picture's full input first
    first_input = {}
    for r in carve_rows:
        picture = r.get("picture", "image")
        first_input.setdefault(picture, r["input"])
        by_input.setdefault((picture, r["input"]), {})[r["algo"]] = r
    ref = by_input.get(("image", full_input), {}).get(REFERENCE)

    def largest_pair(n, picture):
        """(input, n's row, DP's row) for the largest input of `picture` both carved."""
        key = next((k for k, algos in by_input.items()
                    if k[0] == picture and n in algos and REFERENCE in algos), None)
        return (key[1], by_input[key][n], by_input[key][REFERENCE]) if key else (None, None, None)

    lines = ["## What the numbers say", "",
             "Written from this run's numbers; the numbered sections have the detail.", ""]

    for n in others:
        # Items 3, 4 and 5c: seam cost on identical energy maps.
        rows = [r for r in quality if r["algo"] == n and np.isfinite(r["ratio"])]
        if rows:
            mean = float(np.mean([r["ratio"] for r in rows]))
            worst = max(r["ratio"] for r in rows)
            verdict = "yes" if mean <= 1.01 else ("close" if mean <= 1.05 else "no")
            optimal = sum(r["optimal"] for r in rows)
            by_kind = {}
            for r in rows:
                by_kind.setdefault(r["kind"], []).append(r["ratio"])
            if worst < 1.0005:
                lines.append(f"- **Is {n} as good as DP? On seam cost, yes.** {_start(n)} found the "
                             f"cheapest seam on {optimal} of {len(rows)} energy maps "
                             f"({', '.join(by_kind)}), so its seams cost exactly what DP's do "
                             "(section 3).")
            else:
                spread = ", ".join(f"{_more_or_less(float(np.mean(v)))} on {k} maps"
                                   for k, v in by_kind.items())
                lines.append(f"- **Is {n} as good as DP? On seam cost, {verdict}.** {_start(n)} found "
                             f"the cheapest seam on {optimal} of {len(rows)} energy maps. Its "
                             f"seams cost on average {spread}; the worst was "
                             f"{_more_or_less(worst)} (section 3).")

        # Items 3, 4 and 5c: the full carve on each picture, seams and detail kept.
        # The largest input both n and DP carved: the picture itself, or its
        # small copy when n has a pixel limit (graph cut).
        clauses, files, all_same, crop_like = [], [], True, []
        for picture in ("image", "photo"):
            carved, r, r_ref = largest_pair(n, picture)
            if not (r and r_ref and r_ref["energy_removed"] and r["energy_kept"] is not None):
                continue
            full = carved == first_input[picture]
            files.append(f"compare{'' if picture == 'image' else '_photo'}{'' if full else '_small'}.png")
            all_same &= (r["overlap_with_dp"] >= 0.9995
                         and abs(r["energy_removed"] / r_ref["energy_removed"] - 1) < 0.0005)
            clause = (f"on the {picture}{'' if full else f' (small copy, {carved})'}, {n}'s seams "
                      f"removed {_energy_vs_dp(r['energy_removed'] / r_ref['energy_removed'])} and "
                      f"the result keeps {100 * r['energy_kept']:.1f}% of the {picture}'s energy "
                      f"(DP: {100 * r_ref['energy_kept']:.1f}%); the pixels the two remove overlap "
                      f"by {100 * r['overlap_with_dp']:.0f}%")
            if r.get("crop_likeness") is not None and r_ref.get("crop_likeness") is not None:
                clause += (f", and {n}'s removed pixels match the closest crop's by "
                           f"{100 * r['crop_likeness']:.0f}% (DP: {100 * r_ref['crop_likeness']:.0f}%)")
                if r["crop_likeness"] - r_ref["crop_likeness"] >= 0.15:
                    crop_like.append(f"the {picture}")
            clauses.append(clause)
        if clauses:
            text = f"- **Full carve, {n} vs DP:** " + "; ".join(clauses)
            text = text.replace(":** on the", ":** On the", 1) + "."
            if all_same:
                text += f" So the results are the same pictures ({_and_list(files)}, section 4)."
            elif crop_like:
                text += (f" So on {_and_list(crop_like)} {n} behaves far more like a plain crop "
                         "than DP does, cutting off a strip where DP spreads its seams through "
                         f"low-energy areas: compare the seams in {_and_list(files)} (section 4).")
            else:
                text += (" Whether the difference is visible depends on the image: compare the "
                         f"seams in {_and_list(files)} (section 4).")
            lines.append(text)

        # Items 3, 4 and 5b: running time, on the image.
        carved, r, r_ref = largest_pair(n, "image")
        where = "" if carved == full_input else f" of the small copy ({carved})"
        both = [t for t in timing if t[f"{REFERENCE}_ms"] is not None and t[f"{n}_ms"] is not None]
        if both and r and r_ref and r["total_s"] > 0 and r["finder_s"] > 0 and r_ref["finder_s"] > 0:
            t = both[-1]
            lines.append(f"- **Speed, {n} vs DP:** at {t['width']} x {t['height']} one {n} seam "
                         f"takes {t[f'{n}_ms']:.3f} ms against DP's {t[f'{REFERENCE}_ms']:.3f} ms "
                         f"({_times(t[f'{REFERENCE}_ms'] / t[f'{n}_ms'])}). In the full "
                         f"carve{where}, {_carve_speed(n, r, r_ref)} (sections 1 and 4).")

    # Item 5b: time complexity, from the height-vs-width experiment.
    main_shapes = [r for r in shapes if r["ladder"] == "main"]
    tall = [r for r in main_shapes if r["vary"] == "height"]
    wide = [r for r in main_shapes if r["vary"] == "width"]
    h_slope, _, h_rough = _slope_value(tall, "height", f"{REFERENCE}_ms")
    w_slope, _, w_rough = _slope_value(wide, "width", f"{REFERENCE}_ms")
    if h_slope is not None and w_slope is not None:
        how = ("far more than" if h_slope >= w_slope + 0.4 else
               "more than" if h_slope >= w_slope + 0.1 else "about as much as")
        lines.append(f"- **Time complexity (Item 5b):** per seam, DP's time grows with the height "
                     f"(slope {h_slope:.2f}{', rough' if h_rough else ''}) {how} with "
                     f"the width (slope {w_slope:.2f}{', rough' if w_rough else ''}). "
                     + ("Its work is O(H x W), but at these sizes the fixed cost of the NumPy calls "
                        "it makes for every row dominates (section 2)." if how != "about as much as"
                        else "Its work is O(H x W), and at these sizes both sides count about equally "
                        "(section 2).")
                     + (" Greedy is O(H + W) per seam." if "greedy" in names else "")
                     + " Carving k seams costs k x (O(H x W) + the finder's own cost), because "
                     "every seam needs a new energy map and a seam removal; for DP"
                     + (" and greedy" if "greedy" in names else "") + " that is O(k x H x W).")

    # Items 4 and 5b: finders too slow for the main ladder, from the small one.
    small_shapes = [r for r in shapes if r["ladder"] == "small"]
    for n in others:
        if any(r[f"{n}_ms"] is not None for r in main_shapes):
            continue
        tall_s = [r for r in small_shapes if r["vary"] == "height"]
        wide_s = [r for r in small_shapes if r["vary"] == "width"]
        hs, _, hs_rough = _slope_value(tall_s, "height", f"{n}_ms")
        ws, _, ws_rough = _slope_value(wide_s, "width", f"{n}_ms")
        if hs is None or ws is None:
            continue
        dp_hs, _, _ = _slope_value(tall_s, "height", f"{REFERENCE}_ms")
        dp_ws, _, _ = _slope_value(wide_s, "width", f"{REFERENCE}_ms")
        dp_same = (f" (DP on the same maps: {dp_hs:.2f} and {dp_ws:.2f})"
                   if dp_hs is not None and dp_ws is not None else "")
        text = (f"- **Time complexity of {n} (Items 4, 5b):** on the small uniform random maps "
                f"of section 2 its time per seam grows with slope {hs:.2f}{', rough' if hs_rough else ''} in the height and "
                f"{ws:.2f}{', rough' if ws_rough else ''} in the width{dp_same}, so doubling the "
                f"height multiplies its time by about {2 ** hs:.1f} and doubling the width by "
                f"about {2 ** ws:.1f}.")
        if n == "graphcut":
            text += (" It runs Edmonds-Karp max-flow [8] on a graph with one node per pixel and a "
                     "few edges each, so V and E both grow like H x W and the worst case is "
                     "O(V x E^2) = O((H x W)^3), and it rebuilds that graph for every seam.")
            if max(hs, ws) < 3:
                text += (" The measured growth is well below that worst case, under which doubling "
                         "one side would multiply the time by 8.")
        lines.append(text[:-1] + " (section 2).")

    # Items 4 and 5b: does the content of the map change the time?
    effects = {n: _map_effect(map_rows, n) for n in names}
    effects = {n: e for n, e in effects.items() if e}
    if "graphcut" in effects:
        image_row, ratio, searches = effects["graphcut"]
        changed = not _map_independent(map_rows, "graphcut")
        text = (f"- **Does the image's content change the time? (Items 4, 5b):** for graph cut, "
                f"{'yes' if changed else 'hardly'}. At {image_row['width']} x "
                f"{image_row['height']} one graphcut seam took {ratio['random']:.1f}x as long on "
                f"a uniform random map and {ratio['smooth']:.1f}x as long on a smooth one as on "
                "the image's own map")
        if all(searches.values()):
            text += (f", with {searches['random']} and {searches['smooth']} searches for "
                     f"augmenting paths instead of {searches['image']}")
        text += "."
        if "photo" in ratio:
            text += (f" On the photo's map, cropped to the same shape, it took {ratio['photo']:.1f}x "
                     "as long as on the image's"
                     + (f", with {searches['photo']} searches" if searches.get("photo") else "")
                     + ".")
        if changed:
            text += " So graph cut's speed depends on the image as well as its size"
            slower = ratio["random"] > max(1.0, ratio.get("photo", 1.0))
            text += ((", and its times on the random maps of section 2 are slower than "
                      f"{'these images' if 'photo' in ratio else 'this image'} would give.")
                     if slower else ".")
        compared = []
        for n in names:
            factor = _map_factor(map_rows, n) if n != "graphcut" else None
            if factor is None:
                continue
            clause = f"{n}'s time by at most {factor:.2f}x"
            note = _greedy_edge_note(map_rows) if n == "greedy" else None
            if note and not _map_independent(map_rows, n):
                clause += f", because {note}"
            compared.append(clause)
        if compared:
            text += " For comparison, the map changed " + "; ".join(compared) + "."
        lines.append(text[:-1] + " (section 2).")

    # Item 5b / 5d: space, and whether DP needs its whole table.
    with_memory = [t for t in timing if t[f"{REFERENCE}_peak_kb"] is not None]
    if with_memory:
        t = with_memory[-1]
        others_memory = ", ".join(f"{n} {t[f'{n}_peak_kb']:.1f} KB" for n in others
                                  if t[f"{n}_peak_kb"] is not None)
        peak_bytes = 1024 * t[f"{REFERENCE}_peak_kb"]
        text = (f"- **Space and the DP table (Items 5b, 5d):** at {t['width']} x {t['height']} "
                f"DP's peak is {t[f'{REFERENCE}_peak_kb']:.1f} KB, "
                f"{peak_bytes / t['pixels']:.2f} bytes per pixel, of which "
                f"{100 * t['pixels'] / peak_bytes:.0f}% is the int8 backpointer table "
                "(1 byte per pixel)."
                + (f" For comparison: {others_memory}." if others_memory else "") + " DP does not need "
                "the full cost table, since dp.py keeps only one row of costs, but it does "
                "need the backpointers to rebuild the seam, unless it rebuilds it in a way "
                "that trades extra time for memory.")
        # Finders that never reached the largest size: compare where they stopped.
        for n in others:
            if t[f"{n}_peak_kb"] is not None:
                continue
            measured = [m for m in with_memory if m[f"{n}_peak_kb"] is not None]
            if not measured:
                continue
            m = measured[-1]
            text += (f" At {m['width']} x {m['height']}, the largest size {n} ran on, its peak is "
                     f"{m[f'{n}_peak_kb']:.1f} KB ({1024 * m[f'{n}_peak_kb'] / m['pixels']:.0f} "
                     f"bytes per pixel) against DP's {m[f'{REFERENCE}_peak_kb']:.1f} KB "
                     f"({1024 * m[f'{REFERENCE}_peak_kb'] / m['pixels']:.2f})"
                     + (": it holds its whole graph, a Python dict of edges for every pixel."
                        if n == "graphcut" else "."))
        lines.append(text[:-1] + " (section 1).")

    # Item 5d: multithreading.
    carve_speedups = [r["speedup"] for r in threads if r["part"].startswith("full carve")]
    if carve_speedups:
        low, high = min(carve_speedups), max(carve_speedups)
        gain = "no worthwhile gain" if high < 1.2 else f"up to {high:.2f}x faster"
        text = (f"- **Does multithreading help? (Item 5d):** {gain}. OpenCV on "
                f"{setup['cv_threads']} threads instead of 1 changed full-carve times by "
                f"{low:.2f}x to {high:.2f}x. Only the energy map uses threads here: DP's rows "
                "must run in order, and the cells within one row are already done together in "
                "NumPy's compiled code, too little work to share between threads.")
        for n in names:
            if any(r["part"] == f"full carve, {n}" for r in threads):
                continue
            text += (f" {_start(n)} was not timed here, because it cannot carve the full image within "
                     "its pixel limit"
                     + ("; its max-flow runs as plain Python loops, which OpenCV's threads cannot "
                        "speed up, and Edmonds-Karp finds one augmenting path at a time, each "
                        "depending on the flow the earlier ones left; parallel max-flow methods "
                        "such as push-relabel exist, but it does not use one"
                        if n == "graphcut" else "") + ".")
        lines.append(text[:-1] + " (section 5).")

    # The brief's figure: scaling and cropping, on each picture.
    clauses, pictures_used = [], []
    for picture in ("image", "photo"):
        rows = {b["method"]: b for b in base_rows if b.get("picture", "image") == picture}
        dp_row = by_input.get((picture, first_input.get(picture)), {}).get(REFERENCE)
        crop, centre = rows.get("cropping"), rows.get("centre crop")
        if not (crop and dp_row and crop["energy_kept"] is not None and dp_row["energy_kept"] is not None):
            continue
        dp_kept = f"DP's {100 * dp_row['energy_kept']:.1f}%"
        if abs(crop["energy_kept"] - dp_row["energy_kept"]) < 0.0005:
            compare = f"the same as {dp_kept}"
        elif crop["energy_kept"] > dp_row["energy_kept"]:
            compare = f"more than {dp_kept}"
        else:
            compare = f"less than {dp_kept}"
        clauses.append(f"on the {picture} the best crop kept {100 * crop['energy_kept']:.1f}% of "
                       f"the original energy, {compare}"
                       + (f" (a centre crop: {100 * centre['energy_kept']:.1f}%)"
                          if centre and centre["energy_kept"] is not None else ""))
        pictures_used.append("baselines.png" if picture == "image" else "baselines_photo.png")
    scale = next((b for b in base_rows if b["method"] == "scaling" and b.get("picture", "image") == "image"), None)
    crop = next((b for b in base_rows if b["method"] == "cropping" and b.get("picture", "image") == "image"), None)
    if clauses:
        text = "- **Seam carving vs scaling and cropping:** " + "; ".join(clauses) + "."
        text = text.replace(":** on the", ":** On the", 1)
        if ref and scale and crop:
            text += (f" DP's carve of the image took {1000 * ref['total_s']:.0f} ms against "
                     f"{scale['time_ms']:.2f} ms to scale and {crop['time_ms']:.2f} ms to crop.")
        text += (" Energy kept ignores context (a crop can cut part of the scene off), so judge "
                 f"with {_and_list(pictures_used)} (section 4).")
        lines.append(text)

    # Item 2b: would divide and conquer speed up the Sobel filter?
    if sobel_rows:
        t = sobel_rows[-1]
        ratio = t["divide_ms"] / t["direct_ms"]
        if ratio < 0.9:
            verdict = "not by cutting the work"
            slower = [r for r in sobel_rows if r["divide_ms"] > r["direct_ms"]]
            worst = max(slower, key=lambda r: r["divide_ms"] / r["direct_ms"], default=None)
            effect = _tile_effect(tile_rows)
            memory = _memory_effect(sobel_rows)
            detail = (f"Divide and conquer ran {1 / ratio:.1f}x faster than the same filter in one "
                      f"pass with new arrays at {t['width']} x {t['height']}"
                      + (f" but was slower at smaller sizes (up to "
                         f"{worst['divide_ms'] / worst['direct_ms']:.1f}x at {worst['width']} x "
                         f"{worst['height']})" if worst else ""))
            if memory and memory[0] >= 1.5 * memory[1]:
                detail += (f". The same one pass writing into reused arrays took "
                           f"{t['reused_ms']:.2f} ms, {t['direct_ms'] / t['reused_ms']:.1f}x faster "
                           "than with new arrays, so the gain comes from making smaller fresh arrays "
                           "(one tile's at a time), not from less work: it does the same O(H x W) work")
            elif effect and effect[2] >= 1.5:
                detail += (f". With one whole-image tile the same code took {effect[2]:.1f}x as "
                           f"long as with tiles of {effect[0]['label']}, so the gain most likely "
                           "comes from the CPU's cache, not from less work: it does the same "
                           "O(H x W) work")
            else:
                detail += "; it does the same O(H x W) work, and the cause of the gain is unclear"
        else:
            verdict = "no"
            detail = (f"Divide and conquer took {ratio:.2f}x the time of the same filter in one pass: "
                      "it does the same work plus the cost of its recursion")
        fastest = min(SOBEL_METHODS, key=lambda m: t[f"{m}_ms"])
        diff = max(t[f"{m}_diff"] for m in SOBEL_METHODS)
        lines.append(f"- **Would divide and conquer speed up the Sobel filter? (Item 2b):** {verdict}. "
                     f"At {t['width']} x {t['height']}: "
                     + ", ".join(f"{SOBEL_LABELS[m]} {t[m + '_ms']:.2f} ms" for m in SOBEL_METHODS)
                     + f"; every method gives the same map (largest difference {diff:.0e}). "
                     f"{detail}. Every output pixel needs its own 3 x 3 neighbourhood, so any method "
                     "must read all H x W pixels, and FFT convolution, itself a divide-and-conquer "
                     f"algorithm, is O(H x W log(H x W)). The fastest here was {SOBEL_LABELS[fastest]} "
                     "(section 6).")

    unsteady = (len(_unsteady(timing, str, ["energy"] + names))
                + len(_unsteady(shapes, str, names))
                + len(_unsteady(map_rows, str, names))
                + len(_unsteady(threads, str, ["one_thread", "default_threads"]))
                + len(_unsteady(carve_rows, str, ["total", "finder"], "s"))
                + len(_unsteady(sobel_rows, str, list(SOBEL_METHODS)))
                + len(_unsteady(tile_rows, str, ["divide"])))
    images = (f"two test images, `{setup['image']}` and the photo `{setup['photo']}`"
              if setup.get("photo") else f"one test image (`{setup['image']}`)")
    lines.append(f"- **Caveats:** {images}; {unsteady} timing"
                 f"{'' if unsteady == 1 else 's'} flagged as not steady; absolute times change "
                 "between runs on the same machine, so compare ratios rather than milliseconds.")
    return lines + [""]


REFERENCES = [
    "S. Avidan and A. Shamir, \"Seam carving for content-aware image resizing,\" ACM Transactions "
    "on Graphics 26(3), 2007. https://doi.org/10.1145/1275808.1276390",
    "M. Rubinstein, A. Shamir and S. Avidan, \"Improved seam carving for video retargeting,\" ACM "
    "Transactions on Graphics 27(3), 2008 (forward energy, graph-cut seams). "
    "https://doi.org/10.1145/1360612.1360615",
    "A. Das, \"Real-world dynamic programming: seam carving,\" 2019. "
    "https://avikdas.com/2019/05/14/real-world-dynamic-programming-seam-carving.html",
    "Python documentation, \"timeit: Measure execution time of small code snippets.\" "
    "https://docs.python.org/3/library/timeit.html",
    "pyperf documentation, \"Run a benchmark\" and \"Analyze benchmark results.\" "
    "https://pyperf.readthedocs.io/en/latest/run_benchmark.html, "
    "https://pyperf.readthedocs.io/en/latest/analyze.html",
    "Python documentation, \"tracemalloc: Trace memory allocations.\" "
    "https://docs.python.org/3/library/tracemalloc.html",
    "OpenCV documentation, \"Utility and system functions and macros\" (cv::setNumThreads). "
    "https://docs.opencv.org/4.x/db/de0/group__core__utils.html",
    "J. Edmonds and R. M. Karp, \"Theoretical improvements in algorithmic efficiency for network "
    "flow problems,\" Journal of the ACM 19(2), 1972, pp. 248-264. "
    "https://doi.org/10.1145/321694.321699",
]


def _times_short(ratio):
    """1234.5 -> '1235x', 3.21 -> '3.2x'."""
    return f"{ratio:.0f}x" if ratio >= 10 else f"{ratio:.1f}x"


def _memory_effect(rows):
    """How much each one-pass Sobel slowed down per pixel, from its own fastest
    size to the largest: (with new arrays, with reused arrays). None with
    fewer than two sizes."""
    if len(rows) < 2:
        return None

    def growth(method):
        per_pixel = [r[f"{method}_ms"] / r["pixels"] for r in rows]
        return per_pixel[-1] / min(per_pixel)
    return growth("direct"), growth("reused")


def _memory_sentence(rows):
    """Why the one-pass filter slows down per pixel at large sizes, worded from
    its two versions ('' with fewer than two sizes)."""
    effect = _memory_effect(rows)
    if not effect:
        return ""
    fresh, reused = effect
    t = rows[-1]
    size = f"{t['width']} x {t['height']}"
    text = (f"New arrays or reused ones. The same one pass writing into arrays it reuses took "
            f"{t['reused_ms']:.2f} ms at {size}, against {t['direct_ms']:.2f} ms when it makes new "
            f"arrays for each step. Per pixel, from its own fastest size to {size}, the version "
            f"with new arrays slowed {fresh:.1f}x and the version with reused arrays {reused:.1f}x.")
    if fresh >= 1.5 * reused:
        text += (" The arithmetic is the same, so most of the slowdown at large sizes is the cost "
                 "of making fresh arrays"
                 + (f"; the reused version's own {reused:.1f}x is most likely the CPU's cache, as "
                    "the arrays outgrow it" if reused >= 1.2 else "")
                 + ". Large arrays most likely get new memory from the operating system each time, "
                 "which is set up page by page when first used; that step was not timed on its own.")
    else:
        text += (" Both slowed by a similar amount, so the cost is in reaching the memory itself "
                 "(most likely the CPU's cache), not in making fresh arrays.")
    return text


def _energy_memory_note(sobel_rows):
    """Section 1's reason why the energy map's time per pixel grows, backed by
    section 6's one-pass filter with new and with reused arrays."""
    effect = _memory_effect(sobel_rows)
    if effect and effect[0] >= 1.5 * effect[1]:
        return (" Section 6 shows what that costs: the same Sobel arithmetic slowed "
                f"{effect[0]:.1f}x per pixel across these sizes when it made new arrays, but only "
                f"{effect[1]:.1f}x when it reused them. That most likely explains the energy map's "
                "slope above 1 here; it was not measured on the energy map itself.")
    return (" As the image grows those arrays outgrow a CPU core's own fast cache (L2, about 1 to "
            "4 MB on current CPUs), and each pixel waits longer for memory. That most likely "
            "explains a slope above 1 across these sizes; it was not measured directly.")


def _tile_sentence(tile_rows, rows=()):
    """What the tile-size check shows, worded from its numbers ('' if it shows
    nothing). rows: the Sobel methods' rows, for the memory comparison."""
    effect = _tile_effect(tile_rows)
    if not effect:
        return ""
    best, whole, ratio = effect
    smallest = tile_rows[0]
    pixels = whole["width"] * whole["height"]
    kb = lambda tiles: pixels / tiles * 8 / 1024  # one float64 array for the average tile
    text = (f"The same divide-and-conquer code was fastest with tiles of {best['label']} "
            f"({best['divide_ms']:.2f} ms).")
    if smallest is not best and smallest["divide_ms"] > 1.5 * best["divide_ms"]:
        text += (f" Smaller tiles were slower because every tile costs a Python call and a few "
                 f"NumPy calls: {smallest['tiles']} tiles of {smallest['label']} took "
                 f"{smallest['divide_ms']:.2f} ms.")
    if ratio >= 1.5:
        text += (f" Bigger tiles were slower too, up to {whole['divide_ms']:.2f} ms for one "
                 f"whole-image tile ({ratio:.1f}x the best), although the work is the same: every "
                 f"step of the filter makes a new array, about {kb(best['tiles']):.0f} KB for each "
                 f"of the {best['tiles']} tiles of {best['label']} and about {kb(1):.0f} KB for the "
                 "whole image.")
        memory = _memory_effect(rows)
        if memory and memory[0] >= 1.5 * memory[1]:
            text += (" The reused-array pass above shows that large new arrays are what is slow, "
                     "so one whole-image tile pays that cost and small tiles avoid it.")
        else:
            text += (" Big arrays also outgrow the CPU's fast cache (L2, about 1 to 4 MB), which "
                     "most likely explains it.")
    else:
        text += (f" One whole-image tile took {whole['divide_ms']:.2f} ms, close to the best, "
                 "so memory makes little difference at this size.")
    return text


def _sobel_section(rows, tile_rows=()):
    """Section 6: Item 2b, would divide and conquer speed up the Sobel filter?"""
    if not rows:
        return []
    methods = list(SOBEL_METHODS)
    out = ["## 6. Item 2b: would divide and conquer speed up the Sobel filter?", "",
           "The energy map's Sobel filter (|gx| + |gy| on the greyscale image) computed five ways "
           "on the same images: OpenCV's cv2.Sobel, which energy.py uses; the same 3 x 3 filter "
           "in one NumPy pass over the whole image, making new arrays for each step; the same "
           "pass writing into arrays made once and reused, which does the same arithmetic "
           "without making new arrays; divide and conquer, which splits the image in "
           f"half along its longer side until each tile is at most {SOBEL_LEAF} x {SOBEL_LEAF}, "
           "filters each tile with a one-pixel border taken from its neighbours and writes it "
           "into place; and convolution by FFT. Largest difference = the biggest gap from "
           "OpenCV's map anywhere in the image, over the other four methods.", ""]
    table = [[f"{r['width']} x {r['height']}", r["pixels"]] + [_range(r, m) for m in methods]
             + [f"{max(r[f'{m}_diff'] for m in methods):.0e}"] for r in rows]
    out.append(markdown_table(["Size", "Pixels"] + [f"{SOBEL_LABELS[m]} (ms)" for m in methods]
                              + ["Largest difference"], table))
    out += _steadiness(_unsteady(rows, lambda r: f"{r['width']} x {r['height']}", methods))
    out += ["", "Log-log slope of time vs pixels: "
            + _slopes(rows, "pixels", [(SOBEL_LABELS[m], f"{m}_ms") for m in methods]) + "."]
    t = rows[-1]
    ratio = t["divide_ms"] / t["direct_ms"]
    out += ["", f"At {t['width']} x {t['height']}, against the one pass with new arrays: divide and "
            f"conquer {ratio:.2f}x the time, the same pass with reused arrays "
            f"{t['reused_ms'] / t['direct_ms']:.2f}x, FFT convolution "
            f"{t['fft_ms'] / t['direct_ms']:.2f}x, OpenCV {t['opencv_ms'] / t['direct_ms']:.2f}x."]
    reads = (SOBEL_LEAF + 2) ** 2 / SOBEL_LEAF ** 2 - 1
    why = ("Why. Each output pixel depends only on its own 3 x 3 neighbourhood, so every method "
           "has to read all H x W pixels: O(H x W) is a floor no method can go below, and the "
           "one-pass filter already reaches it with a fixed number of operations per pixel. "
           "Divide and conquer splits the image with T(n) = 2T(n/2) + O(1) and does all its "
           "filtering at the leaves, O(n) in total, so it is O(H x W) as well and cannot cut the "
           f"work; it also re-reads the tile borders (a {SOBEL_LEAF} x {SOBEL_LEAF} tile reads "
           f"{100 * reads:.0f}% more pixels than it writes).")
    slower_rows = [r for r in rows if r["divide_ms"] > r["direct_ms"]]
    slower = [f"{r['width']} x {r['height']}" for r in slower_rows]
    effect = _tile_effect(tile_rows)
    memory = _memory_effect(rows)
    if ratio < 0.9:
        why += f" It still ran faster at {t['width']} x {t['height']}"
        if slower:
            why += f", though slower at {_and_list(slower)}"
        if memory and memory[0] >= 1.5 * memory[1]:
            why += (". The one pass with reused arrays shows why: large new arrays are slow to "
                    "make, and divide and conquer only ever makes small ones, one tile's at a "
                    "time. That is a constant-factor gain from memory handling, not less work.")
            if slower and all(r["direct_ms"] <= 1.5 * r["reused_ms"] for r in slower_rows):
                why += (" At the smaller sizes the one pass with new arrays was within 1.5x of the "
                        "reused-array pass, so making arrays cost little there and the tiles only "
                        "added their own cost.")
        elif effect and effect[2] >= 1.5:
            why += (". The tile-size check below shows a constant-factor gain that most likely "
                    "comes from the CPU's cache, not less work.")
        else:
            why += (". Neither the reused-array pass nor the tile-size check shows a clear "
                    "memory effect, so the cause is unclear; the work is the same.")
    else:
        why += " So it can only add the cost of the recursion, as measured."
    why += (" FFT convolution is itself divide and conquer: the FFT splits the transform into "
            "its even and odd halves, T(n) = 2T(n/2) + O(n) = O(n log n), so it costs "
            "O(H x W log(H x W)), more than one pass. It pays off only for large kernels, "
            "where it replaces k x k operations per pixel; with k = 3 there is nothing to save. "
            "What divide and conquer does offer is independent tiles that could be filtered on "
            "different threads at once, but OpenCV can already run its filters on several "
            "threads; section 5 measures what that gains on the energy map.")
    memory_text = _memory_sentence(rows)
    out += ["", why, ""] + ([memory_text, ""] if memory_text else []) + [
        "![Sobel filter, five ways](sobel_methods.png)", ""]
    if tile_rows:
        first = tile_rows[0]
        out += ["### Tile size", "",
                f"Divide and conquer at {first['width']} x {first['height']} with different largest "
                "tiles, up to one tile for the whole image. Every row does the same filtering work; "
                "only the tile size changes.", "",
                markdown_table(["Largest tile", "Tiles", "Time (ms)"],
                               [[r["label"], r["tiles"], _range(r, "divide")] for r in tile_rows])]
        out += _steadiness(_unsteady(tile_rows, lambda r: r["label"], ["divide"]))
        out += ["", _tile_sentence(tile_rows, rows), ""]
    return out


def _method_section(setup, checks, timing, shapes, base_rows):
    """Section 7: why the timing is done this way, backed by this run's checks."""
    names = setup["finders"]
    rows = [["Wall-clock time: time.perf_counter, through timeit [4]", "CPU time (time.process_time)",
             "CPU time adds up the work of every thread, so OpenCV's multithreaded energy map "
             "would look slower the more threads help it. Wall-clock time is what a user waits for."]]
    if checks:
        rows.append(["One untimed warm-up call before timing anything", "Timing from the first call",
                     f"In this run the first energy map at {checks['size']} took "
                     f"{checks['cold_energy_ms']:.2f} ms against {checks['warm_energy_ms']:.3f} ms "
                     f"afterwards ({_times_short(checks['cold_energy_ms'] / checks['warm_energy_ms'])}), "
                     f"and the first {checks['finder']} seam {checks['cold_finder_ms']:.2f} ms against "
                     f"{checks['warm_finder_ms']:.3f} ms "
                     f"({_times_short(checks['cold_finder_ms'] / checks['warm_finder_ms'])})."])
        single, batched = checks["single_drift"], checks["batched_drift"]
        if batched > 0 and single > 1.2 * batched:
            verdict = f"batching made the result {single / batched:.1f}x steadier"
        elif single > 0 and batched > 1.2 * single:
            verdict = "batching was less steady in this run"
        else:
            verdict = "the two were about as steady in this run"
        overhead = checks["single_us"] / checks["batched_us"] if checks["batched_us"] > 0 else 1
        rows.append([f"Batch fast calls so each sample lasts at least {1000 * MIN_SAMPLE_SECONDS:.0f} ms",
                     "One call per sample",
                     f"Five measurements of {checks['fast_finder']} on a {checks['tiny_size']} map "
                     f"each way: one call per sample read {checks['single_us']:.1f} µs and drifted "
                     f"{100 * single:.0f}% (highest minus lowest, over the median); batched read "
                     f"{checks['batched_us']:.1f} µs and drifted {100 * batched:.0f}%, so {verdict}."
                     + (f" One-call samples also time the timer's own overhead, which made them "
                        f"read {overhead:.1f}x the batched time." if overhead > 1.1 else "")
                     + f" time.perf_counter's resolution here is {checks['resolution_us']:.3f} µs."])
    rows.append(["The median of the samples", "The mean, or the minimum (which timeit's documentation "
                 "suggests [4])",
                 "One sample slowed by another program moves a mean but not a median. The minimum "
                 "is the luckiest sample, a lower bound; the median is what a call typically costs, "
                 "and the lowest and highest passes are reported next to it."])
    spread = pass_spread(timing, ["energy"] + names, lambda r: f"{r['width']} x {r['height']}")
    if spread:
        rows.append([f"{PASSES} passes that take turns over every input; median pass reported "
                     "with the range", "One pass",
                     f"Across this run's timings in section 1, the slowest pass was a median "
                     f"{spread[0]:.2f}x the fastest and up to {spread[1]:.1f}x ({spread[2]}). One "
                     "pass would have reported whichever came first, with nothing to show it was off."])
    if checks and checks["traced_ratio"]:
        rows.append(["Memory measured in a separate call", "Measuring memory during the timed calls",
                     f"tracemalloc made {checks['finder']} {checks['traced_ratio']:.1f}x slower in "
                     "this run, so timings taken while tracing would be wrong."])
    rows += [
        ["Each finder timed alone on the same energy map, as well as in full carves",
         "Full carves only",
         "A full carve mixes the finder with the energy map and seam removal, which can be most of "
         "the carve's time (section 4); timing the finder alone shows how each algorithm itself "
         "scales (sections 1 and 2)."],
        ["Full carves take turns between finders", "All runs of one finder, then the next",
         "A slow patch on the machine (another program, the CPU changing speed) then hits every "
         "finder alike, instead of only the one running at that moment."],
        ["One process, with a warm-up", "pyperf's separate worker processes [5]",
         "pyperf spreads its samples over many fresh processes, each warmed up first, so that one "
         "process's random memory layout (ASLR) and hash seed cannot bias the result. One process "
         "keeps a full run to minutes; the cost is that this run's absolute times may sit a little "
         "above or below another process's, which is why the conclusions rest on ratios measured "
         "side by side in the same process. pyperf's calibration, doubling the calls per sample "
         "until a sample is long enough, was adopted."],
        ["timeit samples", "A profiler (cProfile)",
         "A profiler adds its own overhead to every function call it records, so the times it "
         "reports are not the code's normal times; timeit measures the code as it normally runs."],
    ]
    if setup.get("fast_cores") is not None:
        rows.append(["Ask Windows not to throttle the benchmark's process", "Letting Windows decide",
                     "On a CPU with fast and efficient cores, Windows can move a background process "
                     "onto the slower efficient cores while it runs, so one pass can come out much "
                     "slower than another for no reason in the code. "
                     + ("Windows accepted the request for this run."
                        if setup["fast_cores"] else
                        "The request is only available on Windows, so here the system chose the cores.")])
    crops = []
    for picture in ("image", "photo"):
        found = {b["method"]: b for b in base_rows if b.get("picture", "image") == picture}
        if found.get("cropping") and found.get("centre crop") and found["cropping"]["energy_kept"] is not None:
            crops.append(f"on the {picture} a centre crop kept {100 * found['centre crop']['energy_kept']:.1f}% "
                         f"of the energy against the best crop's {100 * found['cropping']['energy_kept']:.1f}%")
    if crops:
        rows.append(["The best possible crop as the baseline", "A centre crop",
                     _and_list(crops)[:1].upper() + _and_list(crops)[1:]
                     + ": the best crop is the stronger rival for seam carving to beat."])
    estimate = _graphcut_full_estimate(setup, timing, shapes)
    if estimate:
        rows.append([f"Graph cut only on inputs up to {PIXEL_LIMITS['graphcut']} pixels, with a "
                     "matching small set for every finder", "Graph cut on the same inputs as the others",
                     estimate])
    out = ["## 7. How the measurements were made, and what was tried instead", "",
           "Every choice below is backed by a check made in this run where it can be measured; "
           "the others give the reason. Spread = median absolute deviation / median.", "",
           markdown_table(["Choice", "Instead of", "Why, with this run's check"], rows), ""]
    return out


def _graphcut_full_estimate(setup, timing, shapes):
    """Why graph cut gets a pixel limit: a rough time for carving the full image,
    extrapolated from its largest measured seam with section 2's slopes."""
    measured = [t for t in timing if t.get("graphcut_ms")]
    small = [r for r in shapes if r["ladder"] == "small"]
    if not measured or not small:
        return None
    hs, _, _ = _slope_value([r for r in small if r["vary"] == "height"], "height", "graphcut_ms")
    ws, _, _ = _slope_value([r for r in small if r["vary"] == "width"], "width", "graphcut_ms")
    if hs is None or ws is None:
        return None
    t = measured[-1]

    def seam_seconds(width, height):
        return t["graphcut_ms"] / 1000 * (width / t["width"]) ** ws * (height / t["height"]) ** hs

    first = seam_seconds(setup["width"], setup["height"])
    last = seam_seconds(setup["new_width"], setup["new_height"])  # the smallest image in the carve
    seams = (setup["width"] - setup["new_width"]) + (setup["height"] - setup["new_height"])
    return (f"One graph cut seam at {t['width']} x {t['height']} took {t['graphcut_ms']:.0f} ms. "
            "Extrapolating with section 2's slopes (from random maps, which grow faster than this "
            f"image's maps, so treat it as rough), one seam of the full {setup['width']} x "
            f"{setup['height']} image would take about {first:.0f} s, and even if every one of "
            f"the {seams} seams cost as little as one on the final {setup['new_width']} x "
            f"{setup['new_height']} image ({last:.1f} s), one carve would take over "
            f"{seams * last / 60:.0f} minutes. Every finder also runs on the small set, so "
            "comparisons stay like for like.")


def build_summary(setup, timing, shapes, quality, carve_rows, threads, base_rows, map_rows=(),
                  sobel_rows=(), checks=None, tile_rows=()):
    names = setup["finders"]
    finder_keys = [(n, f"{n}_ms") for n in names]
    out = ["# Seam finder benchmark", ""]

    # ----- settings, so anyone can rerun it -----
    out += [
        f"- Image: `{setup['image']}` ({setup['width']} x {setup['height']}), "
        f"carved to {setup['new_width']} x {setup['new_height']}",
        f"- Finders: {', '.join(names)}"
        + (f" (skipped, not implemented yet: {', '.join(setup['skipped'])})" if setup["skipped"] else ""),
        f"- Machine: {platform.platform()}, {platform.processor() or 'unknown CPU'}, "
        f"{os.cpu_count()} logical CPUs; Python {platform.python_version()}, "
        f"NumPy {np.__version__}, OpenCV {cv2.__version__} ({setup['cv_threads']} threads by default)"
        + ("; Windows power throttling turned off for this process" if setup.get("fast_cores") else ""),
        f"- Image widths: {', '.join(map(str, setup['sizes']))}"
        + (f"; small set for limited finders: {', '.join(map(str, setup['small_sizes']))}"
           if setup["small_sizes"] else ""),
        f"- Random maps: {setup['trials']} uniform and {setup['trials']} smooth of each shape "
        f"({', '.join(f'{w} x {h}' for h, w in setup['random_shapes'])}), seed {setup['seed']}",
        f"- Timings (sections 1, 2, 5): each is measured in {PASSES} passes that take turns over "
        f"every input. A pass is one warm-up call, then the median of {setup['repeats']} samples "
        f"of at least {1000 * MIN_SAMPLE_SECONDS:.0f} ms each, garbage collector off, following "
        "timeit [4] and pyperf [5]. Tables show the median pass with the lowest and highest in "
        "brackets.",
        f"- Full carves (sections 4, 5): one untimed warm-up run, then {setup['repeats']} runs "
        "taking turns between finders. Median, with the lowest and highest in brackets.",
        f"- Scaling slopes: fitted on the {SLOPE_POINTS} largest sizes each finder ran on, where "
        "the fixed cost of each call matters least, leaving out any timing flagged as not steady.",
    ]
    for name in names:
        if name in PIXEL_LIMITS:
            out.append(f"- {_start(name)} only runs on inputs up to {PIXEL_LIMITS[name]} pixels, so "
                       "every finder also runs on a smaller set of inputs that fits (marked small)")
    if setup.get("photo"):
        out.append(f"- Photo: `{setup['photo']}` ({setup['photo_size']}), a real photo that ships "
                   "with matplotlib (public domain), as a second test image for the experiments "
                   "whose results depend on the picture: sections 2 (energy maps), 3 and 4"
                   if setup.get("photo_bundled") else
                   f"- Photo: `{setup['photo']}` ({setup['photo_size']}), a second test image for "
                   "the experiments whose results depend on the picture: sections 2 (energy maps), "
                   "3 and 4")
    out.append("- Section 7 explains every timing choice, what was tried instead, and the checks "
               "this run made to back it")
    out += ["", "Reproduce with:", "", "```bash", setup["command"], "```", ""]
    out += conclusions(setup, timing, shapes, quality, carve_rows, threads, base_rows, map_rows,
                       sobel_rows, tile_rows)

    # ----- 1. time and memory vs size -----
    out += ["", "## 1. Time to find one seam vs image size", "",
            "Same energy map for every finder; the image is resized to each width.", ""]
    headers = ["Size", "Pixels", "Energy map (ms)", "Energy map (ns per pixel)"] + [f"{n} (ms)" for n in names]
    table = [[f"{r['width']} x {r['height']}", r["pixels"], _range(r, "energy"),
              f"{1e6 * r['energy_ms'] / r['pixels']:.1f}"]
             + [_range(r, n) for n in names] for r in timing]
    out.append(markdown_table(headers, table))
    out += _steadiness(_unsteady(timing, lambda r: f"{r['width']} x {r['height']}",
                                 ["energy"] + names))
    energy_bytes = 1024 * timing[-1]["energy_peak_kb"] / timing[-1]["pixels"]
    out += ["", "Log-log slope of time vs pixels: "
            + _slopes(timing, "pixels", [("energy map", "energy_ms")] + finder_keys) + ". "
            "About 1 means time grows like H x W; about 0.5 means like H + W "
            "(the aspect ratio is fixed)"
            + ("; above 1 means faster than H x W, as for graph cut" if "graphcut" in names else "")
            + ". Section 2 shows why DP can sit below 1."
            + (" Graph cut's time also depends on the values in the map, not only its size, so "
               "its slope here holds for this image's maps: see \"Same sizes, different energy "
               "maps\" in section 2." if map_rows and "graphcut" in names else ""),
            "", "The energy map does the same work for every pixel, so it is O(H x W), yet its time "
            "per pixel is not constant here (see the ns per pixel column). It makes several new "
            f"arrays the size of the image on every call, about {energy_bytes:.0f} bytes of working "
            "memory per pixel (memory table below)."
            + _energy_memory_note(sobel_rows),
            "", "![Finder timing](finder_timing.png)"]

    out += ["", "### Peak extra memory for one seam", "",
            "Measured with tracemalloc [6] in a separate call (tracing slows code down).", ""]
    def per_pixel(r, n):
        return 1024 * r[f"{n}_peak_kb"] / r["pixels"]

    memory_names = names + ["energy"]
    table = [[f"{r['width']} x {r['height']}", r["pixels"]]
             + [("skipped" if r[f"{n}_peak_kb"] is None
                 else f"{r[f'{n}_peak_kb']:.1f} ({per_pixel(r, n):.2f})") for n in memory_names]
             for r in timing]
    out.append(markdown_table(["Size", "Pixels"] + [f"{n}, KB (bytes per pixel)" for n in names]
                              + ["energy map, KB (bytes per pixel)"], table))
    trends = []
    for n in names:
        measured = [r for r in timing if r[f"{n}_peak_kb"] is not None]
        if len(measured) >= 2:
            first, last = measured[0], measured[-1]
            trends.append(f"{n} {per_pixel(first, n):.2f} at {first['width']} x {first['height']} "
                          f"to {per_pixel(last, n):.2f} at {last['width']} x {last['height']}")
    out += ["", "Log-log slope of memory vs pixels: "
            + _slopes(timing, "pixels", [(n, f"{n}_peak_kb") for n in names]) + ". "
            "Bytes per pixel from the smallest to the largest size: " + "; ".join(trends) + ".",
            "", "Expected: DP keeps one int8 backpointer per pixel (1 byte, so O(H x W)), plus "
            "working arrays that grow with the width or the height alone: a few rows of floats "
            "for the row being filled, and the seam itself. Those are a big share of a small image "
            "and a small share of a large one, so DP's bytes per pixel fall towards 1 as the image "
            "grows, and its fitted slope sits below 1 at these sizes. Keeping the full float64 "
            "cost table as well would add 8 bytes per pixel. Greedy only stores the seam, O(H). "
            "With a fixed aspect ratio, H grows like the square root of the pixels, so its slope "
            "should be about 0.5."
            + (" Graph cut builds its whole graph before it starts, a Python dict of edges for "
               "every pixel, so it is O(H x W) too, with a slope of about 1 but hundreds of bytes "
               "per pixel." if "graphcut" in names else ""),
            "", "![Peak memory for one seam](memory.png)"]

    # ----- 2. height vs width -----
    out += ["", "## 2. Height vs width", ""]
    for ladder, base, _steps in setup["shape_ladders"]:
        part = [r for r in shapes if r["ladder"] == ladder]
        picture = "shape_scaling.png" if ladder == "main" else "shape_scaling_small.png"
        label = "" if ladder == "main" else " (small, for limited finders)"
        out += [f"Random energy maps{label}. One side stays at {base} while the other grows.", ""]
        for vary in ("height", "width"):
            rows = [r for r in part if r["vary"] == vary]
            fixed = "width" if vary == "height" else "height"
            table = [[f"{r['width']} x {r['height']}"] + [_range(r, n) for n in names] for r in rows]
            out += [f"Only the {vary} grows ({fixed} = {base}):", "",
                    markdown_table(["Size (W x H)"] + [f"{n} (ms)" for n in names], table)]
            out += _steadiness(_unsteady(rows, lambda r: f"{r['width']} x {r['height']}", names))
            out += ["", f"Log-log slope vs {vary}: {_slopes(rows, vary, finder_keys)}.", ""]
        out += [f"![Height vs width{label}]({picture})", ""]
    out += ["A slope of 1 means time is proportional to that side. A slope below 1 means "
            "part of the time does not depend on that side: for DP, the fixed cost of the "
            "NumPy calls it makes for every row"
            + ("; for greedy, its scan along the whole top row when only the height grows, and "
               "its walk down every row when only the width grows" if "greedy" in names else "")
            + "."
            + (" A slope above 1 means the time grows faster than that side, as for graph cut: "
               "every extra row or column adds nodes and edges that each search for an augmenting "
               "path may walk through, and a bigger graph usually needs more of those searches "
               "(counted below)." if "graphcut" in names else "")]

    # ----- 2b. same sizes, different energy maps -----
    if map_rows:
        counted = any(r["graphcut_searches"] is not None for r in map_rows)
        edges = any(r["greedy_edge"] is not None for r in map_rows)
        has_photo = any(r["map"] == "photo" for r in map_rows)
        out += ["", "### Same sizes, different energy maps", "",
                "At each small size: the image's own energy map (as in section 1), "
                + ("the photo's map (centre-cropped to the same shape), " if has_photo else "")
                + "one uniform random map and one smooth random map (as in section 3), all the "
                "same shape. "
                f"Near zero = share of pixels below {100 * NEAR_ZERO:.0f}% of the map's largest "
                "energy."
                + (" Searches = breadth-first searches graph cut makes for one seam, about one per "
                   "augmenting path." if counted else "")
                + (" Greedy on edge = share of rows where greedy's seam is in the first or last "
                   "column." if edges else ""), ""]
        table = [[f"{r['width']} x {r['height']}", r["map"], f"{100 * r['near_zero']:.1f}%"]
                 + [_range(r, n) for n in names]
                 + ([r["graphcut_searches"] if r["graphcut_searches"] is not None else "-"]
                    if counted else [])
                 + ([f"{100 * r['greedy_edge']:.0f}%" if r["greedy_edge"] is not None else "-"]
                    if edges else []) for r in map_rows]
        out.append(markdown_table(["Size", "Map", "Near zero"] + [f"{n} (ms)" for n in names]
                                  + (["graphcut searches"] if counted else [])
                                  + (["greedy on edge"] if edges else []), table))
        out += _steadiness(_unsteady(map_rows, lambda r: f"{r['width']} x {r['height']} ({r['map']})",
                                     names))
        out += ["", "Log-log slope of time vs pixels: " + "; ".join(
            f"{kind} maps: {_slopes([r for r in map_rows if r['map'] == kind], 'pixels', finder_keys)}"
            for kind in ALL_MAP_KINDS if any(r["map"] == kind for r in map_rows)) + "."]
        effects = {n: _map_effect(map_rows, n) for n in names}
        effects = {n: e for n, e in effects.items() if e}
        if effects:
            size = next(iter(effects.values()))[0]
            others = [k for k in ALL_MAP_KINDS[1:] if all(k in e[1] for e in effects.values())]
            named = {"photo": "the photo's map", "random": "a uniform random map",
                     "smooth": "a smooth random map"}
            out += ["", f"At {size['width']} x {size['height']}, each finder's time on "
                    f"{_and_list(named[k] for k in others)}, as a multiple of its time on the "
                    "image's own map: "
                    + "; ".join(f"{n} " + _and_list(f"{e[1][k]:.2f}x" for k in others)
                                for n, e in effects.items()) + "."]
        steady = [n for n in names if n != "graphcut" and _map_independent(map_rows, n)]
        if steady:
            out[-1] += (f" {_and_list(_start(n) for n in steady)} {'does' if len(steady) == 1 else 'do'} the same "
                        "steps whatever the values, so the map makes little difference to "
                        f"{'its' if len(steady) == 1 else 'their'} time.")
        note = _greedy_edge_note(map_rows) if "greedy" in names and edges else None
        if note and not _map_independent(map_rows, "greedy"):
            out[-1] += (f" Greedy's time moves a little because of where its seam goes: {note}. "
                        "Its work is O(H + W) on every map; only the constant changes.")
        if "graphcut" in effects and counted:
            image_row, ratio, searches = effects["graphcut"]
            near = {r["map"]: r["near_zero"] for r in map_rows if r["width"] == image_row["width"]}
            text = (f"Graph cut needed {searches['image']} searches on the image's map, "
                    f"{searches['random']} on the uniform one and {searches['smooth']} on the "
                    "smooth one: its time follows the number of augmenting paths, and each search "
                    "walks the graph from the source. The image has large flat areas "
                    f"({100 * near['image']:.0f}% of its pixels are near zero, against "
                    f"{100 * near['random']:.0f}% and {100 * near['smooth']:.0f}% on the random "
                    "maps), and a near-zero pixel's edge to its right neighbour can carry almost "
                    "no flow. That most likely leaves fewer augmenting paths to find; it was not "
                    "tested on its own, and random maps of the same kind can need quite different "
                    "numbers of searches.")
            if "photo" in searches and searches["photo"]:
                text += (f" The photo sits in between: {searches['photo']} searches, "
                         f"{100 * near['photo']:.0f}% of its pixels near zero, and "
                         f"{ratio['photo']:.1f}x the image's time, so the photo is harder for "
                         "graph cut than this image but easier than random noise."
                         if searches["random"] > searches["photo"] > searches["image"] else
                         f" On the photo's map it needed {searches['photo']} searches, with "
                         f"{100 * near['photo']:.0f}% of its pixels near zero, and took "
                         f"{ratio['photo']:.1f}x the image's time.")
            s_img, _, _ = _slope_value([r for r in map_rows if r["map"] == "image"], "pixels", "graphcut_ms")
            s_rnd, _, _ = _slope_value([r for r in map_rows if r["map"] == "random"], "pixels", "graphcut_ms")
            if s_img is not None and s_rnd is not None and s_img < s_rnd:
                text += (f" Graph cut's slope against pixels here is {s_img:.2f} on the image's "
                         f"maps and {s_rnd:.2f} on uniform random ones, which is why its slope in "
                         "section 1 (the image) is lower than its slopes on the random maps above.")
                s_one, _, _ = _slope_value(timing, "pixels", "graphcut_ms")
                if s_one is not None and abs(s_one - s_img) >= 0.05:
                    def sizes(rows):
                        return _and_list(f"{r['width']} x {r['height']}"
                                         for r in _fit_rows(rows, "pixels", "graphcut_ms"))
                    image_maps = [r for r in map_rows if r["map"] == "image"]
                    text += (f" Section 1 gives {s_one:.2f} rather than {s_img:.2f} for the same "
                             f"image because each slope is fitted on its own {SLOPE_POINTS} "
                             f"largest sizes: {sizes(timing)} there, {sizes(image_maps)} here. "
                             "Over so few small sizes, the fitted slope moves with the sizes "
                             "chosen.")
            out += ["", text]
        out += ["", "![Same sizes, different energy maps](map_dependence.png)"]

    # ----- 3. seam quality -----
    out += ["", "## 3. Seam quality on the same energy map", "",
            "Ratio = seam cost / DP seam cost on the same map. DP finds the cheapest seam [1, 3], "
            "so 1.000 is the best possible. Random maps are uniform in [0, 1). Smooth maps are "
            f"uniform noise blurred with a Gaussian ({SMOOTH_SIGMA} px) and stretched back to "
            "[0, 1], so neighbouring pixels are alike, as in real images.", ""]
    groups = {}
    for r in quality:
        groups.setdefault((r["kind"], r["algo"]), []).append(r)
    table = []
    for (kind, algo), group in groups.items():
        ratios = [r["ratio"] for r in group]
        table.append([
            kind, algo, len(group),
            f"{np.mean(ratios):.3f}", f"{max(ratios):.3f}",
            f"{100 * np.mean([r['optimal'] for r in group]):.0f}%",
            f"{100 * np.mean([r['rows_same'] for r in group]):.0f}%",
        ])
    out.append(markdown_table(
        ["Energy maps", "Finder", "Maps", "Mean ratio", "Worst ratio",
         "Found the optimum", "Rows on the DP seam"], table))
    if setup["quality_chart"]:
        out += ["", "![Seam quality](seam_quality.png)"]

    # ----- 4. full carve -----
    out += ["", "## 4. Full carve", ""]
    picture_names = {"image": f"Image `{setup['image']}`", "photo": f"Photo `{setup.get('photo')}`"}
    carve_groups = list(dict.fromkeys((r.get("picture", "image"), r["input"]) for r in carve_rows))
    compare_files = []
    for picture, carve_input in carve_groups:
        rows = [r for r in carve_rows
                if r.get("picture", "image") == picture and r["input"] == carve_input]
        full = carve_input == next(i for p, i in carve_groups if p == picture)
        tag = ("" if picture == "image" else "_photo") + ("" if full else "_small")
        compare_files.append((f"compare{tag}.png",
                              f"Seams and results, {picture}{'' if full else ', small copy'}"))
        ref = next((r for r in rows if r["algo"] == REFERENCE), None)
        table = []
        for r in rows:
            table.append([
                r["algo"], r["seams"], _range(r, "total", "s"), _range(r, "finder", "s"),
                f"{100 * r['finder_share']:.0f}%",
                f"{r['outside_per_seam_ms']:.3f}" if r["outside_per_seam_ms"] is not None else "-",
                f"{r['energy_removed']:.0f}",
                f"{r['energy_removed'] / ref['energy_removed']:.3f}"
                if ref and ref["energy_removed"] else "-",
                f"{r['overlap_with_dp']:.2f}" if r["overlap_with_dp"] is not None else "-",
                f"{r['crop_likeness']:.2f}" if r.get("crop_likeness") is not None else "-",
            ])
        out += [f"{picture_names[picture]}{'' if full else ', small copy'}, {carve_input}:", "",
                markdown_table(
            ["Finder", "Seams", "Total (s)", "In finder (s)", "Finder share",
             "Outside finder per seam (ms)", "Energy removed", "Energy vs DP",
             "Pixel overlap with DP", "Like a crop"], table)]
        out += _steadiness(_unsteady(rows, lambda r: r["algo"], ["total", "finder"], "s"))
        out += [""]
        if ref:
            for r in rows:
                if r is ref or r["finder_s"] <= 0 or r["total_s"] <= 0 or ref["finder_s"] <= 0:
                    continue
                out += [f"- {_carve_speed(_start(r['algo']), r, ref)}.", ""]
    out += ["Every seam needs a new energy map and a seam removal, both O(H x W). So carving k "
            "seams costs k x (O(H x W) + the finder's own cost). For DP (O(H x W)) and greedy "
            "(O(H + W)) that is O(k x H x W) either way: a faster finder only lowers the constant."
            + (" Graph cut's own cost grows much faster than H x W (section 2), so for graph cut "
               "the finder is almost the whole carve." if "graphcut" in names else ""),
            "",
            "Energy removed = sum of each seam's energy when it was removed. DP is optimal for "
            "each single seam, but after the first seam the runs are carving different images, "
            "so a lower total for DP is likely, not guaranteed. Pixel overlap = pixels removed "
            "by both / pixels removed by either. Like a crop = the same overlap against the "
            "crop of the same size that matches the carve best: 1 would mean the carve removed "
            "exactly a crop's pixels, a strip off the edges. Outside finder per seam = (total - in "
            "finder) / seams: the energy map and seam removal for one seam, timed inside the "
            "real loop.",
            "", "![Where a full carve spends its time](time_breakdown.png)"]
    for file, label in compare_files:
        out += ["", f"![{label}]({file})"]

    # ----- 4b. scaling and cropping -----
    table = []
    baseline_files = []
    for picture, carve_input in carve_groups:
        if carve_input != next(i for p, i in carve_groups if p == picture):
            continue  # baselines are for each picture's full-size carve
        baseline_files.append("baselines.png" if picture == "image" else "baselines_photo.png")
        for r in base_rows:
            if r.get("picture", "image") == picture:
                table.append([picture, f"{r['method']}{' (best)' if r['method'] == 'cropping' else ''}",
                              _range(r, "time"),
                              f"{100 * r['energy_kept']:.1f}%" if r["energy_kept"] is not None
                              else "does not apply"])
        for r in carve_rows:
            if r.get("picture", "image") == picture and r["input"] == carve_input:
                times = {"time_ms": 1000 * r["total_s"], "time_min_ms": 1000 * r["total_min_s"],
                         "time_max_ms": 1000 * r["total_max_s"]}
                kept = f"{100 * r['energy_kept']:.1f}%" if r["energy_kept"] is not None else "-"
                table.append([picture, f"seam carving ({r['algo']})", _range(times, "time"), kept])
    full_inputs = "; ".join(f"{p} {i}" for p, i in carve_groups
                            if i == next(j for q, j in carve_groups if q == p))
    out += ["", "### Compared with scaling and cropping", "",
            f"Inputs: {full_inputs}. The two simple ways to resize, as in the brief's figure [1]. "
            "Cropping (best) keeps the window with the most original energy, the best possible "
            "crop; a centre crop, which cuts the same amount off each side, is shown for "
            "comparison.", "",
            markdown_table(["Image", "Method", "Time (ms)", "Original energy kept"], table), "",
            "Original energy kept = share of the original image's Sobel energy whose pixels are "
            "still in the result; higher means more detail kept. No finder optimises it directly, "
            "because each seam is chosen on the already carved image. It does not apply to "
            "scaling, which blends pixels instead of keeping some. It also ignores context: a "
            "crop can keep plenty of energy yet cut part of the scene off, so judge the pictures too."]
    for file in baseline_files:
        out += ["", f"![Scaling, cropping and seam carving]({file})"]

    # ----- 5. multithreading -----
    out += ["", "## 5. Does multithreading help?", "",
            f"OpenCV on 1 thread vs its default {setup['cv_threads']} threads, set with "
            "cv2.setNumThreads [7]. Median, with the "
            f"lowest and highest in brackets: over {PASSES} passes for the energy map, over "
            f"{setup['repeats']} runs for full carves. The full-carve rows are the real test. "
            "The energy-map rows time it back to back on its own, which can be faster than "
            "inside the carve loop: compare section 1, where energy maps take turns with seam "
            "finding, and the time outside the finder per seam in section 4.", ""]
    table = [[r["part"], _range(r, "one_thread"), _range(r, "default_threads"),
              f"{r['speedup']:.2f}x"] for r in threads]
    out.append(markdown_table(
        ["Part", "1 thread (ms)", f"{setup['cv_threads']} threads (ms)", "Speed-up"], table))
    out += _steadiness(_unsteady(threads, lambda r: r["part"], ["one_thread", "default_threads"]))
    out += ["", "![Multithreading speed-up](threading.png)"]
    out += ["", "Only OpenCV (the Sobel energy map) can use several threads here. NumPy runs DP's "
            "element-wise maths on one thread, and DP's rows cannot be shared out between threads "
            "because each row needs the finished row above. Splitting one row's W cells between "
            "threads would need the threads to wait for each other on every row, while a row takes "
            "only tens of microseconds (section 2), so the overhead would likely cancel the gain. "
            "A row-split DP was not measured.", ""]

    out += _sobel_section(sobel_rows, tile_rows)
    out += _method_section(setup, checks, timing, shapes, base_rows)

    out += ["## References", ""]
    out += [f"{number}. {reference}" for number, reference in enumerate(REFERENCES, start=1)]
    out += [""]
    return "\n".join(out)


# ---------- command line ----------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark the seam finders against DP")
    parser.add_argument("--image", default=str(REPO_ROOT / "images" / "sample.png"),
                        help="input image (default: images/sample.png)")
    parser.add_argument("--algos", nargs="+", choices=list(FINDERS), default=DEFAULT_ALGOS,
                        help="finders to compare (default: dp greedy graphcut)")
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES,
                        help="image widths for experiments 1, 3 and 5")
    parser.add_argument("--width", type=int, help="carve target width (default: 75%% of the image)")
    parser.add_argument("--height", type=int, help="carve target height (default: 80%% of the image)")
    parser.add_argument("--repeats", type=int, default=5, help="timed samples or runs per measurement")
    parser.add_argument("--trials", type=int, default=100, help="random energy maps of each shape in experiment 3")
    parser.add_argument("--seed", type=int, default=0, help="seed for the random energy maps")
    parser.add_argument("--photo", help="a photo as the second test image (default: matplotlib's "
                        "sample photo, grace_hopper.jpg; 'none' to leave it out)")
    parser.add_argument("--output", default="out/bench", help="folder for the results")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.trials < 0:
        parser.error("--trials cannot be negative")
    if min(args.sizes) < 3:
        parser.error("--sizes must all be at least 3")
    return args


def display_path(path):
    """Path relative to the repo, or just the file name, so summary.md never
    carries anyone's home folder into the report."""
    path = Path(path).resolve()
    return path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else path.name


def main(argv=None):
    args = parse_args(argv)

    try:
        img = load_image(args.image)
    except FileNotFoundError as err:
        raise SystemExit(str(err))
    height, width = img.shape[:2]
    new_width = args.width or round(width * 0.75)
    new_height = args.height or round(height * 0.8)
    if not (2 <= new_width <= width and 2 <= new_height <= height):
        raise SystemExit(f"Target {new_width} x {new_height} must be between 2 x 2 "
                         f"and the image size {width} x {height}.")
    if (new_width, new_height) == (width, height):
        raise SystemExit("Target is the same size as the image, so there is nothing to carve.")

    finders, skipped = available_finders(args.algos)
    for name in skipped:
        print(f"Skipping {name}: not implemented yet")
    if not finders:
        raise SystemExit("No finished seam finders to benchmark.")
    names = list(finders)

    # Limited finders (graph cut) also get a set of image widths that fit their limit.
    limit = smallest_limit(names)
    small = []
    if limit is not None and any(w * resized_height(img, w) > limit for w in args.sizes):
        small = [w for w in small_widths(img, limit) if w not in args.sizes]

    # A second test image, a photo, for the experiments whose results depend
    # on the picture, so the conclusions do not rest on one drawing.
    photo = photo_path = None
    if (args.photo or "").lower() != "none":
        photo_path = Path(args.photo) if args.photo else bundled_photo()
        if photo_path is not None:
            try:
                photo = load_image(photo_path)
            except FileNotFoundError as err:
                raise SystemExit(str(err))

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Keep Windows from moving the process to slower cores mid-run (see prefer_fast_cores).
    fast_cores = prefer_fast_cores()

    # First, while the process is still cold: the checks behind the timing method.
    print("1/7 Checking the timing method...")
    checks = method_checks(img, finders, args.repeats)

    print("2/7 Timing one seam at each image size...")
    timing = finder_timing(img, finders, sorted(set(args.sizes) | set(small)), args.repeats)
    write_csv(timing, out / "finder_timing.csv")
    plot_finder_timing(timing, names, out / "finder_timing.png")
    plot_memory(timing, names, out / "memory.png")

    print("3/7 Timing height vs width...")
    shapes = shape_scaling(finders, args.repeats, args.seed)
    write_csv(shapes, out / "shape_scaling.csv")
    for ladder, base, _steps in shape_ladders(names):
        picture = "shape_scaling.png" if ladder == "main" else "shape_scaling_small.png"
        plot_shape_scaling([r for r in shapes if r["ladder"] == ladder], names, base, out / picture)
    map_rows = []
    if limit is not None:
        # Graph cut's time depends on the values in the map, so the image's own
        # maps (section 1) and random maps (above) are compared at the same sizes.
        print("3/7 Timing the same sizes on different energy maps...")
        map_rows = map_dependence(img, finders, small_widths(img, limit), args.repeats, args.seed,
                                  photo=photo)
        write_csv(map_rows, out / "map_dependence.csv")
        plot_map_dependence(map_rows, names, out / "map_dependence.png")

    print("4/7 Comparing seams with DP on the same energy maps...")
    photo_small = small_widths(photo, limit) if photo is not None and limit is not None else ()
    quality = seam_quality(img, finders, args.sizes, args.trials, args.seed, small,
                           photo=photo, photo_small=photo_small)
    write_csv(quality, out / "seam_quality.csv")
    quality_chart = plot_seam_quality(quality, out / "seam_quality.png")

    # The image, then the photo at the same width (never enlarged), carved by
    # the same share of its width and height.
    to_carve = [("image", img, new_width, new_height)]
    if photo is not None:
        photo_img = resize_to_width(photo, width) if photo.shape[1] > width else photo
        photo_h, photo_w = photo_img.shape[:2]
        to_carve.append(("photo", photo_img, max(2, round(photo_w * new_width / width)),
                         max(2, round(photo_h * new_height / height))))
    carve_rows, base_rows = [], []
    for picture_kind, source, source_w, source_h in to_carve:
        for label, image, target_w, target_h in carve_inputs(source, names, source_w, source_h):
            image_h, image_w = image.shape[:2]
            print(f"5/7 Carving the {picture_kind}, {image_w} x {image_h} -> {target_w} x {target_h}...")
            rows, pictures = carve_comparison(image, finders, target_w, target_h, args.repeats)
            carve_rows += [{"picture": picture_kind,
                            "input": f"{image_w} x {image_h} -> {target_w} x {target_h}", **row}
                           for row in rows]
            tag = ("" if picture_kind == "image" else "_photo") + ("" if label == "full" else "_small")
            for name, picture in pictures.items():
                save_image(picture["result"], out / f"carve{tag}_{name}.png")
                save_image(picture["seams"], out / f"seams{tag}_{name}.png")
            if pictures:
                plot_carve_comparison(image, pictures, out / f"compare{tag}.png")
            if label == "full":
                # Scaling and cropping on the same input, for the brief's three-way comparison.
                rows_b, base_pictures = baselines(image, target_w, target_h, args.repeats)
                base_rows += [{"picture": picture_kind, **row} for row in rows_b]
                panels = ([("original", image)] + list(base_pictures.items())
                          + [(f"seam carving, {name}", p["result"]) for name, p in pictures.items()])
                plot_baselines(image, panels, out / f"baselines{tag}.png")
    write_csv(base_rows, out / "baselines.csv")
    write_csv(carve_rows, out / "carve.csv")
    if carve_rows:
        plot_time_breakdown(carve_rows, out / "time_breakdown.png")

    print("6/7 OpenCV on 1 thread vs its default...")
    cv_threads = cv2.getNumThreads()
    threads = threading_check(img, finders, args.sizes, new_width, new_height, args.repeats)
    write_csv(threads, out / "threading.csv")
    plot_threading(threads, cv_threads, out / "threading.png")

    print("7/7 Item 2b: the Sobel filter five ways...")
    sobel_rows = sobel_methods(img, args.sizes, args.repeats)
    write_csv(sobel_rows, out / "sobel_methods.csv")
    plot_sobel_methods(sobel_rows, out / "sobel_methods.png")
    tile_rows = sobel_tile_sizes(img, max(args.sizes), args.repeats)
    write_csv(tile_rows, out / "sobel_tiles.csv")

    image = display_path(args.image)
    photo_flag = ("" if args.photo is None else
                  " --photo none" if photo is None else f" --photo {display_path(photo_path)}")
    command = (f"python benchmarks/run_bench.py --image {image} --algos {' '.join(args.algos)} "
               f"--sizes {' '.join(map(str, args.sizes))} --width {new_width} --height {new_height} "
               f"--repeats {args.repeats} --trials {args.trials} --seed {args.seed}{photo_flag}")
    setup = {"image": image, "width": width, "height": height,
             "new_width": new_width, "new_height": new_height,
             "finders": names, "skipped": skipped, "repeats": args.repeats,
             "sizes": args.sizes, "small_sizes": small, "trials": args.trials, "seed": args.seed,
             "random_shapes": random_shapes(names), "shape_ladders": shape_ladders(names),
             "cv_threads": cv_threads, "command": command, "quality_chart": quality_chart,
             "fast_cores": fast_cores,
             "photo": display_path(photo_path) if photo is not None else None,
             "photo_bundled": args.photo is None,
             "photo_size": f"{photo.shape[1]} x {photo.shape[0]}" if photo is not None else None}
    summary = build_summary(setup, timing, shapes, quality, carve_rows, threads, base_rows,
                            map_rows, sobel_rows, checks, tile_rows)
    (out / "summary.md").write_text(summary, encoding="utf-8")
    # Some Windows consoles cannot show "–"; print a stand-in instead of crashing.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    print(f"\n{summary}\nResults saved in {out.as_posix()}")


if __name__ == "__main__":
    main()
