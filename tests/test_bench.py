"""Tests for benchmarks/run_bench.py (P3)."""

import csv
import re
from pathlib import Path

import numpy as np
import pytest

from benchmarks import run_bench
from seam_carving.core import carve
from seam_carving.finders import get_finder

SMALL = Path(__file__).parent.parent / "images" / "small.png"


@pytest.fixture(autouse=True)
def fast_settings(monkeypatch):
    # Short timing samples and small shapes keep these tests quick.
    monkeypatch.setattr(run_bench, "MIN_SAMPLE_SECONDS", 0.001)
    monkeypatch.setattr(run_bench, "SHAPE_STEPS", [20, 40])
    monkeypatch.setattr(run_bench, "SHAPE_BASE", 30)


def random_image(height, width, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def dp_and_greedy():
    return {"dp": get_finder("dp"), "greedy": get_finder("greedy")}


def test_available_finders_skips_unfinished():
    def lookup(name):
        if name == "todo":
            def unfinished(energy):
                raise NotImplementedError
            return unfinished
        return get_finder(name)

    finders, skipped = run_bench.available_finders(["dp", "todo", "greedy"], lookup)
    assert list(finders) == ["dp", "greedy"]
    assert skipped == ["todo"]


def test_too_big_uses_pixel_limits(monkeypatch):
    monkeypatch.setitem(run_bench.PIXEL_LIMITS, "greedy", 100)
    assert run_bench.too_big("greedy", 10, 11)
    assert not run_bench.too_big("greedy", 10, 10)
    assert not run_bench.too_big("dp", 1000, 1000)  # no limit set


def test_resize_keeps_aspect_ratio():
    assert run_bench.resize_to_width(random_image(40, 60), 30).shape == (20, 30, 3)
    assert run_bench.resize_to_width(random_image(40, 60), 120).shape == (80, 120, 3)


def test_compare_on_greedy_counterexample():
    # Same map as tests/test_greedy.py: greedy's seam costs 17, DP's costs 4.
    energy = np.array([
        [1, 2, 5],
        [8, 9, 1],
        [9, 8, 1],
    ], dtype=float)
    stats = run_bench.compare_to_reference(energy, np.array([0, 0, 1]), np.array([1, 2, 2]))
    assert stats["cost"] == 17
    assert stats["dp_cost"] == 4
    assert stats["ratio"] == pytest.approx(17 / 4)
    assert not stats["optimal"]
    assert stats["rows_same"] == 0


def test_compare_handles_zero_cost_dp_seam():
    energy = np.array([[0, 1], [0, 1]], dtype=float)
    flat = np.array([0, 0])
    assert run_bench.compare_to_reference(energy, flat, flat)["ratio"] == 1.0
    assert run_bench.compare_to_reference(energy, np.array([1, 1]), flat)["ratio"] == float("inf")


def test_overlap():
    a = np.array([[True, False], [True, False]])
    b = np.array([[True, True], [False, False]])
    assert run_bench.overlap(a, b) == pytest.approx(1 / 3)
    assert run_bench.overlap(a, a) == 1.0
    assert run_bench.overlap(np.zeros((2, 2), bool), np.zeros((2, 2), bool)) == 1.0


def test_loglog_slope():
    pixels = [100, 1000, 10000]
    assert run_bench.loglog_slope(pixels, [1, 10, 100]) == pytest.approx(1.0)
    assert run_bench.loglog_slope(pixels, [1, 10 ** 0.5, 10]) == pytest.approx(0.5)
    assert run_bench.loglog_slope([100], [1]) is None  # needs two points


def test_loglog_slope_ignores_the_smallest_sizes():
    # Flat at the small end (fixed cost per call), then proportional:
    # only the 3 largest sizes are fitted, so the slope is 1.
    sizes = [1, 10, 100, 1000]
    assert run_bench.loglog_slope(sizes, [1, 1, 10, 100]) == pytest.approx(1.0)
    assert run_bench.loglog_slope(sizes, [1, 1, 10, 100], points=4) < 0.9


def test_fit_width_is_the_largest_width_under_the_limit():
    img = random_image(160, 240)
    width = run_bench.fit_width(img, 3600)
    assert width * run_bench.resized_height(img, width) <= 3600
    assert (width + 1) * run_bench.resized_height(img, width + 1) > 3600
    widths = run_bench.small_widths(img, 3600)
    assert widths[0] == 10 and widths[-1] == width
    assert widths == sorted(set(widths))


def test_small_sets_only_appear_for_limited_finders(monkeypatch):
    names = ["dp", "greedy"]
    assert run_bench.random_shapes(names) == [run_bench.RANDOM_SHAPE]
    assert [ladder for ladder, _, _ in run_bench.shape_ladders(names)] == ["main"]

    monkeypatch.setitem(run_bench.PIXEL_LIMITS, "greedy", 600)
    small_shape = run_bench.random_shapes(names)[1]
    assert small_shape[0] * small_shape[1] <= 600
    ladder, base, steps = run_bench.shape_ladders(names)[1]
    assert ladder == "small" and base * max(steps) <= 600


def test_carve_inputs_add_a_small_copy_for_limited_finders(monkeypatch):
    img = random_image(40, 50)
    assert [label for label, *_ in run_bench.carve_inputs(img, ["dp"], 38, 32)] == ["full"]

    monkeypatch.setitem(run_bench.PIXEL_LIMITS, "greedy", 600)
    inputs = run_bench.carve_inputs(img, ["dp", "greedy"], 38, 32)
    label, small, target_w, target_h = inputs[1]
    assert label == "small"
    assert small.shape[0] * small.shape[1] <= 600
    assert target_w < small.shape[1] and target_h < small.shape[0]


def test_smooth_random_maps_are_in_range_and_smoother_than_noise():
    rng = np.random.default_rng(0)
    smooth = run_bench.smooth_random(rng, (60, 80))
    noise = rng.random((60, 80))
    assert smooth.min() == 0 and smooth.max() == 1
    # Neighbouring pixels differ far less than in plain noise.
    assert np.abs(np.diff(smooth, axis=1)).mean() < np.abs(np.diff(noise, axis=1)).mean() / 3


def test_best_crop_matches_brute_force():
    rng = np.random.default_rng(3)
    for _ in range(20):
        energy = rng.random((rng.integers(2, 12), rng.integers(2, 12)))
        height, width = energy.shape
        new_h, new_w = rng.integers(1, height + 1), rng.integers(1, width + 1)
        top, left = run_bench.best_crop(energy, new_h, new_w)
        best = max(energy[t:t + new_h, l:l + new_w].sum()
                   for t in range(height - new_h + 1) for l in range(width - new_w + 1))
        assert energy[top:top + new_h, left:left + new_w].sum() == pytest.approx(best)


def test_baselines_scale_and_crop_to_the_target():
    img = random_image(20, 30)
    rows, pictures = run_bench.baselines(img, 24, 16, repeats=1)
    assert [r["method"] for r in rows] == ["scaling", "cropping", "centre crop"]
    assert pictures["scaling"].shape == pictures["cropping"].shape == (16, 24, 3)
    assert rows[0]["energy_kept"] is None  # scaling blends pixels
    assert 0 < rows[1]["energy_kept"] < 1
    assert all(r["time_min_ms"] <= r["time_ms"] <= r["time_max_ms"] for r in rows)


def test_peak_memory_sees_dps_backpointer_table():
    energy = np.random.default_rng(0).random((100, 200))
    dp_bytes = run_bench.peak_memory(lambda: get_finder("dp")(energy))
    greedy_bytes = run_bench.peak_memory(lambda: get_finder("greedy")(energy))
    assert dp_bytes >= 100 * 200  # one int8 backpointer per pixel
    assert greedy_bytes < dp_bytes / 4  # greedy only keeps the seam


def test_measure_passes_reports_median_and_range():
    calls = []
    measured = run_bench.measure_passes([("a", lambda: calls.append("a")),
                                         ("b", lambda: calls.append("b"))], repeats=2)
    for median, low, high in measured.values():
        assert 0 < low <= median <= high
    # Passes take turns: "a" and "b" each run in every pass, not all of "a" first.
    a_runs = [i for i, name in enumerate(calls) if name == "a"]
    b_runs = [i for i, name in enumerate(calls) if name == "b"]
    assert a_runs[-1] > b_runs[0]


def test_unsteady_flags_timings_with_a_wide_range():
    rows = [{"size": 1, "dp_ms": 1.0, "dp_min_ms": 1.0, "dp_max_ms": 1.2},
            {"size": 2, "dp_ms": 1.0, "dp_min_ms": 0.5, "dp_max_ms": 1.0},
            {"size": 3, "dp_ms": None, "dp_min_ms": None, "dp_max_ms": None}]
    assert run_bench._unsteady(rows, lambda r: r["size"], ["dp"]) == ["dp at 2 (2.0x)"]


def test_slopes_leave_out_unsteady_points():
    def row(pixels, ms, low, high):
        return {"pixels": pixels, "x_ms": ms, "x_min_ms": low, "x_max_ms": high}

    # Proportional to pixels, except one wild point at 1000 pixels whose
    # passes ranged from 1 to 50 ms: it must not drag the slope away from 1.
    rows = [row(10, 1, 1, 1.1), row(100, 10, 10, 11), row(1000, 50, 1, 50), row(10000, 1000, 990, 1010)]
    text = run_bench._slopes(rows, "pixels", [("x", "x_ms")])
    assert text == "x 1.00 (1 unsteady point left out)"
    # With only two points and one unsteady, a rough slope beats no slope.
    two = [row(10, 1, 1, 1.1), row(100, 10, 1, 10)]
    assert run_bench._slopes(two, "pixels", [("x", "x_ms")]).startswith("x 1.00 (rough")
    # Memory columns have no passes, so nothing is left out.
    memory = [{"pixels": 10, "m_kb": 1.0}, {"pixels": 100, "m_kb": 10.0}]
    assert run_bench._slopes(memory, "pixels", [("m", "m_kb")]) == "m 1.00"


def test_threading_check_restores_opencv_threads():
    import cv2
    before = cv2.getNumThreads()
    rows = run_bench.threading_check(random_image(12, 15), dp_and_greedy(), [10], 12, 10, repeats=1)
    assert cv2.getNumThreads() == before
    assert [r["part"] for r in rows] == ["energy map 10 x 8", "full carve, dp", "full carve, greedy"]
    for r in rows:
        assert 0 < r["one_thread_min_ms"] <= r["one_thread_ms"] <= r["one_thread_max_ms"]
        assert 0 < r["default_threads_min_ms"] <= r["default_threads_ms"] <= r["default_threads_max_ms"]


def test_time_call_warms_up_and_batches_fast_calls():
    calls = []
    seconds, spread = run_bench.time_call(lambda: calls.append(1), repeats=3)
    # A call this fast is batched, so it runs far more often than
    # 1 warm-up + 3 samples.
    assert len(calls) > 50
    assert seconds > 0
    assert spread >= 0


def test_time_call_reports_spread_as_a_fraction():
    _, spread = run_bench.time_call(lambda: sum(range(1000)), repeats=5)
    assert 0 <= spread < 1


def test_finder_timing_has_a_time_per_size_and_finder():
    rows = run_bench.finder_timing(random_image(30, 45), dp_and_greedy(), [15, 30], repeats=1)
    assert [(r["width"], r["height"]) for r in rows] == [(15, 10), (30, 20)]
    for row in rows:
        assert row["energy_ms"] > 0 and row["dp_ms"] > 0 and row["greedy_ms"] > 0
        for prefix in ("energy", "dp", "greedy"):
            assert row[f"{prefix}_min_ms"] <= row[f"{prefix}_ms"] <= row[f"{prefix}_max_ms"]
        assert row["dp_peak_kb"] > 0 and row["greedy_peak_kb"] > 0 and row["energy_peak_kb"] > 0


def test_finder_timing_skips_finders_over_their_limit(monkeypatch):
    monkeypatch.setitem(run_bench.PIXEL_LIMITS, "greedy", 300)
    rows = run_bench.finder_timing(random_image(30, 45), dp_and_greedy(), [15, 30], repeats=1)
    assert rows[0]["greedy_ms"] is not None  # 15 x 10 = 150 pixels
    assert rows[1]["greedy_ms"] is None      # 30 x 20 = 600 pixels
    assert rows[1]["greedy_min_ms"] is None and rows[1]["greedy_peak_kb"] is None
    assert rows[1]["dp_ms"] is not None


def test_shape_scaling_varies_one_side_at_a_time():
    rows = run_bench.shape_scaling(dp_and_greedy(), repeats=1)
    shapes = [(r["vary"], r["height"], r["width"]) for r in rows]
    assert shapes == [("height", 20, 30), ("height", 40, 30),
                      ("width", 30, 20), ("width", 30, 40)]
    assert all(r["dp_ms"] > 0 and r["greedy_ms"] > 0 for r in rows)


def test_greedy_never_beats_dp_on_the_same_energy_map():
    rows = run_bench.seam_quality(random_image(30, 45), dp_and_greedy(), [20, 45], trials=20)
    assert len(rows) == (2 + 20 + 20) * 2  # (image sizes + uniform + smooth maps) x finders
    assert {row["kind"] for row in rows} == {"image", "random 100x100", "smooth 100x100"}
    # DP is optimal, so no seam can cost less than DP's on the same map.
    assert all(row["ratio"] >= 1 - 1e-9 for row in rows)
    assert all(row["optimal"] and row["rows_same"] == 1 for row in rows if row["algo"] == "dp")


def test_carve_comparison_matches_plain_carve():
    img = random_image(12, 15)
    finders = dp_and_greedy()
    rows, pictures = run_bench.carve_comparison(img, finders, 10, 9, repeats=2)

    for name, finder in finders.items():
        assert (pictures[name]["result"] == carve(img, 10, 9, finder)).all()

    by_algo = {row["algo"]: row for row in rows}
    assert by_algo["dp"]["seams"] == 5 + 3
    assert by_algo["dp"]["overlap_with_dp"] == 1.0
    assert 0 <= by_algo["greedy"]["overlap_with_dp"] <= 1
    assert 0 < by_algo["dp"]["finder_s"] <= by_algo["dp"]["total_s"]
    assert 0 < by_algo["dp"]["finder_share"] <= 1
    for part in ("total", "finder"):
        dp = by_algo["dp"]
        assert dp[f"{part}_min_s"] <= dp[f"{part}_s"] <= dp[f"{part}_max_s"]
    for row in rows:
        assert 0 < row["energy_kept"] < 1
        assert row["outside_per_seam_ms"] > 0


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_main_writes_every_output(tmp_path):
    run_bench.main(["--image", str(SMALL), "--algos", "dp", "greedy", "--sizes", "20", "40",
                    "--repeats", "2", "--trials", "3", "--output", str(tmp_path)])
    for name in ["finder_timing.csv", "finder_timing.png", "shape_scaling.csv",
                 "shape_scaling.png", "seam_quality.csv", "seam_quality.png", "carve.csv",
                 "threading.csv", "carve_dp.png", "carve_greedy.png", "seams_dp.png",
                 "seams_greedy.png", "compare.png", "time_breakdown.png", "baselines.csv",
                 "baselines.png", "memory.png", "threading.png", "summary.md",
                 "sobel_methods.csv", "sobel_methods.png", "sobel_tiles.csv"]:
        assert (tmp_path / name).exists(), name
    if run_bench.bundled_photo():  # the default second test image
        for name in ["compare_photo.png", "baselines_photo.png", "carve_photo_dp.png"]:
            assert (tmp_path / name).exists(), name
    assert not list(tmp_path.glob("*_small*"))  # no limited finder, no small set
    assert not (tmp_path / "map_dependence.csv").exists()  # only run for graph cut

    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "`images/small.png`" in summary  # path relative to the repo
    assert "python benchmarks/run_bench.py --image images/small.png" in summary
    assert "O(k x H x W)" in summary
    assert "| seam carving (dp) |" in summary and "| cropping (best) |" in summary
    assert "## References" in summary and "doi.org/10.1145/1275808.1276390" in summary
    assert "Bytes per pixel from the smallest to the largest size: dp" in summary
    # The conclusions answer each of the brief's questions.
    for phrase in ["## What the numbers say", "**Is greedy as good as DP? On seam cost,",
                   "**Full carve, greedy vs DP:**", "**Speed, greedy vs DP:**",
                   "**Time complexity (Item 5b):**", "**Space and the DP table (Items 5b, 5d):**",
                   "**Does multithreading help? (Item 5d):**",
                   "**Seam carving vs scaling and cropping:**", "**Caveats:**",
                   "**Would divide and conquer speed up the Sobel filter? (Item 2b):**",
                   "## 6. Item 2b: would divide and conquer speed up the Sobel filter?",
                   "## 7. How the measurements were made, and what was tried instead",
                   "| One untimed warm-up call before timing anything |", "| Like a crop |",
                   "### Tile size",
                   "| centre crop |"]:
        assert phrase in summary, phrase
    if run_bench.bundled_photo():
        assert "two test images" in summary and "Photo `grace_hopper.jpg`" in summary


def test_no_conclusions_without_something_to_compare(tmp_path):
    run_bench.main(["--image", str(SMALL), "--algos", "dp", "--sizes", "20", "40",
                    "--repeats", "1", "--trials", "2", "--output", str(tmp_path)])
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "## What the numbers say" not in summary
    assert "## 1. Time to find one seam" in summary
    # Every image the summary links to was written.
    for picture in re.findall(r"\]\(([\w.]+\.png)\)", summary):
        assert (tmp_path / picture).exists(), picture

    # The summary shows the same numbers as the CSV files.
    def shown(row, prefix, unit):
        median, low, high = (float(row[f"{prefix}{part}_{unit}"]) for part in ("", "_min", "_max"))
        return f"{median:.3f} ({low:.3f}–{high:.3f})"

    for row in read_csv(tmp_path / "carve.csv"):
        assert shown(row, "total", "s") in summary
    for row in read_csv(tmp_path / "finder_timing.csv"):
        assert shown(row, "dp", "ms") in summary and shown(row, "energy", "ms") in summary
    for row in read_csv(tmp_path / "threading.csv"):
        assert shown(row, "one_thread", "ms") in summary


def test_main_compares_a_limited_finder_on_small_inputs(tmp_path, monkeypatch):
    # Pretend greedy is as slow as graph cut: it may only see 600 pixels.
    monkeypatch.setitem(run_bench.PIXEL_LIMITS, "greedy", 600)
    run_bench.main(["--image", str(SMALL), "--algos", "dp", "greedy", "--sizes", "20", "40",
                    "--repeats", "1", "--trials", "3", "--output", str(tmp_path)])

    timing = read_csv(tmp_path / "finder_timing.csv")
    assert sum(row["greedy_ms"] != "" for row in timing) >= 2  # enough for a slope
    shapes = read_csv(tmp_path / "shape_scaling.csv")
    assert any(row["ladder"] == "small" and row["greedy_ms"] != "" for row in shapes)
    quality = read_csv(tmp_path / "seam_quality.csv")
    assert any(row["algo"] == "greedy" and row["kind"].startswith("random") for row in quality)
    carves = read_csv(tmp_path / "carve.csv")
    assert {row["algo"] for row in carves if row["input"] != carves[0]["input"]} == {"dp", "greedy"}
    for name in ["shape_scaling_small.png", "compare_small.png", "carve_small_greedy.png"]:
        assert (tmp_path / name).exists(), name


def test_main_includes_graph_cut_in_the_conclusions(tmp_path, monkeypatch):
    # The real graph cut, kept quick by a tight pixel limit: it only sees small inputs.
    monkeypatch.setitem(run_bench.PIXEL_LIMITS, "graphcut", 300)
    run_bench.main(["--image", str(SMALL), "--sizes", "20", "40", "--repeats", "1",
                    "--trials", "2", "--output", str(tmp_path)])  # default finders

    carves = read_csv(tmp_path / "carve.csv")
    assert {row["algo"] for row in carves if row["input"] != carves[0]["input"]} == {
        "dp", "greedy", "graphcut"}
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    for name in ["map_dependence.csv", "map_dependence.png"]:
        assert (tmp_path / name).exists(), name
    for phrase in ["- Finders: dp, greedy, graphcut",
                   "### Same sizes, different energy maps", "| graphcut searches |",
                   "**Does the image's content change the time? (Items 4, 5b):**",
                   "**Is graphcut as good as DP? On seam cost, yes.**",
                   "**Full carve, graphcut vs DP:** On the image (small copy,",
                   "**Speed, graphcut vs DP:**",
                   "**Time complexity of graphcut (Items 4, 5b):**", "Edmonds-Karp max-flow [8]",
                   "the largest size graphcut ran on",
                   "Graph cut was not timed here", "- Graph cut only runs on inputs up to 300 pixels",
                   "| Graph cut only on inputs up to 300 pixels, with a matching small set",
                   "doi.org/10.1145/321694.321699"]:
        assert phrase in summary, phrase
    # Graph cut is slower than DP, so its carve sentence must not say "only ... faster".
    assert re.search(r"graphcut finds its seams [\d.]+x slower than DP, and the whole carve", summary)
    # No sentence starts with a lowercase finder name.
    assert not re.search(r"(\. |\*\* |^- )(dp|greedy|graphcut) ", summary, re.MULTILINE)
    for picture in re.findall(r"\]\(([\w.]+\.png)\)", summary):
        assert (tmp_path / picture).exists(), picture


def test_map_dependence_times_every_finder_on_three_maps():
    img = run_bench.load_image(str(SMALL))
    finders = {**dp_and_greedy(), "graphcut": get_finder("graphcut")}
    rows = run_bench.map_dependence(img, finders, [10, 16], repeats=1)
    assert [(r["width"], r["map"]) for r in rows] == [
        (w, kind) for w in (10, 16) for kind in ("image", "random", "smooth")]
    for r in rows:
        assert r["dp_ms"] > 0 and r["greedy_ms"] > 0 and r["graphcut_ms"] > 0
        assert r["graphcut_searches"] >= 2  # at least the search that finds no path, then the cut
        assert 0 <= r["near_zero"] <= 1 and 0 <= r["greedy_edge"] <= 1
    # Uniform random values are spread evenly, so hardly any sit near zero.
    assert all(r["near_zero"] < 0.2 for r in rows if r["map"] == "random")


def test_bfs_runs_counts_searches_and_puts_graphcut_back():
    from seam_carving import graphcut
    original = graphcut._bfs
    energy = np.random.default_rng(0).random((8, 8))
    assert run_bench.bfs_runs(energy) >= 2
    assert graphcut._bfs is original


def test_map_effect_and_independence():
    def row(width, kind, dp, gc):
        return {"width": width, "height": width, "pixels": width * width, "map": kind,
                "dp_ms": dp, "graphcut_ms": gc, "graphcut_searches": int(gc)}
    rows = [row(w, kind, 1.0, {"image": 2.0, "random": 20.0, "smooth": 6.0}[kind])
            for w in (10, 20) for kind in ("image", "random", "smooth")]
    image_row, ratio, searches = run_bench._map_effect(rows, "graphcut")
    assert image_row["width"] == 20  # the largest size
    assert ratio == {"image": 1.0, "random": 10.0, "smooth": 3.0}
    assert searches == {"image": 2, "random": 20, "smooth": 6}
    assert run_bench._map_independent(rows, "dp")
    assert not run_bench._map_independent(rows, "graphcut")
    assert run_bench._map_factor(rows, "graphcut") == 10.0
    assert run_bench._map_factor(rows, "dp") == 1.0


def test_edge_share_counts_rows_in_the_first_or_last_column():
    assert run_bench.edge_share([0, 0, 1, 4], width=5) == 0.75
    assert run_bench.edge_share([2, 2], width=5) == 0.0


def test_greedy_edge_note_only_when_the_edge_explains_it():
    def rows(image_edge):
        return [{"width": 30, "map": kind, "greedy_edge": edge}
                for kind, edge in (("image", image_edge), ("random", 0.0), ("smooth", 0.1))]
    note = run_bench._greedy_edge_note(rows(1.0))
    assert note.startswith("on the image's map its seam runs along the edge in 100% of rows")
    assert "two neighbours instead of three" in note
    assert run_bench._greedy_edge_note(rows(0.1)) is None  # same as random: explains nothing


def test_carve_speed_wording_for_faster_and_slower_finders():
    ref = {"finder_s": 1.0, "total_s": 1.2}
    faster = {"finder_s": 0.1, "total_s": 0.3, "finder_share": 0.33}
    slower = {"finder_s": 80.0, "total_s": 80.2, "finder_share": 0.9975}
    assert run_bench._carve_speed("greedy", faster, ref) == (
        "greedy finds its seams 10.0x faster than DP, but the whole carve is only 4.0x faster: "
        "67% of its time goes on rebuilding the energy map and removing seams, which every "
        "finder has to do")
    assert run_bench._carve_speed("graphcut", slower, ref).startswith(
        "graphcut finds its seams 80.0x slower than DP, and the whole carve is 66.8x slower: "
        "over 99% of its time goes on finding seams")


def test_sentence_names_and_lists():
    assert [run_bench._start(n) for n in ("dp", "greedy", "graphcut", "dummy")] == [
        "DP", "Greedy", "Graph cut", "Dummy"]
    assert run_bench._and_list(["a"]) == "a"
    assert run_bench._and_list(["a", "b", "c"]) == "a, b and c"


def test_fit_rows_are_the_largest_steady_sizes(monkeypatch):
    monkeypatch.setattr(run_bench, "SLOPE_POINTS", 2)
    rows = [{"pixels": p, "t_ms": t, "t_min_ms": t, "t_max_ms": t * spread}
            for p, t, spread in ((10, 1, 1), (20, 2, 1), (40, 4, 3), (80, 8, 1))]
    # 40 is unsteady (3x), so the fit uses the two largest steady sizes: 20 and 80.
    assert [r["pixels"] for r in run_bench._fit_rows(rows, "pixels", "t_ms")] == [20, 80]


def test_share_and_energy_wording():
    assert run_bench._share(0.873) == "87%"
    assert run_bench._share(0.998) == "over 99%"
    assert run_bench._share(0.002) == "under 1%"
    assert run_bench._energy_vs_dp(1.45) == "45% more energy than DP's"
    assert run_bench._energy_vs_dp(1.0) == "the same energy as DP's"


def test_graph_cut_is_a_default_finder():
    assert run_bench.parse_args([]).algos == ["dp", "greedy", "graphcut"]


# ---------- Item 2b: the Sobel filter four ways ----------

@pytest.mark.parametrize("shape", [(2, 3), (7, 10), (65, 129), (130, 70)])
def test_every_sobel_method_gives_opencvs_map(shape):
    gray = np.random.default_rng(1).random(shape) * 255
    reference = run_bench.sobel_opencv(gray)
    for name, method in run_bench.SOBEL_METHODS.items():
        assert np.allclose(method(gray), reference, atol=1e-9), name
    # Tiny tiles force many splits along both sides.
    assert np.allclose(run_bench.sobel_divide_conquer(gray, leaf=4), reference, atol=1e-9)


def test_opencv_method_is_exactly_energy_py():
    from seam_carving.energy import compute_energy
    img = run_bench.load_image(str(SMALL))
    assert np.array_equal(run_bench.sobel_opencv(run_bench.gray_of(img)), compute_energy(img))


def test_sobel_methods_rows():
    img = run_bench.load_image(str(SMALL))
    rows = run_bench.sobel_methods(img, [12, 24], repeats=1)
    assert [r["width"] for r in rows] == [12, 24]
    for r in rows:
        for method in run_bench.SOBEL_METHODS:
            assert r[f"{method}_ms"] > 0 and r[f"{method}_diff"] < 1e-9


# ---------- the photo, crops and the method checks ----------

def test_crop_to_shape_keeps_the_middle():
    img = np.zeros((100, 40, 3), dtype=np.uint8)
    img[40:60] = 255  # a white band across the middle
    out = run_bench.crop_to_shape(img, 10, 20)  # wide target: keeps the middle rows
    assert out.shape == (10, 20, 3) and out.mean() > 100
    assert run_bench.crop_to_shape(np.zeros((10, 100, 3), np.uint8), 5, 5).shape == (5, 5, 3)


def test_bundled_photo_is_a_readable_image():
    path = run_bench.bundled_photo()
    if path is None:
        pytest.skip("this matplotlib has no sample data")
    assert run_bench.load_image(path).ndim == 3


def test_bundled_photo_is_none_without_sample_data(monkeypatch):
    from matplotlib import cbook

    def missing(*args, **kwargs):
        raise OSError("no sample data")
    monkeypatch.setattr(cbook, "get_sample_data", missing)
    assert run_bench.bundled_photo() is None


def test_crop_likeness_is_one_for_a_crop_and_lower_for_seams():
    strip = np.zeros((10, 12), dtype=bool)
    strip[:, :3] = True  # a crop to 10 x 9 that cut 3 columns off the left
    assert run_bench.crop_likeness(strip, 10, 9) == 1.0
    seams = np.zeros((10, 12), dtype=bool)
    for row in range(10):
        seams[row, [row % 12, (row + 4) % 12, (row + 8) % 12]] = True  # three slanted seams
    assert run_bench.crop_likeness(seams, 10, 9) < 0.5


def test_baselines_include_a_centre_crop_that_never_beats_the_best_crop():
    img = random_image(20, 30)
    rows, _pictures = run_bench.baselines(img, 24, 15, repeats=1)
    kept = {r["method"]: r["energy_kept"] for r in rows}
    assert set(kept) == {"scaling", "cropping", "centre crop"}
    assert kept["centre crop"] <= kept["cropping"] + 1e-12
    assert all(r["time_ms"] > 0 for r in rows)  # crops are real copies, not free views


def test_method_checks_measure_cold_start_batching_and_tracing():
    img = run_bench.load_image(str(SMALL))
    checks = run_bench.method_checks(img, dp_and_greedy(), repeats=2)
    for key in ["cold_energy_ms", "warm_energy_ms", "cold_finder_ms", "warm_finder_ms",
                "single_us", "batched_us", "resolution_us", "traced_ratio"]:
        assert checks[key] > 0, key
    assert checks["finder"] == "dp" and checks["fast_finder"] == "greedy"
    assert checks["single_drift"] >= 0 and checks["batched_drift"] >= 0


def test_method_checks_skip_tracing_when_tracemalloc_is_already_running():
    import tracemalloc
    tracemalloc.start()
    try:
        checks = run_bench.method_checks(run_bench.load_image(str(SMALL)),
                                         {"greedy": get_finder("greedy")}, repeats=1)
    finally:
        tracemalloc.stop()
    assert checks["traced_ratio"] is None and checks["finder"] == "greedy"


def test_pass_spread_finds_the_widest_timing():
    rows = [{"width": 10, "height": 5, "dp_min_ms": 1.0, "dp_max_ms": 1.2},
            {"width": 20, "height": 9, "dp_min_ms": 2.0, "dp_max_ms": 5.0}]
    median, worst, where = run_bench.pass_spread(rows, ["dp"], lambda r: f"{r['width']} x {r['height']}")
    assert worst == 2.5 and where == "dp at 20 x 9" and median == pytest.approx(1.85)
    assert run_bench.pass_spread([], ["dp"], str) is None


def test_main_without_the_photo(tmp_path):
    run_bench.main(["--image", str(SMALL), "--algos", "dp", "greedy", "--sizes", "20", "40",
                    "--repeats", "1", "--trials", "2", "--photo", "none", "--output", str(tmp_path)])
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "one test image" in summary and "--photo none" in summary
    assert not list(tmp_path.glob("*photo*"))


def test_main_with_a_chosen_photo(tmp_path):
    run_bench.main(["--image", str(SMALL), "--algos", "dp", "greedy", "--sizes", "20", "40",
                    "--repeats", "1", "--trials", "2", "--photo", str(SMALL), "--output", str(tmp_path)])
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "- Photo: `images/small.png`" in summary and "--photo images/small.png" in summary


def test_main_reports_a_missing_photo(tmp_path):
    with pytest.raises(SystemExit, match="Could not read image"):
        run_bench.main(["--image", str(SMALL), "--photo", str(tmp_path / "nope.jpg"),
                        "--output", str(tmp_path)])


# ---------- the summary's wording, on made-up results ----------
# Real runs only reach some wording branches by chance (e.g. divide and conquer
# beating one pass), so these build every input of build_summary by hand.

def timed(prefix, value, unit="ms"):
    if value is None:
        return {f"{prefix}_{unit}": None, f"{prefix}_min_{unit}": None, f"{prefix}_max_{unit}": None}
    return {f"{prefix}_{unit}": value, f"{prefix}_min_{unit}": value * 0.95,
            f"{prefix}_max_{unit}": value * 1.05}


def fake_run():
    names = ["dp", "greedy", "graphcut"]
    setup = {"image": "images/sample.png", "width": 80, "height": 53, "new_width": 60,
             "new_height": 42, "finders": names, "skipped": [], "repeats": 5,
             "sizes": [20, 40, 80], "small_sizes": [10], "trials": 2, "seed": 0,
             "random_shapes": [(100, 100), (8, 8)],
             "shape_ladders": [("main", 40, [20, 40, 80]), ("small", 8, [4, 8, 16])],
             "cv_threads": 8, "command": "python benchmarks/run_bench.py", "quality_chart": True,
             "photo": "grace_hopper.jpg", "photo_bundled": True, "photo_size": "512 x 600"}
    timing = []
    for w, h in ((10, 7), (20, 13), (40, 27), (80, 53)):
        px = w * h
        gc = 0.01 * px ** 1.5 if px <= 1100 else None
        timing.append({"width": w, "height": h, "pixels": px, **timed("energy", 1e-5 * px),
                       "energy_peak_kb": 48 * px / 1024, **timed("dp", 1e-3 * px ** 0.7),
                       "dp_peak_kb": (px + 200) / 1024, **timed("greedy", 1e-4 * px ** 0.5),
                       "greedy_peak_kb": h * 8 / 1024, **timed("graphcut", gc),
                       "graphcut_peak_kb": 700 * px / 1024 if gc else None})
    shapes = []
    for ladder, base, steps in setup["shape_ladders"]:
        for vary in ("height", "width"):
            for step in steps:
                h, w = (step, base) if vary == "height" else (base, step)
                shapes.append({"ladder": ladder, "vary": vary, "height": h, "width": w,
                               **timed("dp", 1e-3 * h), **timed("greedy", 1e-4 * (h + w)),
                               **timed("graphcut", 0.01 * (h * w) ** 2.2 if ladder == "small" else None)})
    quality = []
    for kind, greedy_ratio in (("image", 1.2), ("random 8x8", 1.5), ("photo", 1.3)):
        for algo, ratio in (("dp", 1.0), ("greedy", greedy_ratio), ("graphcut", 1.0)):
            quality.append({"kind": kind, "source": "#0", "algo": algo, "cost": ratio,
                            "dp_cost": 1.0, "ratio": ratio, "optimal": ratio == 1.0, "rows_same": 1.0})

    def carve_row(picture, inp, algo, total, finder, removed, kept, with_dp, like):
        return {"picture": picture, "input": inp, "algo": algo, "seams": 31,
                **timed("total", total, "s"), **timed("finder", finder, "s"),
                "finder_share": finder / total, "outside_per_seam_ms": 1000 * (total - finder) / 31,
                "energy_removed": removed, "energy_kept": kept, "overlap_with_dp": with_dp,
                "crop_likeness": like}
    carve_rows = [
        carve_row("image", "80 x 53 -> 60 x 42", "dp", 0.3, 0.25, 1000, 0.88, 1.0, 0.3),
        carve_row("image", "80 x 53 -> 60 x 42", "greedy", 0.06, 0.01, 1450, 0.71, 0.5, 0.9),
        carve_row("image", "20 x 13 -> 15 x 10", "dp", 0.02, 0.017, 300, 0.87, 1.0, 0.35),
        carve_row("image", "20 x 13 -> 15 x 10", "greedy", 0.003, 0.001, 400, 0.80, 0.4, 0.4),
        carve_row("image", "20 x 13 -> 15 x 10", "graphcut", 1.5, 1.49, 300, 0.87, 1.0, 0.35),
        carve_row("photo", "80 x 94 -> 60 x 75", "dp", 0.6, 0.5, 2000, 0.92, 1.0, 0.5),
        carve_row("photo", "80 x 94 -> 60 x 75", "greedy", 0.15, 0.02, 7000, 0.57, 0.2, 0.4),
        carve_row("photo", "16 x 19 -> 12 x 15", "dp", 0.02, 0.017, 500, 0.9, 1.0, 0.6),
        carve_row("photo", "16 x 19 -> 12 x 15", "greedy", 0.003, 0.001, 900, 0.7, 0.3, 0.5),
        carve_row("photo", "16 x 19 -> 12 x 15", "graphcut", 1.5, 1.49, 500, 0.9, 1.0, 0.6),
    ]
    threads = [{"part": f"energy map {w} x {h}", **timed("one_thread", 0.1), **timed("default_threads", 0.1),
                "speedup": 1.0} for w, h in ((20, 13), (80, 53))]
    threads += [{"part": f"full carve, {n}", **timed("one_thread", 300.0),
                 **timed("default_threads", 297.0), "speedup": 1.01} for n in ("dp", "greedy")]
    base_rows = [{"picture": p, "method": m, **timed("time", t), "energy_kept": k}
                 for p in ("image", "photo")
                 for m, t, k in (("scaling", 0.1, None), ("cropping", 0.3, 0.89), ("centre crop", 0.01, 0.8))]
    map_rows = []
    for w, h in ((10, 7), (20, 13)):
        for kind, gc, searches, near, edge in (("image", 1.0, 20, 0.3, 1.0), ("photo", 3.0, 60, 0.25, 0.2),
                                               ("random", 12.0, 200, 0.01, 0.0), ("smooth", 8.0, 150, 0.0, 0.4)):
            map_rows.append({"width": w, "height": h, "pixels": w * h, "map": kind, "near_zero": near,
                             **timed("dp", 0.1 * w), **timed("greedy", (0.01 if kind == "image" else 0.014) * w),
                             **timed("graphcut", gc * w ** 2), "graphcut_searches": searches * w,
                             "greedy_edge": edge})
    sobel_rows = [{"width": w, "height": h, "pixels": w * h, **timed("opencv", 0.001 * w),
                   **timed("direct", 0.004 * w), **timed("divide", 0.006 * w), **timed("fft", 0.012 * w),
                   **{f"{m}_diff": 1e-13 for m in run_bench.SOBEL_METHODS}}
                  for w, h in ((20, 13), (80, 53))]
    tile_rows = [{"leaf": leaf, "label": label, "tiles": tiles, "width": 80, "height": 53,
                  **timed("divide", ms)}
                 for leaf, label, tiles, ms in ((16, "at most 16 x 16", 24, 9.0),
                                                (32, "at most 32 x 32", 8, 2.0),
                                                (80, "whole image", 1, 6.0))]
    checks = {"size": "80 x 53", "cold_energy_ms": 5.0, "warm_energy_ms": 0.05, "finder": "dp",
              "cold_finder_ms": 2.0, "warm_finder_ms": 1.0, "fast_finder": "greedy",
              "tiny_size": "20 x 13", "single_us": 9.0, "batched_us": 4.0, "single_drift": 0.4,
              "batched_drift": 0.05, "resolution_us": 0.1, "traced_ratio": 5.0}
    return dict(setup=setup, timing=timing, shapes=shapes, quality=quality, carve_rows=carve_rows,
                threads=threads, base_rows=base_rows, map_rows=map_rows, sobel_rows=sobel_rows,
                checks=checks, tile_rows=tile_rows)


def summary_of(run):
    return run_bench.build_summary(**run)


def test_fake_run_summary_has_every_part_and_no_lowercase_sentence_starts():
    summary = summary_of(fake_run())
    for phrase in ["## What the numbers say", "## 6. Item 2b", "## 7. How the measurements were made",
                   "So on the image greedy behaves far more like a plain crop than DP does",
                   "So the results are the same pictures (compare_small.png and compare_photo_small.png",
                   "batching made the result 8.0x steadier",
                   "One-call samples also time the timer's own overhead, which made them read 2.2x",
                   "tracemalloc made dp 5.0x slower", "Divide and conquer took 1.50x the time",
                   "So it can only add the cost of the recursion, as measured.",
                   "DP does the same steps whatever the values",
                   "The photo sits in between: 1200 searches",
                   "slower than these images would give",
                   "one carve would take over"]:
        assert phrase in summary, phrase
    assert not re.search(r"(\. |\*\* |^- )(dp|greedy|graphcut) ", summary, re.MULTILINE)


def test_fake_run_wording_when_divide_and_conquer_wins_through_the_cache():
    run = fake_run()
    run["sobel_rows"][-1]["divide_ms"] = run["sobel_rows"][-1]["direct_ms"] * 0.5  # only the largest
    summary = summary_of(run)
    assert "**Would divide and conquer speed up the Sobel filter? (Item 2b):** not by cutting the work" in summary
    assert "ran 2.0x faster than the same filter in one pass at 80 x 53 but was slower at smaller sizes" in summary
    assert "took 3.0x as long as with tiles of at most 32 x 32, so the gain comes from the CPU's cache" in summary
    assert "It still ran faster at 80 x 53, though slower at 20 x 13. The tile-size check below shows why" in summary
    assert "At the smaller sizes the whole image's arrays fit in the cache anyway" in summary
    assert "### Tile size" in summary
    assert "Smaller tiles were slower because every tile costs a Python call" in summary
    assert "Bigger tiles were slower too, up to 6.00 ms for one whole-image tile (3.0x the best)" in summary


def test_fake_run_wording_when_tiles_show_no_cache_effect():
    run = fake_run()
    for row in run["sobel_rows"]:
        row["divide_ms"] = row["direct_ms"] * 0.5
    for row in run["tile_rows"]:
        row["divide_ms"] = 2.0  # tile size makes no difference
    summary = summary_of(run)
    assert "the cause of the gain is unclear" in summary
    assert "does not show a clear cache effect, so the cause is unclear" in summary
    assert "close to the best, so the cache makes little difference at this size" in summary


def test_tile_effect_needs_small_tiles_to_compare_with():
    assert run_bench._tile_effect([]) is None
    whole_only = [{"leaf": 9, "label": "whole image", "divide_ms": 1.0}]
    assert run_bench._tile_effect(whole_only) is None and run_bench._tile_sentence(whole_only) == ""


def test_tile_count_splits_like_divide_and_conquer():
    assert run_bench.tile_count(4, 4, 4) == 1
    assert run_bench.tile_count(8, 4, 4) == 2
    assert run_bench.tile_count(533, 800, 64) == 256
    calls = []
    original = run_bench._sobel_inside
    try:
        run_bench._sobel_inside = lambda tile: calls.append(tile.shape) or original(tile)
        run_bench.sobel_divide_conquer(np.zeros((37, 51)), leaf=8)
    finally:
        run_bench._sobel_inside = original
    assert len(calls) == run_bench.tile_count(37, 51, 8)


def test_sobel_tile_sizes_end_with_one_whole_image_tile():
    rows = run_bench.sobel_tile_sizes(run_bench.load_image(str(SMALL)), 40, repeats=1)
    assert rows[-1]["label"] == "whole image" and rows[-1]["tiles"] == 1
    assert [r["leaf"] for r in rows[:-1]] == [16, 32]  # tile sizes smaller than the image
    assert all(r["divide_ms"] > 0 for r in rows)


def test_fake_run_wording_when_results_differ_but_neither_is_crop_like():
    run = fake_run()
    for row in run["carve_rows"]:
        row["crop_likeness"] = 0.4
    summary = summary_of(run)
    assert "Whether the difference is visible depends on the image" in summary


def test_fake_run_wording_when_the_crop_ties_with_dp():
    run = fake_run()
    for row in run["base_rows"]:
        if row["method"] == "cropping" and row["picture"] == "image":
            row["energy_kept"] = 0.88  # DP's on the image
    assert "the best crop kept 88.0% of the original energy, the same as DP's 88.0%" in summary_of(run)


def test_fake_run_wording_without_the_photo():
    run = fake_run()
    run["setup"]["photo"] = None
    run["carve_rows"] = [r for r in run["carve_rows"] if r["picture"] == "image"]
    run["base_rows"] = [r for r in run["base_rows"] if r["picture"] == "image"]
    run["map_rows"] = [r for r in run["map_rows"] if r["map"] != "photo"]
    summary = summary_of(run)
    assert "one test image" in summary and "slower than this image would give" in summary


def test_fake_run_wording_when_the_photo_is_not_in_between():
    run = fake_run()
    for row in run["map_rows"]:
        if row["map"] == "photo":
            row["graphcut_searches"] *= 10  # more than the random maps
    assert "On the photo's map it needed 12000 searches" in summary_of(run)  # 60 x 20 x 10


@pytest.mark.parametrize("single, batched, words", [
    (0.4, 0.05, "batching made the result 8.0x steadier"),
    (0.05, 0.4, "batching was less steady in this run"),
    (0.1, 0.1, "the two were about as steady in this run"),
])
def test_fake_run_batching_verdicts(single, batched, words):
    run = fake_run()
    run["checks"].update(single_drift=single, batched_drift=batched, single_us=4.0)
    summary = summary_of(run)
    assert words in summary and "One-call samples also time" not in summary


def test_fake_run_explains_why_two_graph_cut_slopes_differ():
    run = fake_run()
    for row in run["map_rows"]:  # make the image's maps grow more slowly than section 1 does
        if row["map"] == "image":
            row["graphcut_ms"] = row["width"] ** 1.2
            row["graphcut_min_ms"] = row["graphcut_max_ms"] = row["graphcut_ms"]
    assert "Section 1 gives" in summary_of(run) and "fitted on its own 3 largest sizes" in summary_of(run)


def test_fake_run_skips_what_was_not_measured():
    run = fake_run()
    run["shapes"] = [r for r in run["shapes"] if r["ladder"] == "main"]  # no small ladder
    for row in run["timing"]:
        row["graphcut_peak_kb"] = None  # memory never measured
    run["sobel_rows"] = []
    summary = summary_of(run)
    assert "Time complexity of graphcut" not in summary and "## 6. Item 2b" not in summary
    assert "the largest size graphcut ran on" not in summary
    assert run_bench._graphcut_full_estimate(run["setup"], run["timing"], run["shapes"]) is None
    unmeasured = [{**r, "ladder": "small", **timed("graphcut", None)} for r in run["shapes"]]
    assert run_bench._graphcut_full_estimate(run["setup"], run["timing"], unmeasured) is None


# ---------- small branches ----------

@pytest.mark.parametrize("argv, message", [
    (["--trials", "-1"], "--trials cannot be negative"),
    (["--sizes", "2", "10"], "--sizes must all be at least 3"),
])
def test_parse_args_rejects_bad_values(argv, message, capsys):
    with pytest.raises(SystemExit):
        run_bench.parse_args(argv)
    assert message in capsys.readouterr().err


def test_main_skips_unfinished_finders_and_needs_at_least_one(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_bench, "available_finders", lambda names: ({}, ["graphcut"]))
    with pytest.raises(SystemExit, match="No finished seam finders"):
        run_bench.main(["--image", str(SMALL), "--output", str(tmp_path)])
    assert "Skipping graphcut: not implemented yet" in capsys.readouterr().out


def test_fit_width_matches_brute_force_on_awkward_shapes():
    for height, width in ((7, 100), (100, 7), (33, 50), (3, 1000), (17, 23)):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        for limit in (50, 99, 600, 3600):
            expected = max([w for w in range(3, 5000)
                            if w * run_bench.resized_height(img, w) <= limit], default=3)
            assert run_bench.fit_width(img, limit) == expected, (height, width, limit)


def test_bfs_runs_is_none_when_graph_cut_has_no_bfs(monkeypatch):
    from seam_carving import graphcut
    monkeypatch.delattr(graphcut, "_bfs")
    assert run_bench.bfs_runs(np.ones((4, 4))) is None


def test_write_csv_writes_nothing_for_no_rows(tmp_path):
    run_bench.write_csv([], tmp_path / "empty.csv")
    assert not (tmp_path / "empty.csv").exists()


def test_charts_skip_a_finder_with_nothing_measured(tmp_path):
    run = fake_run()
    rows = [{**r, "ghost_peak_kb": None} for r in run["timing"]]
    run_bench.plot_memory(rows, ["dp", "ghost"], tmp_path / "memory.png")
    maps = [{**r, "ghost_ms": None} for r in run["map_rows"]]
    run_bench.plot_map_dependence(maps, ["dp", "ghost"], tmp_path / "maps.png")
    assert (tmp_path / "memory.png").exists() and (tmp_path / "maps.png").exists()


def test_script_runs_directly_and_finds_its_package(monkeypatch, capsys):
    import runpy
    import sys
    script = Path(run_bench.__file__)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != str(run_bench.REPO_ROOT)])
    monkeypatch.setattr(sys, "argv", [str(script), "--help"])
    with pytest.raises(SystemExit) as stop:
        runpy.run_path(str(script), run_name="__main__")
    assert stop.value.code == 0 and "--photo" in capsys.readouterr().out


def test_rejects_bad_arguments():
    with pytest.raises(SystemExit):
        run_bench.parse_args(["--repeats", "0"])


@pytest.mark.parametrize("extra, message", [
    (["--width", "80"], "must be between"),          # small.png is only 50 wide
    (["--width", "50", "--height", "40"], "nothing to carve"),
])
def test_main_rejects_bad_targets(tmp_path, extra, message):
    with pytest.raises(SystemExit, match=message):
        run_bench.main(["--image", str(SMALL), "--output", str(tmp_path)] + extra)


def test_main_reports_a_missing_image(tmp_path):
    with pytest.raises(SystemExit, match="Could not read image"):
        run_bench.main(["--image", str(tmp_path / "nope.png"), "--output", str(tmp_path)])
