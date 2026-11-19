# Seam Carving (CSD3130 Assignment 2, Team 35)

Content-aware image resizing with three seam finders: dynamic programming,
greedy and graph cut (min-cut / max-flow).

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python cli.py images/sample.png --width 180 --height 130 --algo dp --output out/dp.png
pytest
streamlit run app.py   # interactive app, opens http://localhost:8501
```

`--algo` is one of `dummy`, `dp`, `greedy`, `graphcut`. `dummy` is a placeholder
(best straight vertical line) so the pipeline works before the real finders exist.
Use `images/small.png` (50 x 40) for graph cut.

## Project layout

| Path | Owner | What |
| --- | --- | --- |
| `seam_carving/core.py` | P1 | load/save, seam removal, carve loop |
| `seam_carving/energy.py` | P1 | Sobel energy map |
| `seam_carving/seam_utils.py` | P1 | `is_valid_seam`, `seam_cost` (use in your tests) |
| `seam_carving/finders.py` | P1 | algorithm name to finder lookup |
| `seam_carving/dp.py` | P2 | DP seam finder |
| `seam_carving/greedy.py` | P4 | greedy seam finder |
| `seam_carving/graphcut.py` | P5 | graph cut seam finder (min-cut / max-flow, small images only) |
| `seam_carving/visualize.py` | P6 | seam overlay, all-seams image, step-by-step GIF |
| `app.py` | P6 | interactive Streamlit app (resize, step-by-step view, compare) |
| `benchmarks/run_bench.py` | P3 | timing and comparison of the finders |
| `cli.py` | P1 | command-line entry point |
| `tests/` | everyone | `test_<your part>.py` |

## Shared interface

Every seam finder only finds **vertical** seams on an energy array.
`carve()` handles horizontal seams by transposing the image.

```python
find_vertical_seam(energy) -> seam
# energy: H x W float array, values >= 0
# seam:   int array of shape (H,), seam[i] = column removed in row i
#         0 <= seam[i] < W and |seam[i] - seam[i-1]| <= 1
```

`tests/test_finder_contract.py` checks this automatically for every finder.
It skips finders that still raise `NotImplementedError`.

Visualisation hook (P6): `carve(img, new_w, new_h, finder, on_seam=callback)` calls

```python
callback(step, direction, current_img, seam_mask, orig_rows, orig_cols)
# direction:   "vertical" or "horizontal"
# current_img: image before this seam is removed (normal orientation)
# seam_mask:   bool array, same height/width as current_img, True on the seam
# orig_rows, orig_cols: where the removed pixels were in the ORIGINAL image
```

## Benchmarks (P3)

`benchmarks/run_bench.py` times and compares every finished finder on the same inputs.
Its results are the source of report Sections 3.3 and 8.

```bash
python benchmarks/run_bench.py                    # dp, greedy and graphcut, about 5 minutes
python benchmarks/run_bench.py --algos dp greedy  # leave out graph cut
```

It runs six experiments, after a few checks on the timing method itself:

1. Time and memory to find one seam as the image grows (Item 5b).
2. Height against width, and the same sizes on different energy maps (Items 4, 5b).
3. Seam cost against DP's on the same energy map, for images and random maps (Items 3, 4, 5c).
4. Full carves of the drawing and a photo, against scaling and the best crop (Items 3, 4, 5c).
5. OpenCV on 1 thread against its default number of threads (Item 5d).
6. The Sobel filter five ways: OpenCV, one NumPy pass with new or reused arrays, divide and conquer, FFT (Item 2b).

- Results go to `out/bench/` (git-ignored): `summary.md` has every table and conclusion, plus CSVs, charts and seam pictures.
- Graph cut only runs on inputs up to 3,600 pixels (60 x 60, the same limit as the app), so every finder also runs on small copies of the inputs and graph cut is always compared on the same ones.
- Each timing is the median of 3 passes that take turns over the inputs. A timing whose slowest pass is over 1.5x its fastest is flagged in `summary.md` and left out of the growth fits, so close heavy programs (games, video) while it runs.
- On Windows the script asks not to be throttled, which keeps it on the fast cores of CPUs that also have efficient ones.
- The second test image is matplotlib's sample photo `grace_hopper.jpg`; `--photo none` leaves it out. `--help` lists the other options (`--image`, `--sizes`, `--width`, `--height`, `--repeats`, `--trials`, `--seed`, `--output`).
- `tests/test_bench.py` checks the benchmark itself and runs every line of it.

## Graph cut (P5)

`seam_carving/graphcut.py` finds the seam as a minimum s–t cut:

1. One node per pixel, plus source `s` and sink `t`.
   `s` → every column-0 pixel (∞), each pixel → its right neighbour with capacity = its energy,
   last-column pixels → `t` with capacity = their energy.
2. ∞ backward edges `(i, j+1) → (i, j)` force exactly one cut per row; ∞ diagonal edges
   `(i, j) → (i±1, j-1)` keep the seam connected.
3. Edmonds–Karp max-flow, then a BFS from `s` in the residual graph finds the source side of the cut.
4. Seam pixel in each row = the rightmost pixel still reachable from `s`.

The seam has the same total energy as DP's (max-flow min-cut theorem); `tests/test_graphcut.py`
checks this on random arrays. Unlike the brief, the last column drains to `t` with capacity = energy
rather than ∞, so the last column can be removed and 1-column images don't give infinite flow.

Edmonds–Karp is O(V·E²) per seam, so use `images/small.png`:

```bash
python cli.py images/small.png --width 40 --height 30 --algo graphcut --output out/graphcut.png
```

## Visualisation (P6)

`seam_carving/visualize.py` records every seam while `carve()` runs and draws it.

```python
from seam_carving.core import carve, load_image, save_image
from seam_carving.finders import get_finder
from seam_carving.visualize import SeamRecorder, all_seams_on_original, make_gif

img = load_image("images/sample.png")
recorder = SeamRecorder()                      # pass as the on_seam callback
result = carve(img, 180, 130, get_finder("dp"), on_seam=recorder)

# Every removed seam on the original image (red = vertical, blue = horizontal)
save_image(all_seams_on_original(img, recorder.removed), "out/all_seams.png")

# Step-by-step animation: every 2nd seam, 2x enlarged, ending on the result
make_gif(recorder.frames, "out/steps.gif", every=2, scale=2, final_frame=result)
```

- `recorder.removed`: list of `(direction, orig_rows, orig_cols)`, one per seam, in original-image coordinates.
- `recorder.frames`: one image per seam with that seam drawn on it. Use `SeamRecorder(keep_frames=False)` for big images if you only need the all-seams picture.
- `tests/test_visualize.py` checks that deleting the recorded pixels from the original gives exactly the carved result.

## Workflow

1. Branch off `main`: `git checkout -b p2-dp`
2. Commit and push your branch; CI runs the tests on every push.
3. Open a pull request into `main`. Merge only when CI is green.
4. Log any AI prompts you used in `AI_LOG.md`.

After a merge into `main`, CI also carves the sample image with each finished
finder and uploads the results under the run's **Artifacts** (`carved-samples`).
