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
| `seam_carving/graphcut.py` | P5 | graph cut seam finder |
| `seam_carving/visualize.py` | P6 | seam overlay, all-seams image, step-by-step GIF |
| `app.py` | P6 | interactive app (to add) |
| `benchmarks/run_bench.py` | P3 | timing and comparison (to add) |
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
