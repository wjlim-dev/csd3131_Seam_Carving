"""Tests for seam_carving/visualize.py (P6)."""

import numpy as np
import pytest
from PIL import Image

from seam_carving.core import carve
from seam_carving.finders import get_finder
from seam_carving.visualize import (
    HORIZONTAL_COLOUR,
    VERTICAL_COLOUR,
    SeamRecorder,
    all_seams_on_original,
    make_gif,
    overlay_seam,
    pad_to,
)


def random_image(height, width, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def removed_mask(recorder, height, width):
    """True at every original pixel that some seam removed."""
    mask = np.zeros((height, width), dtype=bool)
    for _, rows, cols in recorder.removed:
        mask[rows, cols] = True
    return mask


# ---------- overlay_seam ----------

def test_overlay_paints_only_seam_pixels():
    img = random_image(6, 8)
    mask = np.zeros((6, 8), dtype=bool)
    mask[np.arange(6), [3, 3, 4, 4, 3, 2]] = True

    out = overlay_seam(img, mask)

    assert (out[mask] == VERTICAL_COLOUR).all()
    assert (out[~mask] == img[~mask]).all()


def test_overlay_does_not_change_input():
    img = random_image(5, 5)
    before = img.copy()
    overlay_seam(img, np.ones((5, 5), dtype=bool))
    assert (img == before).all()


# ---------- SeamRecorder ----------

def test_recorder_one_entry_per_seam_in_order():
    img = random_image(12, 15)
    recorder = SeamRecorder()
    carve(img, 10, 9, get_finder("dp"), on_seam=recorder)

    directions = [d for d, _, _ in recorder.removed]
    assert directions == ["vertical"] * 5 + ["horizontal"] * 3
    assert len(recorder.frames) == 8


def test_recorder_without_frames():
    recorder = SeamRecorder(keep_frames=False)
    carve(random_image(10, 10), 8, 10, get_finder("dp"), on_seam=recorder)
    assert len(recorder.removed) == 2
    assert recorder.frames == []


def test_frames_shrink_by_one_column_per_vertical_seam():
    recorder = SeamRecorder()
    carve(random_image(10, 12), 9, 10, get_finder("dp"), on_seam=recorder)
    widths = [f.shape[1] for f in recorder.frames]
    assert widths == [12, 11, 10]  # each frame is the image BEFORE that seam


def test_no_original_pixel_is_removed_twice():
    height, width = 12, 15
    recorder = SeamRecorder(keep_frames=False)
    carve(random_image(height, width), 10, 9, get_finder("dp"), on_seam=recorder)

    positions = [(r, c) for _, rows, cols in recorder.removed for r, c in zip(rows, cols)]
    assert len(positions) == len(set(positions))
    assert len(positions) == height * width - 10 * 9


def test_recorded_positions_match_carved_result():
    # Strongest check: if we delete exactly the recorded pixels from the
    # original, what is left must be the carved image, pixel for pixel.
    height, width, new_width = 10, 14, 9
    img = random_image(height, width, seed=3)
    recorder = SeamRecorder(keep_frames=False)
    result = carve(img, new_width, height, get_finder("dp"), on_seam=recorder)

    keep = ~removed_mask(recorder, height, width)
    rebuilt = img[keep].reshape(height, new_width, 3)
    assert (rebuilt == result).all()


def test_recorded_positions_match_after_horizontal_seams():
    height, width, new_height = 12, 10, 8
    img = random_image(height, width, seed=4)
    recorder = SeamRecorder(keep_frames=False)
    result = carve(img, width, new_height, get_finder("dp"), on_seam=recorder)

    # Horizontal seams remove one pixel per COLUMN, so rebuild column by column.
    keep = ~removed_mask(recorder, height, width)
    rebuilt = img.transpose(1, 0, 2)[keep.T].reshape(width, new_height, 3).transpose(1, 0, 2)
    assert (rebuilt == result).all()


# ---------- all_seams_on_original ----------

def test_all_seams_colours_by_direction():
    img = random_image(10, 10)
    recorder = SeamRecorder(keep_frames=False)
    carve(img, 8, 8, get_finder("dp"), on_seam=recorder)

    out = all_seams_on_original(img, recorder.removed)

    direction, rows, cols = recorder.removed[0]
    assert direction == "vertical"
    assert (out[rows, cols] == VERTICAL_COLOUR).all()
    direction, rows, cols = recorder.removed[-1]
    assert direction == "horizontal"
    assert (out[rows, cols] == HORIZONTAL_COLOUR).all()

    untouched = ~removed_mask(recorder, 10, 10)
    assert (out[untouched] == img[untouched]).all()


# ---------- pad_to / make_gif ----------

def test_pad_to_keeps_image_top_left():
    img = random_image(3, 4)
    out = pad_to(img, 5, 6)
    assert out.shape == (5, 6, 3)
    assert (out[:3, :4] == img).all()
    assert (out[3:] == 255).all() and (out[:, 4:] == 255).all()


def test_make_gif_frame_count_and_size(tmp_path):
    img = random_image(10, 12)
    recorder = SeamRecorder()
    result = carve(img, 8, 10, get_finder("dp"), on_seam=recorder)  # 4 seams
    path = tmp_path / "steps.gif"

    make_gif(recorder.frames, path, every=2, scale=3, final_frame=result)

    gif = Image.open(path)
    assert gif.n_frames == 2 + 1          # steps 0 and 2, plus the result
    assert gif.size == (12 * 3, 10 * 3)   # (width, height), original size scaled


def test_make_gif_rejects_empty(tmp_path):
    with pytest.raises(ValueError):
        make_gif([], tmp_path / "empty.gif")
