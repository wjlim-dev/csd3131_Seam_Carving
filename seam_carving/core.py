"""Core pipeline: load/save images, remove seams, and the carve loop.

Seam finders only ever see an energy array and only find VERTICAL seams.
Horizontal seams are handled here by transposing the image, so the
finders (DP, greedy, graph cut) never need to know about directions.
"""

import cv2
import numpy as np

from seam_carving.energy import compute_energy
from seam_carving.seam_utils import is_valid_seam


def load_image(path):
    """Read an image file as an H x W x 3 uint8 RGB array."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    # OpenCV loads BGR; the rest of the project works in RGB.
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(img, path):
    """Write an RGB array to disk (format picked from the file extension)."""
    bgr = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise IOError(f"Could not write image: {path}")


def remove_vertical_seam(arr, seam):
    """Remove one pixel per row from arr at the columns given by seam.

    Works for H x W x 3 images and for H x W arrays (the index maps below).
    A boolean mask avoids a slow Python loop over every pixel.
    """
    height, width = arr.shape[:2]

    keep = np.ones((height, width), dtype=bool)
    keep[np.arange(height), seam] = False

    # arr[keep] flattens the kept pixels row by row, so reshaping
    # back to (H, W-1, ...) puts every pixel in its correct place.
    return arr[keep].reshape(height, width - 1, *arr.shape[2:])


def carve(img, new_width, new_height, finder, on_seam=None):
    """Shrink img to new_width x new_height by removing low-energy seams.

    finder:  function(energy) -> seam, e.g. dp.find_vertical_seam
    on_seam: optional callback for visualisation, called once per seam as
             on_seam(step, direction, current_img, seam_mask, orig_rows, orig_cols)
               current_img        image before this seam is removed
               seam_mask          H x W bool, True on the seam pixels
               orig_rows/orig_cols  where the removed pixels were in the ORIGINAL image
    """
    height, width = img.shape[:2]
    _check_target(new_width, width, "width")
    _check_target(new_height, height, "height")

    # Index maps: for every current pixel, its (row, col) in the original
    # image. Seams are removed from these too, so they stay in sync.
    orig_rows, orig_cols = np.indices((height, width))
    state = {"step": 0}

    # 1) Vertical seams shrink the width.
    img, orig_rows, orig_cols = _remove_seams(
        img, orig_rows, orig_cols, width - new_width,
        finder, on_seam, "vertical", state)

    # 2) Horizontal seams: transpose so rows become columns, remove
    #    "vertical" seams, then transpose back.
    img_t, rows_t, cols_t = _remove_seams(
        img.transpose(1, 0, 2), orig_rows.T, orig_cols.T, height - new_height,
        finder, on_seam, "horizontal", state)

    return np.ascontiguousarray(img_t.transpose(1, 0, 2))


def _remove_seams(img, orig_rows, orig_cols, count, finder, on_seam, direction, state):
    """Find and remove `count` vertical seams from img (and the index maps)."""
    for _ in range(count):
        # Energy is recomputed every time, as the brief requires (Item 1c):
        # removing a seam changes the gradients next to it.
        energy = compute_energy(img)
        seam = np.asarray(finder(energy))

        if not is_valid_seam(seam, *energy.shape):
            raise ValueError(
                f"{getattr(finder, '__module__', finder)} returned an invalid seam")

        if on_seam is not None:
            _notify(on_seam, state["step"], direction, img, seam, orig_rows, orig_cols)

        img = remove_vertical_seam(img, seam)
        orig_rows = remove_vertical_seam(orig_rows, seam)
        orig_cols = remove_vertical_seam(orig_cols, seam)
        state["step"] += 1

    return img, orig_rows, orig_cols


def _notify(on_seam, step, direction, img, seam, orig_rows, orig_cols):
    """Call on_seam with everything in normal (non-transposed) orientation."""
    rows = np.arange(len(seam))
    mask = np.zeros(img.shape[:2], dtype=bool)
    mask[rows, seam] = True

    removed_rows = orig_rows[rows, seam]
    removed_cols = orig_cols[rows, seam]

    if direction == "horizontal":
        img = img.transpose(1, 0, 2)
        mask = mask.T

    on_seam(step, direction, img, mask, removed_rows, removed_cols)


def _check_target(target, current, name):
    """Only shrinking is supported; enlarging is not part of the brief."""
    if not 2 <= target <= current:
        raise ValueError(f"new {name} must be between 2 and {current}, got {target}")
