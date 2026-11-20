"""Seam visualisation (bonus Item 7a).

Uses the on_seam hook in core.carve() to draw seams as they are removed.
"""

import numpy as np

SEAM_COLOUR = (255, 0, 0)  # red, images in this project are RGB


def overlay_seam(img, mask, colour=SEAM_COLOUR):
    """Return a copy of img with the seam pixels painted in colour.

    img:  H x W x 3 uint8 RGB image (the image BEFORE the seam is removed)
    mask: H x W bool array, True on the seam pixels
    """
    out = img.copy()      # copy so we never change the image carve() is using
    out[mask] = colour    # boolean indexing: paint every pixel where mask is True
    return out


VERTICAL_COLOUR = (255, 0, 0)      # red: seams that shrink the width
HORIZONTAL_COLOUR = (0, 120, 255)  # blue: seams that shrink the height


class SeamRecorder:
    """Pass an instance as carve(..., on_seam=recorder).

    carve() calls it once per seam. It remembers where every removed
    seam was in the ORIGINAL image, and (optionally) keeps a frame of
    each step with the seam drawn on it, for the step-by-step GIF.
    """

    def __init__(self, keep_frames=True):
        self.keep_frames = keep_frames
        self.removed = []  # list of (direction, orig_rows, orig_cols)
        self.frames = []   # one RGB image per seam, seam drawn in colour

    def __call__(self, step, direction, current_img, seam_mask, orig_rows, orig_cols):
        # orig_rows/orig_cols are already in original-image coordinates
        # (core.carve tracks this with its index maps), so we just store them.
        self.removed.append((direction, orig_rows.copy(), orig_cols.copy()))

        if self.keep_frames:
            colour = VERTICAL_COLOUR if direction == "vertical" else HORIZONTAL_COLOUR
            self.frames.append(overlay_seam(current_img, seam_mask, colour))


def all_seams_on_original(orig_img, removed):
    """Return a copy of the original image with every removed seam drawn on it.

    orig_img: the image BEFORE any carving
    removed:  SeamRecorder.removed, a list of (direction, orig_rows, orig_cols)
    """
    out = orig_img.copy()
    for direction, rows, cols in removed:
        colour = VERTICAL_COLOUR if direction == "vertical" else HORIZONTAL_COLOUR
        out[rows, cols] = colour  # fancy indexing: paint pixel (rows[k], cols[k]) for every k
    return out


def pad_to(img, height, width, fill=255):
    """Place img in the top-left corner of a height x width canvas.

    The image gets smaller after every seam, but every frame of a GIF
    must be the same size, so we pad each frame back up to the original
    size (white by default).
    """
    canvas = np.full((height, width, 3), fill, dtype=np.uint8)
    h, w = img.shape[:2]
    canvas[:h, :w] = img
    return canvas


def make_gif(frames, path, every=1, scale=1, duration_ms=60,
             final_frame=None, hold_ms=1500):
    """Save the step-by-step frames as an animated GIF.

    frames:      SeamRecorder.frames (one image per seam, seam drawn on it)
    every:       keep only every Nth frame, to keep the file small
    scale:       enlarge small images by this factor (nearest-neighbour,
                 so seams stay sharp 1-pixel lines instead of blurring)
    duration_ms: how long each step is shown
    final_frame: optional carved result, shown at the end for hold_ms
    """
    from PIL import Image  # Pillow, installed with imageio

    if not frames:
        raise ValueError("no frames to save: did carve() remove any seams?")

    chosen = list(frames[::every])
    durations = [duration_ms] * len(chosen)
    if final_frame is not None:
        chosen.append(final_frame)
        durations.append(hold_ms)

    # The first frame is the biggest (nothing removed yet), so use its size.
    height, width = frames[0].shape[:2]

    pil_frames = []
    for frame in chosen:
        padded = pad_to(frame, height, width)
        if scale != 1:
            padded = np.repeat(np.repeat(padded, scale, axis=0), scale, axis=1)
        pil_frames.append(Image.fromarray(padded))

    pil_frames[0].save(path, save_all=True, append_images=pil_frames[1:],
                       duration=durations, loop=0)  # loop=0: repeat forever
