"""Energy map: how "important" each pixel is.

High energy = strong edge (removing it would be noticeable).
Low energy  = flat area (safe to remove).
"""

import cv2
import numpy as np


def compute_energy(img):
    """Return an H x W float64 energy map using the Sobel gradient.

    energy = |dI/dx| + |dI/dy| on the grayscale image.
    The brief allows OpenCV's Sobel filter, so we use it instead of
    writing the convolution ourselves.
    """
    # cv2 needs a contiguous array; a transposed view (used for
    # horizontal seams) is not contiguous, so copy it if needed.
    img = np.ascontiguousarray(img)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64)

    # 3x3 Sobel kernels: gx reacts to vertical edges, gy to horizontal ones.
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # L1 magnitude (|gx| + |gy|) is cheaper than sqrt(gx^2 + gy^2)
    # and ranks pixels almost the same way.
    return np.abs(gx) + np.abs(gy)
