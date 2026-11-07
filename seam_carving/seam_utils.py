"""Small helpers every seam finder and test can share."""

import numpy as np


def is_valid_seam(seam, height, width):
    """True if seam has one column per row, stays in bounds,
    and moves at most one column between neighbouring rows."""
    seam = np.asarray(seam)

    if seam.shape != (height,):
        return False
    if not np.issubdtype(seam.dtype, np.integer):
        return False
    if seam.min() < 0 or seam.max() >= width:
        return False

    # Adjacent rows must be in the same or a neighbouring column.
    return bool(np.all(np.abs(np.diff(seam)) <= 1))


def seam_cost(energy, seam):
    """Total energy of the pixels on the seam (what DP minimises)."""
    rows = np.arange(energy.shape[0])
    return float(energy[rows, seam].sum())
