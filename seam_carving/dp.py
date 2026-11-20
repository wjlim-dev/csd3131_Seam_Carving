"""Dynamic programming (brief Item 1b). Owner: P2.

Contract (see README "Shared interface"):
    find_vertical_seam(energy) -> seam
    energy: H x W float array, all values >= 0
    seam:   int array of shape (H,), seam[i] = column removed in row i,
            with |seam[i] - seam[i-1]| <= 1

Test without the rest of the pipeline, e.g.:
    energy = np.random.default_rng(0).random((20, 30))

Recurrence (vertical seam, rows top to bottom):
    M[0, j] = e[0, j]
    M[i, j] = e[i, j] + min(M[i-1, j-1], M[i-1, j], M[i-1, j+1])
Out-of-range neighbours count as infinity.
"""

import numpy as np


def find_vertical_seam(energy: np.ndarray) -> np.ndarray:
    """Return the minimum-energy vertical seam.

    energy: H x W float array, all values >= 0.
    Returns shape (H,), dtype int; seam[row] = column removed in that row.
    """
    height, width = energy.shape

    # Item 5d improvement: costs only need the previous row, so we keep one
    # row of M (Theta(W)) instead of the full table. Backpointers still need
    # Theta(HW) to rebuild the seam, but int8 is 8x smaller than float64.
    previous_row_cost = energy[0].astype(np.float64)
    backpointer_offsets = np.zeros((height, width), dtype=np.int8)
    column_indices = np.arange(width)

    for row in range(1, height):
        # Pad with inf so edge pixels never pick an out-of-range neighbour.
        padded_cost = np.pad(previous_row_cost, 1, constant_values=np.inf)

        # Row 0 = up-left, row 1 = straight up, row 2 = up-right. Shape (3, W).
        neighbour_costs = np.stack(
            (padded_cost[:-2], padded_cost[1:-1], padded_cost[2:])
        )

        # All cells in one row are independent, so NumPy handles the whole
        # row at once. Rows still have to go in order.
        best_offset = np.argmin(neighbour_costs, axis=0)
        backpointer_offsets[row] = best_offset
        previous_row_cost = energy[row] + neighbour_costs[best_offset, column_indices]

    # Backtrack: start at the cheapest pixel in the bottom row and walk up.
    seam = np.empty(height, dtype=np.int64)
    seam[-1] = np.argmin(previous_row_cost)
    for row in range(height - 1, 0, -1):
        # Offset 0/1/2 means the parent column is -1/0/+1 from this one.
        seam[row - 1] = seam[row] + backpointer_offsets[row, seam[row]] - 1

    return seam
