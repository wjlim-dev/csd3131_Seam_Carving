"""Greedy, row by row (brief Item 3). Owner: P4.

Contract (see README "Shared interface"):
    find_vertical_seam(energy) -> seam
    energy: H x W float array, all values >= 0
    seam:   int array of shape (H,), seam[i] = column removed in row i,
            with |seam[i] - seam[i-1]| <= 1

Test without the rest of the pipeline, e.g.:
    energy = np.random.default_rng(0).random((20, 30))
"""

import numpy as np


def find_vertical_seam(energy):
    # energy is a grid of numbers: energy[row][col]. A low number means "this pixel is safe to remove".
    num_rows = energy.shape[0]
    num_cols = energy.shape[1]

    # seam will hold one column number for each row, top to bottom.
    seam = []

    # Step 1: the top row
    # Nothing comes before the top row, so just pick its cheapest pixel.
    best_col = 0
    for col in range(num_cols):
        if energy[0][col] < energy[0][best_col]:
            best_col = col
    seam.append(best_col)

    # Step 2: every row below
    # From where we are now, we may only step down-left, straight down,
    # or down-right. Pick whichever of those is cheapest.
    for row in range(1, num_rows):
        current_col = seam[-1]  # the column we chose in the row above

        # The columns we are allowed to move to.
        options = [current_col - 1, current_col, current_col + 1]

        best_col = None
        for col in options:
            # Skip columns that fall off the left or right edge.
            if col < 0 or col >= num_cols:
                continue
            # Keep this column if it is the first one, or cheaper than the best so far.
            if best_col is None or energy[row][col] < energy[row][best_col]:
                best_col = col

        seam.append(best_col)

    # The rest of the program expects a numpy array of whole numbers.
    return np.array(seam, dtype=int)
