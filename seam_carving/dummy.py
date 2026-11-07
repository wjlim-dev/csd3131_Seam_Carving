"""Placeholder finder so the pipeline can run before DP/greedy/graph cut exist.

Picks the straight vertical line (one column, top to bottom) with the lowest
total energy. It is a valid seam, just not a good one, so it is also a
handy "worst reasonable" baseline in comparisons.
"""

import numpy as np


def find_vertical_seam(energy):
    height = energy.shape[0]
    best_col = int(np.argmin(energy.sum(axis=0)))
    return np.full(height, best_col, dtype=int)
