"""Graph cut / min-cut max-flow (brief Item 4). Owner: P5.

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
    raise NotImplementedError("P5: graphcut.find_vertical_seam is not written yet")
