"""Extra checks for the graph cut seam finder (P5).

The contract test only checks the seam is VALID. These also check it is the
CHEAPEST, by comparing its total energy against a small reference DP written
here (so this file does not depend on dp.py being finished).
"""

import numpy as np
import pytest

from seam_carving.graphcut import find_vertical_seam


def reference_min_seam_cost(energy):
    """Plain DP: M[i, j] = e[i, j] + min of the 3 cells above. Answer = min of last row."""
    M = energy.astype(float).copy()
    H, W = M.shape
    for i in range(1, H):
        for j in range(W):
            M[i, j] += M[i - 1, max(j - 1, 0):j + 2].min()
    return M[-1].min()


def seam_cost(energy, seam):
    return sum(energy[i, seam[i]] for i in range(len(seam)))


def test_worked_example():
    # The 2 x 3 example done by hand: seam [0, 1], cost 1 + 2 = 3.
    energy = np.array([[1.0, 4, 3], [5, 2, 6]])
    assert list(find_vertical_seam(energy)) == [0, 1]


def test_can_remove_last_column():
    # The cheapest seam is the last column. The brief's "INF to sink" version
    # could never pick it; ours must.
    energy = np.array([[5.0, 4, 1], [6, 3, 1]])
    assert list(find_vertical_seam(energy)) == [2, 2]


def test_single_column():
    energy = np.array([[2.0], [3.0], [1.0]])
    assert list(find_vertical_seam(energy)) == [0, 0, 0]


@pytest.mark.parametrize("seed", range(30))
def test_matches_dp_cost(seed):
    rng = np.random.default_rng(seed)
    H, W = rng.integers(1, 9, size=2)
    energy = rng.random((H, W))
    seam = find_vertical_seam(energy)
    # valid: right length, in range, connected
    assert seam.shape == (H,)
    assert seam.min() >= 0 and seam.max() < W
    assert np.all(np.abs(np.diff(seam)) <= 1)
    # optimal: same total energy as the DP minimum (seams may differ on ties)
    assert seam_cost(energy, seam) == pytest.approx(reference_min_seam_cost(energy))
