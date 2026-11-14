import numpy as np
import pytest

from seam_carving import dp
from seam_carving.greedy import find_vertical_seam
from seam_carving.seam_utils import is_valid_seam, seam_cost

@pytest.mark.parametrize("shape", [(3, 4), (10, 1), (1, 10), (25, 17)])
def test_valid_seam(shape):
    energy = np.random.default_rng(7).random(shape)
    assert is_valid_seam(find_vertical_seam(energy), *shape)

@pytest.mark.parametrize("seed", range(10))
def test_each_step_is_cheapest_neighbour(seed):
    energy = np.random.default_rng(seed).random((15, 10))
    seam = find_vertical_seam(energy)
    assert seam[0] == np.argmin(energy[0])
    for row in range(1, len(seam)):
        left = max(seam[row - 1] - 1, 0)
        right = min(seam[row - 1] + 1, energy.shape[1] - 1)
        assert energy[row, seam[row]] == energy[row, left:right + 1].min()

def test_hand_traced_example():
    # row 0 -> col 3 (1), then the cheapest of the 3 cells below each time
    energy = np.array([
        [5, 3, 8, 1, 7],
        [2, 6, 4, 9, 3],
        [7, 1, 5, 2, 8],
        [4, 8, 3, 6, 1],
        [9, 2, 7, 5, 4],
    ], dtype=float)
    assert find_vertical_seam(energy).tolist() == [3, 4, 3, 4, 4]

def test_runs_down_right_edge():
    energy = np.array([
        [9, 9, 9, 0],
        [9, 9, 9, 0],
        [9, 9, 9, 0],
    ], dtype=float)
    assert find_vertical_seam(energy).tolist() == [3, 3, 3]

def test_ties_choose_leftmost():
    assert find_vertical_seam(np.ones((4, 5))).tolist() == [0, 0, 0, 0]

def test_greedy_misses_cheaper_seam():
    # Greedy takes the 1 top-left and gets stuck in the 50s (cost 101),
    # but the seam [3, 3, 3] down the right side only costs 6.
    energy = np.array([
        [1, 2, 2, 2],
        [50, 50, 2, 2],
        [50, 50, 50, 2],
    ], dtype=float)
    seam = find_vertical_seam(energy)
    assert seam_cost(energy, seam) == 101
    assert seam_cost(energy, np.array([3, 3, 3])) == 6

def test_greedy_worse_than_dp_counterexample():
    # Greedy takes the 1 top-left (cheaper than the 2 next to it), but from
    # col 0 the 1s in col 2 are out of reach: [0, 0, 1] costs 1 + 8 + 8 = 17.
    # DP looks ahead and finds [1, 2, 2], costing 2 + 1 + 1 = 4.
    energy = np.array([
        [1, 2, 5],
        [8, 9, 1],
        [9, 8, 1],
    ], dtype=float)
    greedy_seam = find_vertical_seam(energy)
    dp_seam = dp.find_vertical_seam(energy)
    assert greedy_seam.tolist() == [0, 0, 1]
    assert seam_cost(energy, greedy_seam) == 17
    assert dp_seam.tolist() == [1, 2, 2]
    assert seam_cost(energy, dp_seam) == 4
