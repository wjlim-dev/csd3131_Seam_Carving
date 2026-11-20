import itertools

import numpy as np
import pytest

from seam_carving.dp import find_vertical_seam
from seam_carving.seam_utils import is_valid_seam, seam_cost


def brute_force_min_cost(energy):
    """Try every possible seam. Only usable on tiny arrays (W * 3^(H-1) seams)."""
    height, width = energy.shape
    best_cost = np.inf
    for start_column in range(width):
        for steps in itertools.product((-1, 0, 1), repeat=height - 1):
            columns = [start_column]
            for step in steps:
                columns.append(columns[-1] + step)
            if all(0 <= column < width for column in columns):
                best_cost = min(best_cost, seam_cost(energy, np.array(columns)))
    return best_cost


@pytest.mark.parametrize("shape", [(1, 1), (1, 5), (5, 1), (2, 2), (20, 30), (40, 7)])
def test_seam_is_valid(shape):
    rng = np.random.default_rng(0)
    energy = rng.random(shape)
    seam = find_vertical_seam(energy)
    assert np.issubdtype(seam.dtype, np.integer)
    assert is_valid_seam(seam, *shape)


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("shape", [(4, 4), (5, 3), (3, 6)])
def test_matches_brute_force(seed, shape):
    rng = np.random.default_rng(seed)
    energy = rng.random(shape)
    seam = find_vertical_seam(energy)
    assert seam_cost(energy, seam) == pytest.approx(brute_force_min_cost(energy))


def test_follows_obvious_low_energy_path():
    energy = np.array([
        [9, 0, 9, 9],
        [9, 9, 0, 9],
        [9, 9, 9, 0],
        [9, 9, 0, 9],
    ], dtype=np.float64)
    assert find_vertical_seam(energy).tolist() == [1, 2, 3, 2]


def test_greedy_trap():
    # Greedy grabs the 0 in the top-left and walks into the 100s; DP avoids it.
    energy = np.array([
        [0, 1, 1],
        [100, 100, 1],
        [100, 100, 1],
    ], dtype=np.float64)
    seam = find_vertical_seam(energy)
    assert seam_cost(energy, seam) == 3
