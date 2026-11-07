"""Checks every seam finder follows the shared interface.

Unfinished finders (still raising NotImplementedError) are skipped, so
this file starts testing your finder automatically once you write it.
"""

import numpy as np
import pytest

from seam_carving.finders import FINDERS, get_finder
from seam_carving.seam_utils import is_valid_seam

# Graph cut is slow, so keep the arrays small for every finder.
SHAPES = [(1, 5), (5, 1), (6, 8), (12, 15)]


@pytest.mark.parametrize("name", list(FINDERS))
@pytest.mark.parametrize("shape", SHAPES)
def test_finder_returns_valid_seam(name, shape):
    finder = get_finder(name)
    energy = np.random.default_rng(42).random(shape)

    try:
        seam = finder(energy)
    except NotImplementedError:
        pytest.skip(f"{name} not implemented yet")

    assert is_valid_seam(np.asarray(seam), *shape)
