import numpy as np
import pytest

from seam_carving import dummy
from seam_carving.core import carve, remove_vertical_seam
from seam_carving.energy import compute_energy
from seam_carving.seam_utils import is_valid_seam, seam_cost


def random_image(height, width, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


# ---------- energy ----------

def test_energy_shape_and_type():
    energy = compute_energy(random_image(10, 12))
    assert energy.shape == (10, 12)
    assert energy.dtype == np.float64
    assert (energy >= 0).all()


def test_flat_image_has_zero_energy():
    flat = np.full((8, 8, 3), 120, dtype=np.uint8)
    assert np.allclose(compute_energy(flat), 0)


def test_edge_has_more_energy_than_flat_area():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:, 5:] = 255  # vertical edge between columns 4 and 5
    energy = compute_energy(img)
    assert energy[5, 4] > 0 and energy[5, 5] > 0
    assert energy[5, 0] == 0


def test_energy_accepts_transposed_view():
    img = random_image(6, 9)
    assert compute_energy(img.transpose(1, 0, 2)).shape == (9, 6)


# ---------- seam helpers ----------

def test_is_valid_seam():
    assert is_valid_seam(np.array([1, 2, 2, 1]), 4, 3)
    assert not is_valid_seam(np.array([0, 2, 2, 1]), 4, 3)   # jumps 2 columns
    assert not is_valid_seam(np.array([1, 2, 3, 2]), 4, 3)   # column 3 out of range
    assert not is_valid_seam(np.array([1, 1, 1]), 4, 3)      # wrong length


def test_seam_cost():
    energy = np.arange(12, dtype=float).reshape(3, 4)
    assert seam_cost(energy, np.array([0, 1, 2])) == 0 + 5 + 10


# ---------- seam removal ----------

def test_remove_vertical_seam_removes_the_right_pixels():
    arr = np.arange(12).reshape(3, 4)
    out = remove_vertical_seam(arr, np.array([0, 1, 3]))
    expected = np.array([[1, 2, 3],
                         [4, 6, 7],
                         [8, 9, 10]])
    assert np.array_equal(out, expected)


def test_remove_vertical_seam_keeps_colour_channels():
    img = random_image(5, 6)
    assert remove_vertical_seam(img, np.zeros(5, dtype=int)).shape == (5, 5, 3)


# ---------- carve loop ----------

@pytest.mark.parametrize("new_w, new_h", [(15, 12), (20, 9), (13, 10), (20, 12)])
def test_carve_output_size(new_w, new_h):
    out = carve(random_image(12, 20), new_w, new_h, dummy.find_vertical_seam)
    assert out.shape == (new_h, new_w, 3)


def test_carve_rejects_enlarging_and_tiny_targets():
    img = random_image(10, 10)
    with pytest.raises(ValueError):
        carve(img, 11, 10, dummy.find_vertical_seam)
    with pytest.raises(ValueError):
        carve(img, 1, 10, dummy.find_vertical_seam)


def test_carve_rejects_an_invalid_seam():
    def broken_finder(energy):
        return np.zeros(energy.shape[0] - 1, dtype=int)  # one row short

    with pytest.raises(ValueError):
        carve(random_image(8, 8), 7, 8, broken_finder)


def test_on_seam_reports_every_seam_in_original_coordinates():
    img = random_image(9, 11)
    events = []

    def record(step, direction, current_img, seam_mask, orig_rows, orig_cols):
        assert current_img.shape[:2] == seam_mask.shape
        events.append((step, direction, list(zip(orig_rows, orig_cols))))

    carve(img, 8, 7, dummy.find_vertical_seam, on_seam=record)

    assert [e[0] for e in events] == list(range(5))
    assert [e[1] for e in events] == ["vertical"] * 3 + ["horizontal"] * 2

    # Every removed pixel is inside the original image and none is removed twice.
    removed = [pixel for e in events for pixel in e[2]]
    assert len(removed) == len(set(removed))
    assert all(0 <= r < 9 and 0 <= c < 11 for r, c in removed)
