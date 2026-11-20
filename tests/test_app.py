"""Tests for the plain (non-UI) helper functions in app.py."""

import numpy as np
import pytest

import app
from seam_carving.core import carve
from seam_carving.finders import get_finder


def random_image(height, width, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def test_png_round_trip():
    img = random_image(7, 9)
    assert (app.decode_image(app.encode_png(img)) == img).all()


def test_decode_rejects_non_image():
    with pytest.raises(ValueError):
        app.decode_image(b"not an image")


def test_limit_size_keeps_aspect_ratio():
    out = app.limit_size(random_image(100, 200), 50)
    assert out.shape[:2] == (25, 50)


def test_limit_size_leaves_small_images_alone():
    img = random_image(10, 20)
    assert app.limit_size(img, 50) is img


def test_run_carve_matches_plain_carve():
    img = random_image(12, 15)
    run = app.run_carve(img, "dp", 10, 9)
    expected = carve(img, 10, 9, get_finder("dp"))
    assert (run["result"] == expected).all()
    assert run["seams"] == 5 + 3
    assert len(run["frames"]) == 8
    assert run["total_energy"] >= 0


def test_dp_removes_no_more_energy_per_seam_than_dummy():
    # On the FIRST seam both see the same energy map, and DP is optimal,
    # so its seam can never cost more than dummy's straight line.
    img = random_image(12, 15, seed=1)
    dp = app.run_carve(img, "dp", 14, 12, keep_frames=False)
    dummy = app.run_carve(img, "dummy", 14, 12, keep_frames=False)
    assert dp["total_energy"] <= dummy["total_energy"]


def test_gif_bytes_is_a_gif():
    img = random_image(8, 10)
    run = app.run_carve(img, "dp", 7, 8)
    data = app.gif_bytes(run["frames"], run["result"], every=1, scale=2)
    assert data[:6] in (b"GIF87a", b"GIF89a")
