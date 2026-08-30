import numpy as np

from core.masking import apply_keep_mask, convex_hull_keep_mask


def test_convex_hull_mask_keeps_inside_and_masks_outside():
    x = [0.0, 1.0, 0.0]
    y = [0.0, 0.0, 1.0]
    grid_x = np.array([[0.25, 1.25]])
    grid_y = np.array([[0.25, 1.25]])
    keep = convex_hull_keep_mask(x, y, grid_x, grid_y)
    assert keep[0, 0]
    assert not keep[0, 1]


def test_apply_keep_mask_sets_outside_to_nan():
    z = np.array([[10.0, 20.0]])
    keep = np.array([[True, False]])
    masked = apply_keep_mask(z, keep)
    assert masked[0, 0] == 10.0
    assert np.isnan(masked[0, 1])

