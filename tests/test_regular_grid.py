import numpy as np
import pytest

from bris.regular_grid import find_largest_regular_grid, subset_from_grid_indices


def lattice_points(nx, ny, dx=2500.0, dy=2500.0, x0=1000.0, y0=-500.0):
    xx, yy = np.meshgrid(x0 + np.arange(nx) * dx, y0 + np.arange(ny) * dy)
    return np.column_stack((xx.ravel(), yy.ravel()))


def test_full_rectangle():
    points = lattice_points(6, 4)
    subset = find_largest_regular_grid(points, x_spacing=2500, y_spacing=2500)
    assert subset.shape == (4, 6)
    assert subset.spacing == (2500.0, 2500.0)
    np.testing.assert_allclose(subset.x, 1000.0 + np.arange(6) * 2500.0)
    np.testing.assert_allclose(subset.y, -500.0 + np.arange(4) * 2500.0)
    # Row-major: values[node_indices].reshape(shape) reproduces the grid
    np.testing.assert_array_equal(
        points[subset.node_indices, 0].reshape(subset.shape), np.tile(subset.x, (4, 1))
    )
    np.testing.assert_array_equal(
        points[subset.node_indices, 1].reshape(subset.shape),
        np.tile(subset.y[:, None], (1, 6)),
    )


def test_cropped_domain_and_inferred_spacing():
    # A 10 x 8 lattice with the two upper corners removed (a lat-cropped domain): the
    # largest rectangle is the lower 6 rows x 10 columns (60 points), not 8 rows x 6 cols (48)
    points = lattice_points(10, 8)
    xi = np.rint((points[:, 0] - 1000.0) / 2500.0)
    yi = np.rint((points[:, 1] + 500.0) / 2500.0)
    keep = ~((yi >= 6) & ((xi < 2) | (xi > 7)))
    points = points[keep][np.random.default_rng(0).permutation(keep.sum())]

    for kwargs in [{}, {"x_spacing": 2500, "y_spacing": 2500}]:
        subset = find_largest_regular_grid(points, **kwargs)
        assert subset.shape == (6, 10)
        assert subset.spacing == (2500.0, 2500.0)
        np.testing.assert_allclose(subset.y, -500.0 + np.arange(6) * 2500.0)
        sel = points[subset.node_indices]
        np.testing.assert_allclose(sel[:, 0].reshape(6, 10)[0], subset.x)
        np.testing.assert_allclose(sel[:, 1].reshape(6, 10)[:, 0], subset.y)


def test_tolerance():
    points = lattice_points(5, 5)
    # Residuals are relative to the (also perturbed) origin point, so up to 6 m
    points += np.random.default_rng(1).uniform(-3.0, 3.0, points.shape)
    subset = find_largest_regular_grid(
        points, x_spacing=2500, y_spacing=2500, absolute_tolerance=7.0
    )
    assert subset.shape == (5, 5)
    with pytest.raises(ValueError, match="lattice"):
        find_largest_regular_grid(
            points, x_spacing=2500, y_spacing=2500, absolute_tolerance=1.0
        )


def test_errors():
    with pytest.raises(ValueError):
        find_largest_regular_grid(np.zeros((3, 2)))
    with pytest.raises(ValueError, match="same lattice cell"):
        find_largest_regular_grid(
            np.array([[0, 0], [0, 0], [1, 0], [0, 1]]), x_spacing=1, y_spacing=1
        )
    with pytest.raises(ValueError, match="2 by 2"):
        # A single row
        find_largest_regular_grid(lattice_points(5, 1), x_spacing=2500, y_spacing=2500)


def test_subset_from_grid_indices():
    # The same cropped domain, with the rectangle given as (row, column) indices in a shuffled order
    points = lattice_points(10, 8)
    xi = np.rint((points[:, 0] - 1000.0) / 2500.0).astype(int)
    yi = np.rint((points[:, 1] + 500.0) / 2500.0).astype(int)
    keep = ~((yi >= 6) & ((xi < 2) | (xi > 7)))
    perm = np.random.default_rng(0).permutation(keep.sum())
    points, xi, yi = points[keep][perm], xi[keep][perm], yi[keep][perm]
    indices = np.full((len(points), 2), -1, dtype=np.int64)
    inside = yi < 6
    indices[inside, 0] = yi[inside]
    indices[inside, 1] = xi[inside]

    subset = subset_from_grid_indices(indices, points[:, 0], points[:, 1])
    reference = find_largest_regular_grid(points, x_spacing=2500, y_spacing=2500)
    assert subset.shape == reference.shape == (6, 10)
    np.testing.assert_array_equal(subset.node_indices, reference.node_indices)
    np.testing.assert_allclose(subset.x, reference.x)
    np.testing.assert_allclose(subset.y, reference.y)
    assert subset.spacing == (2500.0, 2500.0)

    # Incomplete rectangle
    bad = indices.copy()
    bad[np.flatnonzero(inside)[0]] = -1
    with pytest.raises(ValueError, match="complete"):
        subset_from_grid_indices(bad, points[:, 0], points[:, 1])
    # Wrong projection: rows not increasing in y
    with pytest.raises(ValueError, match="projection"):
        subset_from_grid_indices(indices, points[:, 0], -points[:, 1])
