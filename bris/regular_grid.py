"""Largest complete regular rectangle of a set of points in a map projection

Used by the netcdf output to write a point cloud that is a (cropped, non-rectangular) subset
of a projected regular grid as a 2-D x/y field: the output is the largest complete rectangle
of grid points, the same box anemoi-training uses for the FFT losses
(anemoi.graphs.nodes.attributes.RegularGridIndices). The rectangle is either read from that
node attribute of the checkpoint graph (netcdf grid_indices_attribute option,
subset_from_grid_indices) or recomputed from the points (regular_grid option,
find_largest_regular_grid, a port of anemoi.graphs.generate.regular_grid, so the same
projection, spacing and tolerances give the same rectangle as the training graph attribute).
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class RegularGridSubset:
    """The largest complete regular rectangle found in projected coordinates

    Attributes:
        node_indices: Indices (into the input points) of the rectangle points in row-major
            order (rows of increasing y, columns of increasing x), so that
            values[node_indices].reshape(shape) is the gridded field.
        shape: (ny, nx)
        x: Projected x coordinate of the nx columns
        y: Projected y coordinate of the ny rows
        spacing: (x_spacing, y_spacing) in projected units
    """

    node_indices: np.ndarray
    shape: tuple[int, int]
    x: np.ndarray
    y: np.ndarray
    spacing: tuple[float, float]


def _infer_axis_spacing(
    coordinates: np.ndarray,
    axis: int,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> float:
    """Infer the lattice spacing along one axis from the nearest axis-aligned neighbours

    For each point the smallest displacement to a neighbour that is aligned with the axis
    (cross-axis displacement within tolerance) is a candidate; the spacing is the median of
    the candidates.
    """
    neighbours = min(9, len(coordinates))
    _, indices = cKDTree(coordinates).query(coordinates, k=neighbours)
    delta = np.abs(coordinates[indices[:, 1:]] - coordinates[:, None, :])
    axis_delta = delta[..., axis]
    cross_delta = delta[..., 1 - axis]
    aligned = (axis_delta > absolute_tolerance) & (
        cross_delta <= np.maximum(absolute_tolerance, relative_tolerance * axis_delta)
    )
    candidates = np.where(aligned, axis_delta, np.inf).min(axis=1)
    candidates = candidates[np.isfinite(candidates)]
    if len(candidates) == 0:
        raise ValueError(f"Could not infer projected grid spacing along axis {axis}.")

    spacing = float(np.median(candidates))
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError(
            f"Inferred invalid projected grid spacing {spacing} along axis {axis}."
        )
    return spacing


def _largest_complete_rectangle(occupancy: np.ndarray) -> tuple[int, int, int, int]:
    """Return inclusive-exclusive (top, bottom, left, right) bounds of the largest all-true
    rectangle (maximal histogram algorithm; ties broken as in anemoi-graphs)"""
    heights = np.zeros(occupancy.shape[1], dtype=np.int64)
    best: tuple[int, int, int, int] | None = None
    best_key: tuple[int, int, int, int, int] | None = None

    for bottom in range(occupancy.shape[0]):
        heights = np.where(occupancy[bottom], heights + 1, 0)
        stack: list[tuple[int, int]] = []
        for column in range(occupancy.shape[1] + 1):
            height = int(heights[column]) if column < occupancy.shape[1] else 0
            start = column
            while stack and stack[-1][1] > height:
                left, rectangle_height = stack.pop()
                start = left
                top = bottom - rectangle_height + 1
                right = column
                point_count = rectangle_height * (right - left)
                interior_area = (rectangle_height - 1) * (right - left - 1)
                key = (point_count, interior_area, -top, -left, rectangle_height)
                if (
                    rectangle_height >= 2
                    and right - left >= 2
                    and (best_key is None or key > best_key)
                ):
                    best_key = key
                    best = (top, bottom + 1, left, right)
            if not stack or stack[-1][1] < height:
                stack.append((start, height))

    if best is None:
        raise ValueError(
            "Projected points do not contain a complete regular rectangle of at least 2 by 2 points."
        )
    return best


def find_largest_regular_grid(
    projected_coordinates: np.ndarray,
    *,
    x_spacing: float | None = None,
    y_spacing: float | None = None,
    relative_tolerance: float = 1.0e-4,
    absolute_tolerance: float = 1.0e-6,
    maximum_grid_cells: int = 100_000_000,
) -> RegularGridSubset:
    """Find the complete axis-aligned regular rectangle containing the most points

    Args:
        projected_coordinates: Projected (x, y) point coordinates with shape (N, 2)
        x_spacing, y_spacing: Grid spacing in projected units. Inferred from the nearest
            axis-aligned neighbours when None.
        relative_tolerance, absolute_tolerance: Maximum residual of a point from the lattice,
            as a fraction of the spacing and in projected units (the larger applies)
        maximum_grid_cells: Safety limit for the lattice occupancy matrix
    """
    coordinates = np.asarray(projected_coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(
            f"projected_coordinates must have shape (N, 2), got {coordinates.shape}."
        )
    if len(coordinates) < 4:
        raise ValueError(
            "At least four projected points are required to find a 2D regular rectangle."
        )
    if not np.isfinite(coordinates).all():
        raise ValueError("Projected coordinates must contain only finite values.")
    if relative_tolerance < 0 or absolute_tolerance < 0:
        raise ValueError("Grid tolerances must be non-negative.")
    if maximum_grid_cells <= 0:
        raise ValueError("maximum_grid_cells must be positive.")

    spacing_x = (
        _infer_axis_spacing(coordinates, 0, relative_tolerance, absolute_tolerance)
        if x_spacing is None
        else float(x_spacing)
    )
    spacing_y = (
        _infer_axis_spacing(coordinates, 1, relative_tolerance, absolute_tolerance)
        if y_spacing is None
        else float(y_spacing)
    )
    spacing = np.array([spacing_x, spacing_y])
    if not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("Projected grid spacing must be finite and positive.")

    origin = coordinates.min(axis=0)
    lattice = np.rint((coordinates - origin) / spacing).astype(np.int64)
    reconstructed = origin + lattice * spacing
    tolerance = np.maximum(absolute_tolerance, relative_tolerance * spacing)
    residual = np.abs(coordinates - reconstructed)
    if np.any(residual > tolerance):
        worst = np.unravel_index(np.argmax(residual / tolerance), residual.shape)
        raise ValueError(
            "Projected coordinates do not lie on one regular lattice within tolerance: "
            f"point {worst[0]}, axis {worst[1]}, residual {residual[worst]:.6g}."
        )

    lattice -= lattice.min(axis=0)
    n_columns = int(lattice[:, 0].max()) + 1
    n_rows = int(lattice[:, 1].max()) + 1
    if n_rows * n_columns > maximum_grid_cells:
        raise ValueError(
            f"Inferred occupancy grid has {n_rows * n_columns} cells, exceeding "
            f"maximum_grid_cells={maximum_grid_cells}. Check the spacing and tolerance settings."
        )

    cells = lattice[:, 1] * n_columns + lattice[:, 0]
    if len(np.unique(cells)) != len(cells):
        raise ValueError("Multiple projected points occupy the same lattice cell.")
    occupancy = np.zeros(n_rows * n_columns, dtype=bool)
    occupancy[cells] = True
    source_indices = np.full(n_rows * n_columns, -1, dtype=np.int64)
    source_indices[cells] = np.arange(len(cells))
    occupancy = occupancy.reshape(n_rows, n_columns)
    source_indices = source_indices.reshape(n_rows, n_columns)

    top, bottom, left, right = _largest_complete_rectangle(occupancy)
    selected = source_indices[top:bottom, left:right].reshape(-1)
    # Axes from the actual projected coordinates of the rectangle points (as
    # subset_from_grid_indices), not from the lattice: residuals are within tolerance anyway
    shape = (bottom - top, right - left)
    x = coordinates[selected, 0].reshape(shape).mean(axis=0)
    y = coordinates[selected, 1].reshape(shape).mean(axis=1)
    return RegularGridSubset(
        node_indices=selected,
        shape=shape,
        x=x,
        y=y,
        spacing=(spacing_x, spacing_y),
    )


def subset_from_grid_indices(
    grid_indices: np.ndarray, x: np.ndarray, y: np.ndarray
) -> RegularGridSubset:
    """The rectangle encoded by (row, column) grid indices per point, as the RegularGridIndices
    node attribute of an anemoi graph stores it: -1 for points outside the rectangle

    Args:
        grid_indices: (N, 2) integer array of (row, column) per point, -1 outside
        x, y: Projected coordinates of the N points
    """
    grid_indices = np.asarray(grid_indices)
    if (
        grid_indices.ndim != 2
        or grid_indices.shape[1] != 2
        or len(grid_indices) != len(x)
    ):
        raise ValueError(
            f"grid_indices must have shape ({len(x)}, 2), got {grid_indices.shape}"
        )
    rows = grid_indices[:, 0].astype(np.int64)
    columns = grid_indices[:, 1].astype(np.int64)
    selected = np.flatnonzero((rows >= 0) & (columns >= 0))
    if len(selected) == 0:
        raise ValueError("grid_indices selects no points")
    rows, columns = rows[selected], columns[selected]
    n_rows, n_columns = int(rows.max()) + 1, int(columns.max()) + 1
    cells = rows * n_columns + columns
    if len(np.unique(cells)) != len(cells) or len(cells) != n_rows * n_columns:
        raise ValueError(
            f"grid_indices do not form a complete {n_rows} x {n_columns} rectangle "
            f"({len(cells)} points)"
        )
    # Row-major order (rows of increasing y, columns of increasing x)
    node_indices = selected[np.argsort(cells)]
    xx = np.asarray(x, dtype=np.float64)[node_indices].reshape(n_rows, n_columns)
    yy = np.asarray(y, dtype=np.float64)[node_indices].reshape(n_rows, n_columns)
    x_axis = xx.mean(axis=0)
    y_axis = yy.mean(axis=1)
    if n_columns < 2 or n_rows < 2:
        raise ValueError("grid_indices rectangle must be at least 2 by 2 points")
    spacing_x = float(np.median(np.diff(x_axis)))
    spacing_y = float(np.median(np.diff(y_axis)))
    if spacing_x <= 0 or spacing_y <= 0:
        raise ValueError(
            "grid_indices rows/columns are not ordered with increasing projected y/x: "
            "wrong projection?"
        )
    for name, values, axis in [("x", xx, x_axis[None, :]), ("y", yy, y_axis[:, None])]:
        worst = np.abs(values - axis).max()
        if worst > 0.01 * (spacing_x if name == "x" else spacing_y):
            raise ValueError(
                f"grid_indices points are not on a regular {name} lattice in this projection "
                f"(max deviation {worst:.3g}): wrong projection?"
            )
    return RegularGridSubset(
        node_indices=node_indices,
        shape=(n_rows, n_columns),
        x=x_axis,
        y=y_axis,
        spacing=(spacing_x, spacing_y),
    )


@dataclass(frozen=True)
class BoundingGridSubset:
    """The regular projected grid that contains all the grid (LAM) points, with the cells
    that hold no grid point either empty or filled from the nearest other (global) point

    Attributes:
        cell_source: (ny * nx,) index into the input points of the point written in each
            cell, row-major (rows of increasing y, columns of increasing x); -1 for an empty
            cell. values[..., cell_source] with -1 -> NaN, reshaped to shape, is the field.
        shape: (ny, nx)
        x: Projected x coordinate of the nx columns
        y: Projected y coordinate of the ny rows
        spacing: (x_spacing, y_spacing) in projected units
        num_filled: Number of cells filled from the other points
    """

    cell_source: np.ndarray
    shape: tuple[int, int]
    x: np.ndarray
    y: np.ndarray
    spacing: tuple[float, float]
    num_filled: int


def bounding_grid_subset(
    projected_coordinates: np.ndarray,
    grid_mask: np.ndarray,
    *,
    x_spacing: float | None = None,
    y_spacing: float | None = None,
    fill: str = "nan",
    relative_tolerance: float = 1.0e-4,
    absolute_tolerance: float = 1.0e-6,
    fill_margin_cells: int = 50,
    maximum_grid_cells: int = 100_000_000,
) -> BoundingGridSubset:
    """The axis-aligned regular rectangle that contains all the grid points

    The grid points (grid_mask True) must lie on one regular lattice in the projection (a
    cropped, non-rectangular subset of a projected grid, e.g. a lat-cropped LAM domain). The
    output is the smallest rectangle of that lattice holding all of them. Cells without a
    grid point are empty (fill="nan") or take the nearest other point (fill="global": the
    points with grid_mask False, e.g. the global part of a stretched grid, nearest neighbour
    in the projection).

    Args:
        projected_coordinates: Projected (x, y) coordinates of all points, shape (N, 2)
        grid_mask: (N,) boolean, True for the points that define the grid
        x_spacing, y_spacing: Lattice spacing in projected units; inferred from the grid
            points when None
        fill: "nan" or "global"
        relative_tolerance, absolute_tolerance: Maximum residual of a grid point from the
            lattice, as a fraction of the spacing and in projected units (the larger applies)
        fill_margin_cells: Other points further than this many cells outside the rectangle
            are not used for filling
        maximum_grid_cells: Safety limit for the rectangle size
    """
    coordinates = np.asarray(projected_coordinates, dtype=np.float64)
    grid_mask = np.asarray(grid_mask, dtype=bool)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(
            f"projected_coordinates must have shape (N, 2), got {coordinates.shape}."
        )
    if grid_mask.shape != (len(coordinates),):
        raise ValueError(
            f"grid_mask must have shape ({len(coordinates)},), got {grid_mask.shape}."
        )
    if fill not in ("nan", "global"):
        raise ValueError(f"fill must be 'nan' or 'global', got {fill!r}.")
    if relative_tolerance < 0 or absolute_tolerance < 0:
        raise ValueError("Grid tolerances must be non-negative.")
    grid_indices = np.flatnonzero(grid_mask)
    grid_points = coordinates[grid_indices]
    if len(grid_points) < 4:
        raise ValueError("At least four grid points are required.")
    if not np.isfinite(grid_points).all():
        raise ValueError("Projected grid point coordinates must be finite.")

    spacing_x = (
        _infer_axis_spacing(grid_points, 0, relative_tolerance, absolute_tolerance)
        if x_spacing is None
        else float(x_spacing)
    )
    spacing_y = (
        _infer_axis_spacing(grid_points, 1, relative_tolerance, absolute_tolerance)
        if y_spacing is None
        else float(y_spacing)
    )
    spacing = np.array([spacing_x, spacing_y])
    if not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("Projected grid spacing must be finite and positive.")

    origin = grid_points.min(axis=0)
    lattice = np.rint((grid_points - origin) / spacing).astype(np.int64)
    tolerance = np.maximum(absolute_tolerance, relative_tolerance * spacing)
    residual = np.abs(grid_points - (origin + lattice * spacing))
    if np.any(residual > tolerance):
        worst = np.unravel_index(np.argmax(residual / tolerance), residual.shape)
        raise ValueError(
            "Projected grid points do not lie on one regular lattice within tolerance: "
            f"point {grid_indices[worst[0]]}, axis {worst[1]}, residual {residual[worst]:.6g}."
        )
    n_columns = int(lattice[:, 0].max()) + 1
    n_rows = int(lattice[:, 1].max()) + 1
    if n_rows * n_columns > maximum_grid_cells:
        raise ValueError(
            f"The bounding grid has {n_rows * n_columns} cells, exceeding "
            f"maximum_grid_cells={maximum_grid_cells}. Check the spacing and tolerance settings."
        )
    cells = lattice[:, 1] * n_columns + lattice[:, 0]
    if len(np.unique(cells)) != len(cells):
        raise ValueError("Multiple projected grid points occupy the same lattice cell.")
    cell_source = np.full(n_rows * n_columns, -1, dtype=np.int64)
    cell_source[cells] = grid_indices

    num_filled = 0
    empty = np.flatnonzero(cell_source < 0)
    if fill == "global" and len(empty) > 0:
        others = np.flatnonzero(~grid_mask)
        if len(others) == 0:
            raise ValueError(
                "fill='global' needs points outside the grid (the output must hold the whole "
                "stretched grid, not only the LAM domain)."
            )
        other_xy = coordinates[others]
        margin = fill_margin_cells * spacing
        lower = origin - margin
        upper = origin + (np.array([n_columns, n_rows]) - 1) * spacing + margin
        usable = (
            np.isfinite(other_xy).all(axis=1)
            & (other_xy >= lower).all(axis=1)
            & (other_xy <= upper).all(axis=1)
        )
        if not usable.any():
            raise ValueError(
                "fill='global': no points outside the grid project into the bounding rectangle "
                f"(plus a margin of {fill_margin_cells} cells)."
            )
        rows, columns = np.divmod(empty, n_columns)
        centres = origin + np.column_stack((columns, rows)) * spacing
        _, nearest = cKDTree(other_xy[usable]).query(centres)
        cell_source[empty] = others[usable][nearest]
        num_filled = len(empty)

    return BoundingGridSubset(
        cell_source=cell_source,
        shape=(n_rows, n_columns),
        x=origin[0] + np.arange(n_columns) * spacing_x,
        y=origin[1] + np.arange(n_rows) * spacing_y,
        spacing=(spacing_x, spacing_y),
        num_filled=num_filled,
    )
