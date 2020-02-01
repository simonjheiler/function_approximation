import pdb  # noqa F401

import numpy as np
import pandas as pd
import scipy.interpolate as sp_interpolate
from interpolation.smolyak.grid import SmolyakGrid as sg
from interpolation.smolyak.interp import SmolyakInterp as si
from interpolation.splines import CubicSpline as spline

from src.auxiliary import get_corner_points
from src.auxiliary import get_local_grid
from src.pysg import sparseGrid
from src.sparsegrid import SparseInterpolator

#########################################################################
# FUNCTIONS
#########################################################################


def get_interpolation_grid_regular(dims, grid_min, grid_max, interp_params):

    # load interpolation parameters
    interpolation_points = interp_params["linear"]["interpolation points"]

    # generate grid
    grid_interp = []
    for idx in range(dims):
        grid_dim = np.linspace(grid_min[idx], grid_max[idx], interpolation_points)
        grid_interp.append(grid_dim)

    grid_interp = pd.DataFrame(
        index=pd.MultiIndex.from_product(grid_interp, names=range(dims))
    ).reset_index()

    # transform to np.array
    grid_interp = np.array(object=grid_interp)
    n_gridpoints_effective = grid_interp.shape[0]

    return grid_interp, n_gridpoints_effective


def get_interpolation_grid_sparse(dims, grid_min, grid_max, interp_params):

    # load interpolation parameters
    level = interp_params["linear"]["sparse grid level"]

    # generate grid
    sparse_grid = sparseGrid(dims, level)
    sparse_grid.generatePoints()

    grid_interp = []
    for i in range(len(sparse_grid.indices)):
        # for j in range(len(sparse_grid.gP[tuple(sparse_grid.indices[i])].pos)):
        grid_interp_tmp = sparse_grid.gP[tuple(sparse_grid.indices[i])].pos
        grid_interp.append(grid_interp_tmp)

    # transform to support and store as np.array
    grid_interp = np.array(
        object=(
            grid_interp * grid_min
            + (np.ones((len(grid_interp), len(grid_min))) - grid_interp) * grid_max
        ),
        dtype=float,
    )
    n_gridpoints_effective = grid_interp.shape[0]

    return grid_interp, n_gridpoints_effective


def get_not_interpolated_indicator_random(interpolation_points, n_states, seed):
    """Get indicator for states which will be not interpolated.

    Randomness in this function is held constant for each period but not across periods.
    This is done by adding the period to the seed set for the solution.

    Parameters
    ----------
    interpolation_points : int
        Number of states which will be interpolated.
    n_states : int
        Total number of states in period.
    seed : int
        Seed to set randomness.

    Returns
    -------
    not_interpolated : numpy.ndarray
        Array of shape (n_states,) indicating states which will not be interpolated.

    """
    np.random.seed(seed)

    indices = np.random.choice(n_states, size=interpolation_points, replace=False)

    not_interpolated = np.full(n_states, False)
    not_interpolated[indices] = True

    return not_interpolated


def interpolate_linear(points, grid, func, interp_params):

    # get number of states, number of dimensions and index of states
    dims = len(grid)
    grid_min = np.array(object=[min(v) for _, v in grid.items()])
    grid_max = np.array(object=[max(v) for _, v in grid.items()])

    # generate interpolation grid and number of interpolation points
    if interp_params["linear"]["grid method"] == "regular":
        grid_interp, n_gridpoints_effective = get_interpolation_grid_regular(
            dims, grid_min, grid_max, interp_params
        )
    elif interp_params["linear"]["grid method"] == "sparse":
        grid_interp, n_gridpoints_effective = get_interpolation_grid_sparse(
            dims, grid_min, grid_max, interp_params
        )

    # make sure corner states are included in interpolation grid
    corner_points = get_corner_points(grid)
    grid_interp = np.append(grid_interp, corner_points).reshape(
        corner_points.shape[0] + grid_interp.shape[0], grid_interp.shape[1]
    )

    # evaluate function on grid
    if interp_params["linear"]["evaluate off-grid"] == "True":
        f_on_grid = func(grid_interp)
    elif interp_params["linear"]["evaluate off-grid"] == "False":
        f_on_grid = interpolate_locally_batch(grid_interp, grid, func)

    # generate interpolator
    interpolator = sp_interpolate.LinearNDInterpolator(
        grid_interp, f_on_grid, rescale=True,
    )

    # calculate interpolated points
    results_interp = interpolator.__call__(points)

    return results_interp, n_gridpoints_effective


def interpolate_smolyak(points, grid, func, interp_params):

    # load interpolation parameters
    level = interp_params["smolyak"]["sparse grid level"]

    # get number of states, number of dimensions and index of states
    dims = len(grid)
    grid_min = np.array(object=[min(v) for _, v in grid.items()])
    grid_max = np.array(object=[max(v) for _, v in grid.items()])

    # generate smolyak grid
    s_grid = sg(dims, level, grid_min, grid_max)
    n_gridpoints_effective = len(s_grid.grid)
    grid_interp = s_grid.grid

    # evaluate function on grid
    if interp_params["smolyak"]["evaluate off-grid"] == "True":
        f_on_grid = func(grid_interp)
    elif interp_params["smolyak"]["evaluate off-grid"] == "False":
        f_on_grid = interpolate_locally_batch(grid_interp, grid, func)

    # generate interpolator
    interpolator = si(s_grid, f_on_grid)

    # calculate interpolated points
    results_interp = interpolator.interpolate(points)

    return results_interp, n_gridpoints_effective


def interpolate_spline(points, grid, func, interp_params):

    # load interpolation parameters
    interpolation_points = interp_params["spline"]["interpolation points"]

    # get number of states, number of dimensions and index of states
    dims = len(grid)
    grid_min = np.array(object=[min(v) for _, v in grid.items()])
    grid_max = np.array(object=[max(v) for _, v in grid.items()])
    orders = [interpolation_points] * dims

    # generate interpolation grid and number of interpolation points
    grid_interp, n_gridpoints_effective = get_interpolation_grid_regular(
        dims, grid_min, grid_max, interp_params
    )

    # evaluate function on grid
    f_on_grid = func(grid_interp)

    # generate interpolator
    interpolator = spline(grid_min, grid_max, orders, f_on_grid)

    # calculate interpolated points
    results_interp = interpolator(points)

    return results_interp, n_gridpoints_effective


def interpolate_sparse(points, grid, func, interp_params):

    # load interpolation parameters
    level = interp_params["sparse"]["sparse grid level"]
    interpolation_type = interp_params["sparse"]["polynomial family"]

    # get number of states, number of dimensions and index of states
    dims = len(grid)
    grid_min = np.array(object=[min(v) for _, v in grid.items()])
    grid_max = np.array(object=[max(v) for _, v in grid.items()])

    intval = np.asarray([grid_min, grid_max])

    # generate interpolator
    interp = SparseInterpolator(level, dims, interpolation_type, intval)

    # evaluate function on grid
    _ = interp.fit(func, points)

    # generate interpolator
    results_interp = interp.evaluate(points)
    n_gridpoints_effective = len(interp.grid)

    return results_interp, n_gridpoints_effective


def interpolate_locally_step(point, local_grid, func):

    f_on_grid = func(local_grid)
    interpolator = sp_interpolate.LinearNDInterpolator(local_grid, f_on_grid)
    f_interp = interpolator.__call__(point)

    return f_interp


def interpolate_locally_batch(points, grid, func):

    f_interp = np.full(points.shape[0], np.nan)
    for idx in range(points.shape[0]):
        local_grid = get_local_grid(points[idx, :], grid)
        f_interp[idx] = interpolate_locally_step(points[idx, :], local_grid, func)

    return f_interp
