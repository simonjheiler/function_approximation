import pdb  # noqa F401

import numpy as np
import pandas as pd
import scipy.interpolate as sp_interpolate
from interpolation.smolyak.grid import SmolyakGrid as sg
from interpolation.smolyak.interp import SmolyakInterp as si
from interpolation.splines import CubicSpline as spline

# from interpolation.multilinear.mlinterp import mlinterp
# from src.auxiliary import get_corner_states
# from src.auxiliary import get_dims_state_grid
# from src.auxiliary import get_grid
# from src.auxiliary import inputs_from_ids_batch
# from src.auxiliary import states_to_ids_batch

# import numba as nb


#########################################################################
# FUNCTIONS
#########################################################################


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

    # load interpolation parameters
    interpolation_points = interp_params["linear"]["interpolation_points"]

    # get number of states, number of dimensions and index of states
    n_dims = len(grid)
    grid_min = np.array(object=[min(v) for _, v in grid.items()])
    grid_max = np.array(object=[max(v) for _, v in grid.items()])

    # generate regular interpolation grid and number of interpolation points
    grid_interp, n_gridpoints_effective = get_interpolation_grid(
        n_dims, grid_min, grid_max, interpolation_points
    )

    # evaluate function on grid
    f_on_grid = func(grid_interp)

    # generate interpolator
    interpolator = sp_interpolate.LinearNDInterpolator(
        grid_interp, f_on_grid, rescale=True,
    )

    # calculate interpolated points
    results_interp = interpolator.__call__(points)

    return results_interp, n_gridpoints_effective


def interpolate_smolyak(points, grid, func, interp_params):

    # load interpolation parameters
    mu = interp_params["smolyak"]["mu"]

    # get number of states, number of dimensions and index of states
    n_dims = len(grid)
    grid_min = np.array(object=[min(v) for _, v in grid.items()])
    grid_max = np.array(object=[max(v) for _, v in grid.items()])

    # generate smolyak grid
    s_grid = sg(n_dims, mu, grid_min, grid_max)
    n_gridpoints_effective = len(s_grid.grid)
    grid_interp = s_grid.grid

    # evaluate function on grid
    f_on_grid = func(grid_interp)

    # generate interpolator
    interpolator = si(s_grid, f_on_grid)

    # calculate interpolated points
    results_interp = interpolator.interpolate(points)

    return results_interp, n_gridpoints_effective


def interpolate_spline(points, grid, func, interp_params):

    # load interpolation parameters
    interpolation_points = interp_params["spline"]["interpolation_points"]

    # get number of states, number of dimensions and index of states
    n_dims = len(grid)
    grid_min = np.array(object=[min(v) for _, v in grid.items()])
    grid_max = np.array(object=[max(v) for _, v in grid.items()])
    orders = [interpolation_points] * n_dims

    # generate regular interpolation grid and number of interpolation points
    grid_interp, n_gridpoints_effective = get_interpolation_grid(
        n_dims, grid_min, grid_max, interpolation_points
    )

    # evaluate function on grid
    f_on_grid = func(grid_interp)

    # generate interpolator
    interpolator = spline(grid_min, grid_max, orders, f_on_grid)

    # calculate interpolated points
    results_interp = interpolator(points)

    return results_interp, n_gridpoints_effective


def get_interpolation_grid(n_dims, grid_min, grid_max, interpolation_points):

    grid_interp = []
    for idx in range(n_dims):
        grid_dim = np.linspace(grid_min[idx], grid_max[idx], interpolation_points)
        grid_interp.append(grid_dim)

    grid_interp = pd.DataFrame(
        index=pd.MultiIndex.from_product(grid_interp, names=range(n_dims))
    ).reset_index()
    grid_interp = np.array(object=grid_interp)
    n_gridpoints_effective = grid_interp.shape[0]

    return grid_interp, n_gridpoints_effective
