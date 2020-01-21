import pdb  # noqa F401

import numpy as np
import pandas as pd
import scipy.interpolate as sp_interpolate
from interpolation.multilinear.mlinterp import mlinterp
from interpolation.smolyak.grid import SmolyakGrid as sg
from interpolation.smolyak.interp import SmolyakInterp as si
from interpolation.splines import CubicSpline as spline

# import numba as nb


#########################################################################
# FUNCTIONS
#########################################################################


def get_grid(dims_state_grid, grid_min, grid_max):

    n_dims = len(dims_state_grid)

    grid = {}

    for idx in range(n_dims):
        grid_values_tmp = np.linspace(
            grid_min[idx], grid_max[idx], dims_state_grid[idx],
        )
        grid[idx] = grid_values_tmp

    return grid


def get_dims_state_grid(n_dims, n_gridpoints):

    tmp = np.zeros(n_dims, dtype=np.int)
    for idx in range(n_dims):
        tmp[idx] = n_gridpoints[idx]

    dims_state_grid = np.array(object=list(tmp))

    return dims_state_grid


def get_corner_states(dims_state_grid):

    grids_bounds = []
    for idx in range(dims_state_grid.size):
        tmp = [0, dims_state_grid[idx] - 1]
        grids_bounds.append(tmp)

    corner_states = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            grids_bounds, names=range(dims_state_grid.size),
        )
    ).reset_index()

    corner_states = np.array(object=corner_states)

    return corner_states


def state_to_id(state, dims_state_grid):
    id = 0
    for i in range(len(dims_state_grid) - 1):
        step_size = 1
        for d in dims_state_grid[i + 1 :]:
            step_size *= d
        id += state[i] * step_size
    id += state[-1]
    return id


def state_from_id(index, dims_state_grid):
    """Convert *idx* to state array structured according to *dims_state_grid*"""

    entries = [index] * len(dims_state_grid)
    for i in range(1, len(dims_state_grid)):
        value = 1
        for j in range(i, len(dims_state_grid)):
            value *= dims_state_grid[j]
        for k in range(i - 1, len(dims_state_grid)):
            if k == i - 1:
                entries[k] //= value
            else:
                entries[k] %= value

    out = np.array(object=entries)

    return out


def inputs_from_state(state, grid):

    inputs_tmp = []

    for idx, val in enumerate(state):
        input = grid[idx][val]
        inputs_tmp.append(input)

    inputs = np.array(object=inputs_tmp)

    return inputs


def states_from_ids_batch(index, dims_state_grid):

    states_tmp = []

    for _idx, val in enumerate(index):
        state = state_from_id(val, dims_state_grid)
        states_tmp.append(state)

    states = np.array(object=states_tmp)

    return states


def inputs_from_ids_batch(index, dims_state_grid, grid):

    inputs_tmp = []

    for _idx, val in enumerate(index):
        state = state_from_id(val, dims_state_grid)
        input = inputs_from_state(state, grid)
        inputs_tmp.append(input)

    inputs = np.array(object=inputs_tmp)

    return inputs


def evaluation_batch(points, func):

    n_states, n_dims = points.shape
    out = []

    for idx in range(n_states):
        result_tmp = func(points[idx, :])
        out.append(result_tmp)

    results = np.array(object=out)

    return results


def states_to_ids_batch(states, dims_state_grid):

    n_states, _ = states.shape
    ids = []

    for idx in range(n_states):
        id_tmp = state_to_id(states[idx, :], dims_state_grid)
        ids.append(id_tmp)

    return ids


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


def get_data(func, grid_size, n_vars, dims_state_grid, grid):

    # filenames
    file_states = func.__name__ + "_" + grid_size + "_" + str(n_vars) + "_points.npy"
    file_results = func.__name__ + "_" + grid_size + "_" + str(n_vars) + "_results.npy"

    try:
        points = np.load("./results/sandbox/" + file_states)
        results = np.load("./results/sandbox/" + file_results)
    except FileNotFoundError:
        n_states = dims_state_grid.prod()
        index = np.array(object=range(n_states), dtype=int)
        points = inputs_from_ids_batch(index, dims_state_grid, grid)
        results = evaluation_batch(points, func)
        # np.save("./results/sandbox/" + file_states, points)
        # np.save("./results/sandbox/" + file_results, results)

    return points, results


def interpolate_linear(grid, func, interp_params):

    # load interpolation parameters
    seed = interp_params["linear"]["seed"]
    n_interpolation_points = interp_params["linear"]["n_interpolation_points"]
    interpolation_points = interp_params["linear"]["interpolation_points"]
    grid_method = interp_params["linear"]["grid_method"]

    # get number of states, number of dimensions and index of states
    n_dims = len(grid)
    n_gridpoints = np.array(object=[len(v) for _, v in grid.items()])
    grid_min = np.array(object=[min(v) for _, v in grid.items()])
    grid_max = np.array(object=[max(v) for _, v in grid.items()])

    dims_state_grid = get_dims_state_grid(n_dims, n_gridpoints)
    n_states = dims_state_grid.prod()
    index = np.array(object=range(n_states))

    # generate full grid
    grid = get_grid(dims_state_grid, grid_min, grid_max)

    # generate random interpolation grid
    if grid_method == "random":
        corner_states = get_corner_states(dims_state_grid)
        corner_index = states_to_ids_batch(corner_states, dims_state_grid)
        not_interpolated = get_not_interpolated_indicator_random(
            n_interpolation_points, n_states, seed
        )
        basis_index = np.unique(np.concatenate((index[not_interpolated], corner_index)))
        grid_interp = inputs_from_ids_batch(basis_index, dims_state_grid, grid)

    elif grid_method == "regular":
        # generate regular interpolation grid
        grid_interp = []

        for idx in range(n_dims):
            grid_dim = np.linspace(grid_min[idx], grid_max[idx], interpolation_points,)
            grid_interp.append(grid_dim)

        grid_interp = pd.DataFrame(
            index=pd.MultiIndex.from_product(grid_interp, names=range(n_dims),)
        ).reset_index()
        grid_interp = np.array(object=grid_interp)

    # evaluate function on grid and store number of interpolation points
    n_gridpoints_effective = grid_interp.shape[0]
    f_on_grid = evaluation_batch(grid_interp, func)

    # generate interpolator
    interpolator = sp_interpolate.LinearNDInterpolator(
        grid_interp, f_on_grid, rescale=True,
    )

    # calculate interpolated points
    # not_interpolated = np.in1d(index, basis_index)
    # interpolated = np.logical_not(not_interpolated)
    # index_interpolated = index[interpolated]
    states_interpolated = inputs_from_ids_batch(index, dims_state_grid, grid)
    results_interp = interpolator.__call__(states_interpolated)

    return results_interp, n_gridpoints_effective


def interpolate_linear_2(grid, func, interp_params):

    # load interpolation parameters
    interpolation_points = interp_params["spline"]["interpolation_points"]

    # get number of states, number of dimensions and index of states
    n_dims = len(grid)
    n_gridpoints = np.array(object=[len(v) for _, v in grid.items()])
    grid_min = np.array(object=[min(v) for _, v in grid.items()])
    grid_max = np.array(object=[max(v) for _, v in grid.items()])

    dims_state_grid = get_dims_state_grid(n_dims, n_gridpoints)
    n_states = dims_state_grid.prod()
    index = np.array(object=range(n_states))

    # generate grid
    grid_interp = []

    for idx in range(n_dims):
        grid_tmp = np.linspace(grid_min[idx], grid_max[idx], interpolation_points,)
        grid_interp.append(grid_tmp)

    basis_points = pd.DataFrame(
        index=pd.MultiIndex.from_product(grid_interp, names=range(n_dims),)
    ).reset_index()
    basis_points = np.array(object=basis_points)
    n_gridpoints_effective, _ = basis_points.shape

    # evaluate function on grid
    f_on_grid = evaluation_batch(basis_points, func)

    # calculate interpolated points
    states_interpolated = inputs_from_ids_batch(index, dims_state_grid, grid)
    results_interp = mlinterp(basis_points, f_on_grid, states_interpolated)

    return results_interp, n_gridpoints_effective


def interpolate_smolyak(grid, func, interp_params):

    # load interpolation parameters
    mu = interp_params["smolyak"]["mu"]

    # get number of states, number of dimensions and index of states
    n_dims = len(grid)
    n_gridpoints = np.array(object=[len(v) for _, v in grid.items()])
    grid_min = np.array(object=[min(v) for _, v in grid.items()])
    grid_max = np.array(object=[max(v) for _, v in grid.items()])

    dims_state_grid = get_dims_state_grid(n_dims, n_gridpoints)
    n_states = dims_state_grid.prod()
    index = np.array(object=range(n_states))

    # generate smolyak grid
    s_grid = sg(n_dims, mu, grid_min, grid_max)
    n_gridpoints_effective = len(s_grid.grid)

    # evaluate function on grid
    f_on_grid = evaluation_batch(s_grid.grid, func)

    # generate interpolator
    interpolator = si(s_grid, f_on_grid)

    # calculate interpolated points
    states_interpolated = inputs_from_ids_batch(index, dims_state_grid, grid)
    results_interp = interpolator.interpolate(states_interpolated)

    return results_interp, n_gridpoints_effective


def interpolate_spline(grid, func, interp_params):

    # load interpolation parameters
    interpolation_points = interp_params["spline"]["interpolation_points"]

    # get number of states, number of dimensions and index of states
    n_dims = len(grid)
    n_gridpoints = np.array(object=[len(v) for _, v in grid.items()])
    grid_min = np.array(object=[min(v) for _, v in grid.items()])
    grid_max = np.array(object=[max(v) for _, v in grid.items()])

    dims_state_grid = get_dims_state_grid(n_dims, n_gridpoints)
    n_states = dims_state_grid.prod()
    index = np.array(object=range(n_states))
    orders = [interpolation_points] * n_dims
    # generate grid
    grid_interp = []

    for idx in range(n_dims):
        grid_tmp = np.linspace(grid_min[idx], grid_max[idx], interpolation_points,)
        grid_interp.append(grid_tmp)

    basis_points = pd.DataFrame(
        index=pd.MultiIndex.from_product(grid_interp, names=range(n_dims),)
    ).reset_index()
    basis_points = np.array(object=basis_points)
    n_gridpoints_effective, _ = basis_points.shape
    # evaluate function on grid
    f_on_grid = evaluation_batch(basis_points, func)

    # generate interpolator
    interpolator = spline(grid_min, grid_max, orders, f_on_grid)

    # calculate interpolated points
    states_interpolated = inputs_from_ids_batch(index, dims_state_grid, grid)
    results_interp = interpolator(states_interpolated)

    return results_interp, n_gridpoints_effective


def mse(x1, x2, axis=0):
    """Calculate mean squared error.

    If ``x1`` and ``x2`` have different shapes, then they need to broadcast. This uses
    :func:`numpy.asanyarray` to convert the input. Whether this is the desired result or
    not depends on the array subclass, for example NumPy matrices will silently
    produce an incorrect result.

    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two arrays.
    axis : int
       Axis along which the summary statistic is calculated

    Returns
    -------
    mse : numpy.ndarray or float
       Mean squared error along given axis.

    """
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.mean((x1 - x2) ** 2, axis=axis)


def rmse(x1, x2, axis=0):
    """Calculate root mean squared error.

    If ``x1`` and ``x2`` have different shapes, then they need to broadcast. This uses
    :func:`numpy.asanyarray` to convert the input. Whether this is the desired result or
    not depends on the array subclass, for example NumPy matrices will silently
    produce an incorrect result.

    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two arrays.
    axis : int
       Axis along which the summary statistic is calculated.

    Returns
    -------
    rmse : numpy.ndarray or float
       Root mean squared error along given axis.

    """
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.sqrt(mse(x1, x2, axis=axis))
