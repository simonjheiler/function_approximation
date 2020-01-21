import pdb  # noqa F401

import numpy as np
import pandas as pd

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


def states_to_ids_batch(states, dims_state_grid):

    n_states, _ = states.shape
    ids = []

    for idx in range(n_states):
        id_tmp = state_to_id(states[idx, :], dims_state_grid)
        ids.append(id_tmp)

    return ids


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
        results = func(points)
        # np.save("./results/sandbox/" + file_states, points)
        # np.save("./results/sandbox/" + file_results, results)

    return points, results


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
