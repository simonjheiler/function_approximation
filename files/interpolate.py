# import pdb
# import numba as nb
import numpy as np
import pandas as pd
import scipy.interpolate
from functions_to_approximate import borehole


def get_grids_values(dims_state_grid, grid_min, grid_max):

    n_dims = len(dims_state_grid)

    grids_values = []

    for idx in range(n_dims):
        grid_values_tmp = np.linspace(
            grid_min[idx], grid_max[idx], dims_state_grid[idx],
        )
        grids_values.append(grid_values_tmp)

    grids_values = np.array(object=grids_values)

    return grids_values


def get_dims_state_grid(n_state_variables, n_gridpoints):

    tmp = np.zeros(n_state_variables, dtype=np.int)
    for idx in range(n_state_variables):
        tmp[idx] = n_gridpoints[idx]

    dims_state_grid = np.array(object=list(tmp))

    return dims_state_grid


def get_grids_indices(dims_state_grid):

    grids_indices = []

    for idx in range(dims_state_grid.size):
        tmp = np.arange(dims_state_grid[idx])
        grids_indices.append(tmp)

    return grids_indices


def get_states_grid_dense(dims_state_grid):

    grids_indices = get_grids_indices(dims_state_grid)

    state_grid = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            grids_indices, names=range(dims_state_grid.size),
        )
    ).reset_index()

    return state_grid


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


def state_from_id(idx, dims_state_grid):
    """Convert *idx* to state series structured according to *dims_state_grid*"""

    entries = [idx] * len(dims_state_grid)
    divisors = []
    for i in range(1, len(dims_state_grid)):
        value = 1
        for j in range(i, len(dims_state_grid)):
            value *= dims_state_grid[j]
        divisors.append(value)
        for k in range(len(dims_state_grid)):
            if k == i - 1:
                entries[k] //= value
            else:
                entries[k] %= value
        out = np.array(object=entries)

    return out


# @nb.guvectorize(
#     ["float64, float64[:], float64[:]"],
#     "(), (n) -> (n)",
#     nopython=True,
#     target="parallel",
# )
# def states_from_ids_batch(index, dims_state_grid, states):
#
#     states[0] = state_from_id(index[0], dims_state_grid)


# @nb.vectorize(
#     ["float64(float64,float64[:])"],
#     nopython = True
# )
# def states_from_ids_batch(index, dims_state_grid):
#
#     idx = index
#     n_dims = len(dims_state_grid)
#
#     entries = [idx] * n_dims
#     divisors = []
#     for i in range(1, n_dims):
#         value = 1
#         for j in range(i, n_dims):
#             value *= dims_state_grid[j]
#         divisors.append(value)
#         for k in range(n_dims):
#             if k == i - 1:
#                 entries[k] //= value
#             else:
#                 entries[k] %= value
#         states = np.array(object=entries)
#
#     return states

# @nb.guvectorize(
#     ["f8[:], f8"],
#     "(n) -> ()",
#     nopython = True,
#     target = "parallel",
# )
# def evaluation_batch(states, )
#
#     states[0] =

# @nb.guvectorize(
#     ["f8[:], f8"],
#     "(n) -> ()",
#     nopython = True,
#     target = "parallel",
# )
# def evaluation_batch(states, )
#
#     states[0] =


def states_from_ids_batch(index, dims_state_grid):

    states_tmp = []

    for idx in range(len(index)):
        state = state_from_id(idx, dims_state_grid)
        states_tmp.append(state)

    states = np.array(object=states_tmp)

    return states


def inputs_from_ids_batch(index, dims_state_grid, grids_values):

    inputs_tmp = []

    for _idx, val in enumerate(index):
        state = state_from_id(val, dims_state_grid)
        input = inputs_from_state(state, grids_values)
        inputs_tmp.append(input)

    inputs = np.array(object=inputs_tmp)

    return inputs


def inputs_from_state(state, grids_values):

    inputs_tmp = []

    for idx, val in enumerate(state):
        input = grids_values[idx][val]
        inputs_tmp.append(input)

    inputs = np.array(object=inputs_tmp)

    return inputs


def evaluation_batch(states, grids_values):

    n_states, n_dims = states.shape

    out = []

    for idx in range(n_states):

        inputs_tmp = inputs_from_state(states[idx, :], grids_values)
        result_tmp = borehole(inputs_tmp)

        out.append(result_tmp)

    results = np.array(object=out)

    return results


def state_to_id(state, dims_state_grid):
    id = 0
    for i in range(len(dims_state_grid) - 1):
        step_size = 1
        for d in dims_state_grid[i + 1 :]:
            step_size *= d
        id += state[i] * step_size
    id += state[-1]
    return id


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


def get_data(dims_state_grid, grids_values):

    try:
        states = np.load("../data/sandbox/states.npy")
        results = np.load("../data/sandbox/results.npy")
    except FileNotFoundError:
        n_states = dims_state_grid.prod()
        states_index = np.array(object=range(n_states), dtype=int)
        states = states_from_ids_batch(states_index, dims_state_grid)
        results = evaluation_batch(states, grids_values)
        np.save("../data/sandbox/states", states)
        np.save("../data/sandbox/results", results)

    return states, results


def interpolate_linear(state, basis_points, basis_results):

    interpolator = scipy.interpolate.LinearNDInterpolator(basis_points, basis_results)

    predicted_output = interpolator.__call__(state)

    return predicted_output


n_state_variables = 4

n_gridpoints_defaults = np.array(object=[3, 7, 4, 3, 10, 10, 10, 10], dtype=int,)
grid_min_defaults = np.array(
    object=[63070.0, 990.0, 700.0, 100.0, 0.05, 1120.0, 1500.0, 63.1], dtype=float,
)
grid_max_defaults = np.array(
    object=[115600.0, 1110.0, 820.0, 50000.0, 0.15, 1680.0, 15000.0, 116.0],
    dtype=float,
)

n_gridpoints = n_gridpoints_defaults[:n_state_variables]
grid_min = grid_min_defaults[:n_state_variables]
grid_max = grid_max_defaults[:n_state_variables]

n_states = n_gridpoints.prod()
seed = 123
interpolation_points = 50

dims_state_grid = get_dims_state_grid(n_state_variables, n_gridpoints)
grids_values = get_grids_values(dims_state_grid, grid_min, grid_max)


states, results = get_data(dims_state_grid, grids_values)

n_states, n_dims = states.shape

states_index = np.array(object=range(n_states))
not_interpolated = get_not_interpolated_indicator_random(
    interpolation_points, n_states, seed
)


corner_states = get_corner_states(dims_state_grid)
corner_index = states_to_ids_batch(corner_states, dims_state_grid)

basis_index = np.unique(np.concatenate((states_index[not_interpolated], corner_index)))

basis_points = inputs_from_ids_batch(basis_index, dims_state_grid, grids_values)
basis_results = results[basis_index]

states_grid = np.array(object=get_states_grid_dense(dims_state_grid))


index_test = [17, 22, 34, 50, 110, 12]

inputs_test = inputs_from_ids_batch(index_test, dims_state_grid, grids_values)
predict_test = interpolate_linear(inputs_test, basis_points, basis_results)
actual_test = results[index_test]

print(predict_test)
print(actual_test)
print(np.sum((predict_test - actual_test) ** 2) / len(predict_test))
