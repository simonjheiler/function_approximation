import pdb  # noqa F401

import numpy as np
from functions_to_approximate import borehole  # noqa F401
from interpolate import evaluation_batch  # noqa F401
from interpolate import get_corner_states  # noqa F401
from interpolate import get_data  # noqa F401
from interpolate import get_dims_state_grid  # noqa F401
from interpolate import get_grids_indices  # noqa F401
from interpolate import get_grids_values  # noqa F401
from interpolate import get_not_interpolated_indicator_random  # noqa F401
from interpolate import get_states_grid_dense  # noqa F401
from interpolate import inputs_from_ids_batch  # noqa F401
from interpolate import inputs_from_state  # noqa F401
from interpolate import interpolate_linear  # noqa F401
from interpolate import state_from_id  # noqa F401
from interpolate import state_to_id  # noqa F401
from interpolate import states_from_ids_batch  # noqa F401
from interpolate import states_to_ids_batch  # noqa F401

# import pandas as pd

# import numba as nb

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


index_test = np.array(object=[17, 22, 34, 50, 110, 12])

inputs_test = inputs_from_ids_batch(index_test, dims_state_grid, grids_values)
predict_test = interpolate_linear(inputs_test, basis_points, basis_results)
actual_test = results[index_test]

print(predict_test)
print(actual_test)
print(np.sum((predict_test - actual_test) ** 2) / len(predict_test))


index_test = range(29)
dims_state_grid_test = np.array(object=[3, 2, 5])
states_test = states_from_ids_batch(index_test, dims_state_grid_test)
print(states_test)

pdb.set_trace()