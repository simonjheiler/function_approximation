import pdb  # noqa F401

import matplotlib.pyplot as plt
import numpy as np
from functions_to_approximate import borehole  # noqa F401
from interpolate import evaluation_batch  # noqa F401
from interpolate import get_corner_states  # noqa F401
from interpolate import get_data  # noqa F401
from interpolate import get_dims_state_grid  # noqa F401
from interpolate import get_grids_indices  # noqa F401
from interpolate import get_grids_values  # noqa F401
from interpolate import get_mean_squared_error  # noqa F401
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


n_gridpoints_defaults = np.array(object=[10, 10, 10, 10, 10, 10, 10, 10], dtype=int,)
grid_min_defaults = np.array(
    object=[63070.0, 990.0, 700.0, 100.0, 0.05, 1120.0, 1500.0, 63.1], dtype=float,
)
grid_max_defaults = np.array(
    object=[115600.0, 1110.0, 820.0, 50000.0, 0.15, 1680.0, 15000.0, 116.0],
    dtype=float,
)

seed = 123
interpolation_points = {
    1: [0, 1, 5, 10],
    2: [0, 10, 50, 100],
    3: [0, 100, 500, 1000],
    4: [0, 1000, 5000, 10000],
    5: [0, 10000, 50000, 100000],
}

vars = [2, 3, 4]
iterations = [0, 1, 2, 3]

mse = {}

for n_vars in vars:
    n_state_variables = n_vars

    n_gridpoints = n_gridpoints_defaults[:n_state_variables]
    grid_min = grid_min_defaults[:n_state_variables]
    grid_max = grid_max_defaults[:n_state_variables]

    n_states = n_gridpoints.prod()

    mse_tmp = []

    for iteration in iterations:
        n_interpolation_points = interpolation_points[n_vars][iteration]

        dims_state_grid = get_dims_state_grid(n_state_variables, n_gridpoints)
        grids_values = get_grids_values(dims_state_grid, grid_min, grid_max)

        n_states = dims_state_grid.prod()
        states_index = np.array(object=range(n_states), dtype=int)
        states = states_from_ids_batch(states_index, dims_state_grid)
        results = evaluation_batch(states, grids_values)

        index = np.array(object=range(n_states))
        not_interpolated = get_not_interpolated_indicator_random(
            n_interpolation_points, n_states, seed
        )

        print("get interpolation points")

        corner_states = get_corner_states(dims_state_grid)
        corner_index = states_to_ids_batch(corner_states, dims_state_grid)
        basis_index = np.unique(np.concatenate((index[not_interpolated], corner_index)))
        basis_points = inputs_from_ids_batch(basis_index, dims_state_grid, grids_values)
        basis_results = results[basis_index]

        print("interpolate")

        not_interpolated = np.in1d(index, basis_index)
        interpolated = np.logical_not(not_interpolated)
        index_not_interpolated = index[not_interpolated]
        index_interpolated = index[interpolated]

        inputs = inputs_from_ids_batch(
            index_interpolated, dims_state_grid, grids_values
        )
        # basis_points32 = basis_points.astype(np.float32)
        predict = interpolate_linear(inputs, basis_points, basis_results)
        # pdb.set_trace()
        results_predicted = np.full_like(results, np.nan)
        results_predicted[not_interpolated] = results[not_interpolated]
        results_predicted[interpolated] = predict
        results_calculated = results
        # pdb.set_trace()

        print("calculate mean squared error")

        mse_iter = get_mean_squared_error(results_predicted, results_calculated)

        mse_tmp.append(mse_iter)

        print(mse_iter)

    mse[n_vars] = np.array(object=mse_tmp)

print(mse)


plt.plot(mse)
