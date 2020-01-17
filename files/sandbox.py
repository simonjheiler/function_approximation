import os  # noqa F401
import pdb  # noqa F401
import pickle  # noqa F401
import sys

sys.path.insert(
    0,
    (
        "C:/Users/simon/Documents/Uni/3_Bonn/3_WiSe19-20/"
        "topics_SBE/3_project/student-project-simonjheiler"
    ),
)

import matplotlib.pyplot as plt  # noqa F401
import numpy as np
from time import time

from files.functions_to_approximate import borehole  # noqa F401
from files.interpolate import evaluation_batch  # noqa F401
from files.interpolate import get_corner_states  # noqa F401
from files.interpolate import get_data  # noqa F401
from files.interpolate import get_dims_state_grid  # noqa F401
from files.interpolate import get_grids_indices  # noqa F401
from files.interpolate import get_grids_values  # noqa F401
from files.interpolate import get_not_interpolated_indicator_random  # noqa F401
from files.interpolate import get_states_grid_dense  # noqa F401
from files.interpolate import inputs_from_ids_batch  # noqa F401
from files.interpolate import inputs_from_state  # noqa F401
from files.interpolate import interpolate_linear  # noqa F401
from files.interpolate import interpolate_smolyak  # nopa F401
from files.interpolate import mse as mean_squared_error  # noqa F401
from files.interpolate import rmse as root_mean_squared_error  # noqa F401
from files.interpolate import state_from_id  # noqa F401
from files.interpolate import state_to_id  # noqa F401
from files.interpolate import states_from_ids_batch  # noqa F401
from files.interpolate import states_to_ids_batch  # noqa F401

# import pandas as pd
# import numba as nb

# default parameters for simulation study
grid_min_defaults = np.array(
    object=[63070.0, 990.0, 700.0, 100.0, 0.05, 1120.0, 1500.0, 63.1], dtype=float,
)
grid_max_defaults = np.array(
    object=[115600.0, 1110.0, 820.0, 50000.0, 0.15, 1680.0, 15000.0, 116.0],
    dtype=float,
)

n_gridpoints_dim_params = {}
n_gridpoints_dim_params["small"] = 5
n_gridpoints_dim_params["medium"] = 10
n_gridpoints_dim_params["large"] = 100

grid_density_params = {}
grid_density_params["small"] = np.array(object=[0, 1])
grid_density_params["medium"] = np.array(object=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
grid_density_params["large"] = np.array(
    object=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
)

vars_params = {}
vars_params["small"] = [2, 3, 4, 5, 6, 7, 8]
vars_params["medium"] = [2, 3, 4, 5]
vars_params["large"] = [2]

# set parameters for simulation study
seed = 123
load_data = False
grid_size = "small"

# load parameters
n_gridpoints_dim = n_gridpoints_dim_params[grid_size]
grid_density = grid_density_params[grid_size]
vars = vars_params[grid_size]

n_gridpoints_defaults = np.array(object=[5, 5, 5, 5, 5, 5, 5, 5], dtype=int,)

interpolation_points = {}

interp_params = {}
interp_params["linear"] = {}
interp_params["linear"]["seed"] = seed
interp_params["smolyak"] = {}

study_params = {
    "linear": {},
    "smolyak": {},
}

study_params["linear"]["iterations"] = []
study_params["linear"]["grid_size"] = "small"

study_params["smolyak"]["variables"] = [2, 3, 4, 5, 6, 7, 8]
study_params["smolyak"]["iterations"] = [1, 2, 3]


for n_vars in vars:
    interpolation_points[n_vars] = (
        grid_density * (n_gridpoints_dim ** (n_vars - 1))
    ).tolist()


if not (load_data):

    results = {}
    results["linear"] = {"rmse": {}, "runtime": {}}
    results["smolyak"] = {"rmse": {}, "runtime": {}}

    vars = study_params["smolyak"]["variables"]
    for n_vars in vars:
        n_dims = n_vars

        n_gridpoints = n_gridpoints_defaults[:n_dims]
        grid_min = grid_min_defaults[:n_dims]
        grid_max = grid_max_defaults[:n_dims]

        n_states = n_gridpoints.prod()
        dims_state_grid = get_dims_state_grid(n_dims, n_gridpoints)
        grid = get_grids_values(dims_state_grid, grid_min, grid_max)
        index = np.array(object=range(n_states))

        mse_tmp = []
        runtime_tmp = []

        for iteration in study_params["linear"]["iterations"]:
            print(f"dimension: {n_vars}; iteration: {iteration}")

            # adjust interpolation parameters
            interp_params["linear"]["n_interpolation_points"] = interpolation_points[
                n_vars
            ][iteration]

            # interpolate and capture computation time
            start_lin = time()
            results_linear = interpolate_linear(grid, borehole, interp_params)
            stop_lin = time()

            # calculate actuals
            inputs = inputs_from_ids_batch(index, dims_state_grid, grid)
            results_calc = evaluation_batch(inputs, borehole)

            # assess interpolation accuracy
            print("calculate mean squared error")
            mse_iter_lin = mean_squared_error(results_linear, results_calc)

            mse_tmp.append(mse_iter_lin)
            runtime_tmp.append(stop_lin - start_lin)
            print(f"mean squared error: linear {mse_iter_lin}")
            print("computation time: linear {}".format(stop_lin - start_lin))

        results["linear"]["rmse"][n_vars] = np.array(object=mse_tmp)
        results["linear"]["runtime"][n_vars] = np.array(object=runtime_tmp)

        for iteration in study_params["smolyak"]["iterations"]:
            print(f"dimension: {n_vars}; iteration: {iteration}")

            # adjust interpolation parameters
            interp_params["smolyak"]["mu"] = iteration

            # interpolate and capture computation time
            start_smo = time()
            results_smolyak = interpolate_smolyak(grid, borehole, interp_params)
            stop_smo = time()

            # calculate actuals
            inputs = inputs_from_ids_batch(index, dims_state_grid, grid)
            results_calc = evaluation_batch(inputs, borehole)

            # assess interpolation accuracy
            print("calculate mean squared error")
            mse_iter_smo = mean_squared_error(results_smolyak, results_calc)

            mse_tmp.append(mse_iter_smo)
            runtime_tmp.append(stop_smo - start_smo)
            print(f"mean squared error: smolyak {mse_iter_smo}")
            print("computation time: smolyak {}".format(stop_smo - start_smo))

        results["smolyak"]["rmse"][n_vars] = np.array(object=mse_tmp)
        results["smolyak"]["runtime"][n_vars] = np.array(object=runtime_tmp)

elif load_data:
    if grid_size == "large":
        mse = {
            2: np.array(
                object=[
                    1.15827133e-10,
                    4.10937299e-15,
                    1.75534356e-15,
                    1.24543726e-15,
                    1.53823343e-16,
                    7.30845478e-17,
                    3.89361083e-17,
                    2.42273906e-17,
                    1.10140911e-17,
                    5.25685920e-18,
                    0.00000000e00,
                ]
            ),
        }
        interpolation_points_effective = {
            2: np.array(
                object=[4, 1004, 2004, 3004, 4004, 5004, 6004, 7003, 8001, 9001, 10000]
            ),
        }
    elif grid_size == "medium":
        mse = {
            2: np.array(
                object=[
                    1.16843211e-10,
                    6.48991577e-11,
                    3.45252454e-11,
                    8.79772638e-12,
                    2.79300317e-12,
                    8.63672700e-13,
                    3.56032139e-13,
                    2.45258087e-13,
                    8.05782712e-14,
                    2.76833230e-14,
                    0.00000000e00,
                ]
            ),
            3: np.array(
                object=[
                    2.42575238e-10,
                    1.03582223e-11,
                    3.50616603e-12,
                    1.18854452e-12,
                    9.72378989e-13,
                    4.77674206e-13,
                    3.29491019e-13,
                    2.57437621e-13,
                    1.96565431e-13,
                    7.02029625e-14,
                    0.00000000e00,
                ]
            ),
            4: np.array(
                object=[
                    1.87607523e-03,
                    8.59999910e-05,
                    4.27297374e-05,
                    3.19191028e-05,
                    2.25815252e-05,
                    1.45095742e-05,
                    1.04099826e-05,
                    9.48770708e-06,
                    4.81283348e-06,
                    3.20003480e-06,
                    0.00000000e00,
                ]
            ),
            5: np.array(
                object=[
                    9.29339622e01,
                    5.03343766e-01,
                    2.22732461e-01,
                    1.10035992e-01,
                    6.85425988e-02,
                    4.92652689e-02,
                    3.77860899e-02,
                    2.74544516e-02,
                    1.74367184e-02,
                    8.54009361e-03,
                    0.00000000e00,
                ]
            ),
        }
        interpolation_points_effective = {
            2: np.array(object=[4, 13, 23, 32, 41, 51, 61, 71, 80, 90, 100]),
            3: np.array(object=[8, 108, 208, 307, 406, 505, 604, 703, 802, 902, 1000]),
            4: np.array(
                object=[16, 1014, 2014, 3011, 4009, 5009, 6007, 7005, 8002, 9001, 10000]
            ),
            5: np.array(
                object=[
                    32,
                    10028,
                    20023,
                    30019,
                    40016,
                    50012,
                    60009,
                    70008,
                    80004,
                    90002,
                    100000,
                ]
            ),
        }
    elif grid_size == "small":
        mse = {
            2: np.array(
                object=[
                    1.16297227e-10,
                    7.73656025e-11,
                    5.70148687e-11,
                    2.30639637e-12,
                    1.13256826e-12,
                    0.00000000e00,
                ]
            ),
            3: np.array(
                object=[
                    2.22628314e-10,
                    4.90331304e-11,
                    3.37867597e-11,
                    5.83973883e-12,
                    3.26716674e-12,
                    0.00000000e00,
                ]
            ),
            4: np.array(
                object=[
                    1.43189982e-03,
                    2.12970441e-04,
                    8.87126508e-05,
                    4.84834642e-05,
                    3.18754793e-05,
                    0.00000000e00,
                ]
            ),
            5: np.array(
                object=[
                    86.47001962,
                    5.51067626,
                    2.46586576,
                    1.15209044,
                    0.47468751,
                    0.00000000e00,
                ]
            ),
        }
        interpolation_points_effective = {
            2: np.array(object=[4, 9, 13, 17, 20, 25]),
            3: np.array(object=[8, 29, 52, 75, 100, 125]),
            4: np.array(object=[16, 135, 256, 378, 501, 625]),
            5: np.array(object=[32, 653, 1270, 1885, 2505, 3125]),
        }
    else:
        print("grid size not found")


# with open('../data/sandbox/mse.pkl', 'rb') as data_to_load:
#     mse = pickle.load(data_to_load)

# results_to_store = open("../data/sandbox/mse.pkl","wb")
# pickle.dump(mse, results_to_store)
# results_to_store.close()


# plot_legend = []
# plot_x = []
# plot_y = []
# for n_vars in vars:
#     plot_legend.append(n_vars)
#     plot_x.append(interpolation_points_effective[n_vars])
#     plot_y.append(np.sqrt(mse[n_vars]))
#
# for idx in range(len(vars)):
#     plt.plot(plot_x[idx], plot_y[idx])
#
# plt.xscale("log")
# plt.yscale("log")
# plt.xlabel("number of interpolation points (log axis)")
# plt.ylabel("root mean squared error (log axis)")
# plt.legend(plot_legend)
# plt.title("Interpolation accuracy (" + grid_size + " grid)")
# plt.show()
