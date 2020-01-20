import pdb  # noqa F401
import pickle
import sys

sys.path.insert(
    0,
    (
        "C:/Users/simon/Documents/Uni/3_Bonn/3_WiSe19-20/"
        "topics_SBE/3_project/student-project-simonjheiler"
    ),
)

import matplotlib.pyplot as plt
import numpy as np
from time import time

from src.functions_to_approximate import borehole
from src.interpolate import get_data
from src.interpolate import get_grid
from src.interpolate import interpolate_linear
from src.interpolate import interpolate_smolyak
from src.interpolate import interpolate_spline
from src.interpolate import rmse as root_mean_squared_error

# import pandas as pd
# import numba as nb


# set defaults for interpolation parameters
interp_params = {}
interp_params["linear"] = {}
interp_params["linear"]["seed"] = 123
interp_params["linear"]["grid_method"] = "regular"
interp_params["linear"]["n_interpolation_points"] = 0
interp_params["smolyak"] = {}
interp_params["smolyak"]["mu"] = 1
interp_params["spline"] = {}
interp_params["spline"]["interpolation_points"] = 3

# set study parameters
study_params = {}

study_params["controls"] = {
    "load data": False,
    "method": "linear",
    "grid size": "medium",
    "variables": [2, 3, 4, 5],
    "function to approximate": borehole,
}

study_params["grid"] = {
    "lower bounds": np.array(
        object=[63070.0, 990.0, 700.0, 100.0, 0.05, 1120.0, 1500.0, 63.1], dtype=float,
    ),
    "upper bounds": np.array(
        object=[115600.0, 1110.0, 820.0, 50000.0, 0.15, 1680.0, 15000.0, 116.0],
        dtype=float,
    ),
    "n_gridpoints": {"small": 5, "medium": 10, "large": 100},
}

study_params["linear"] = {
    "interpolator": interpolate_linear,
    "share_not_interpolated": [0.0, 0.2, 0.4],
    "interpolation_points": [2, 3, 5, 7, 9, 11],
}
study_params["smolyak"] = {
    "interpolator": interpolate_smolyak,
    "mu": [1, 2, 3],
}
study_params["spline"] = {
    "interpolator": interpolate_spline,
    "interpolation_points": [2, 3, 5, 7, 9, 11],
}

# run routine
if not study_params["controls"]["load data"]:

    # load parameters
    method = study_params["controls"]["method"]
    func = study_params["controls"]["function to approximate"]
    grid_size = study_params["controls"]["grid size"]
    if method == "linear":
        iterations = study_params["linear"]["interpolation_points"]
    elif method == "smolyak":
        iterations = study_params["smolyak"]["mu"]
    elif method == "spline":
        iterations = study_params["spline"]["interpolation_points"]
    interpolator = study_params[method]["interpolator"]

    # initiate dict to store results
    results = {}
    results[method] = {"rmse": {}, "runtime": {}, "gridpoints": {}}

    for n_vars in study_params["controls"]["variables"]:

        # generate grid
        n_gridpoints = study_params["grid"]["n_gridpoints"][grid_size]
        grid_min = study_params["grid"]["lower bounds"][:n_vars]
        grid_max = study_params["grid"]["upper bounds"][:n_vars]
        dims_state_grid = np.array(object=[n_gridpoints] * n_vars, dtype=int,)
        n_states = dims_state_grid.prod()
        grid = get_grid(dims_state_grid, grid_min, grid_max)
        index = np.array(object=range(n_states))

        # load or calculate actuals
        _, results_calc = get_data(func, grid_size, n_vars, dims_state_grid, grid)

        # initiate objects to store results
        rmse_tmp = []
        runtime_tmp = []
        n_gridpoints_effective_tmp = []

        # iterate over settings
        for iteration in range(len(iterations)):
            print(f"dimension: {n_vars}; iteration: {iteration + 1}")

            # adjust interpolation parameters
            if study_params["controls"]["method"] == "linear":
                interp_params["linear"]["interpolation_points"] = study_params[
                    "spline"
                ]["interpolation_points"][iteration]
            elif study_params["controls"]["method"] == "spline":
                interp_params["spline"]["interpolation_points"] = study_params[
                    "spline"
                ]["interpolation_points"][iteration]
            elif study_params["controls"]["method"] == "smolyak":
                interp_params["smolyak"]["mu"] = study_params["smolyak"]["mu"][
                    iteration
                ]

            # interpolate and capture computation time
            start = time()
            results_interp, n_gridpoints_effective = interpolator(
                grid, func, interp_params
            )
            stop = time()

            # assess interpolation accuracy
            rmse_iter = root_mean_squared_error(results_interp, results_calc)

            # print and store results
            print("root mean squared error: " + method + f" {rmse_iter}")
            print("computation time: " + method + " {}".format(stop - start))
            print(f"gridpoints: {n_gridpoints_effective}")

            rmse_tmp.append(rmse_iter)
            runtime_tmp.append(stop - start)
            n_gridpoints_effective_tmp.append(n_gridpoints_effective)

        results[method]["rmse"][n_vars] = np.array(object=rmse_tmp)
        results[method]["runtime"][n_vars] = np.array(object=runtime_tmp)
        results[method]["gridpoints"][n_vars] = np.array(
            object=n_gridpoints_effective_tmp
        )

    # save results to file
    file_to_store = open(
        "C:/Users/simon/Documents/Uni/3_Bonn/3_WiSe19-20/topics_SBE/"
        "3_project/student-project-simonjheiler/results/sandbox/test_results.pkl",
        "wb",
    )
    pickle.dump(dict, file_to_store)
    file_to_store.close()

elif study_params["controls"]["load data"]:
    method = study_params["controls"]["method"]
    stored_results = open(
        "C:/Users/simon/Documents/Uni/3_Bonn/3_WiSe19-20/topics_SBE/"
        "3_project/student-project-simonjheiler/results/sandbox/test_results.pkl",
        "rb",
    )
    results = pickle.load(stored_results)

# plot results
plot_legend = []
plot_x = []
plot_y = []
for n_vars in study_params["controls"]["variables"]:
    plot_legend.append(n_vars)
    plot_x.append(results[method]["gridpoints"][n_vars])
    plot_y.append(results[method]["rmse"][n_vars])

for idx in range(len(study_params["controls"]["variables"])):
    plt.plot(plot_x[idx], plot_y[idx])

plt.xscale("log")
plt.yscale("log")
plt.xlabel("number of interpolation points (log axis)")
plt.ylabel("root mean squared error (log axis)")
plt.legend(plot_legend)
plt.title("Interpolation accuracy (" + grid_size + " grid)")
plt.show()
