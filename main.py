import os
import pdb  # noqa:F401
import pickle
import sys

root = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.insert(0, root)

import matplotlib.pyplot as plt
import numpy as np
from time import time

from src.functions_to_approximate import borehole_numba as borehole  # noqa:F401
from src.functions_to_approximate import zhou_readable as zhou  # noqa:F401
from src.auxiliary import get_data
from src.auxiliary import get_grid
from src.auxiliary import rmsre as root_mean_squared_relative_error
from src.parameters import study_params
from src.parameters import interp_params

#########################################################################
# PARAMETERS
#########################################################################

study_params["controls"] = {
    "load data": False,
    "method": "linear",
    "grid size": "small",
    "variables": [2, 3, 4, 5, 6],
    "function to approximate": zhou,
}


#########################################################################
# EXECUTION
#########################################################################

if not study_params["controls"]["load data"]:

    # load parameters
    method = study_params["controls"]["method"]
    func = study_params["controls"]["function to approximate"]
    grid_size = study_params["controls"]["grid size"]
    iterations = study_params[method]["iterations"]
    interpolator = study_params[method]["interpolator"]

    # initiate dict to store results
    results = {}
    results[method] = {"rmse": {}, "runtime": {}, "gridpoints": {}}

    for n_vars in study_params["controls"]["variables"]:

        # generate grid
        n_gridpoints = study_params["grid"]["zhou"]["n_gridpoints"][grid_size]
        grid_min = study_params["grid"]["zhou"]["lower bounds"][:n_vars]
        grid_max = study_params["grid"]["zhou"]["upper bounds"][:n_vars]
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
        for iteration in range(iterations):
            print(f"dimension: {n_vars}; iteration: {iteration + 1}")

            # adjust interpolation parameters
            if study_params["controls"]["method"] == "linear":
                interp_params["linear"]["interpolation_points"] = study_params[
                    "linear"
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
            rmse_iter = root_mean_squared_relative_error(results_interp, results_calc)

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

#########################################################################
# PLOTS
#########################################################################

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
