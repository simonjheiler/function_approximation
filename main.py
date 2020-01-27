import json
import os
import pdb  # noqa:F401
import pickle
import sys

root = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.insert(0, root)

import matplotlib.pyplot as plt
import numpy as np
import src.interpolate as interpolators
import src.functions_to_approximate as functions

from mpl_toolkits.mplot3d import Axes3D  # noqa:F401
from src.auxiliary import get_grid
from src.auxiliary import get_interpolation_points
from src.auxiliary import rmse as root_mean_squared_error
from time import time


#########################################################################
# PARAMETERS
#########################################################################

# load default interpolation parameters
with open("./src/interp_params.json") as json_file:
    interp_params = json.load(json_file)

# load default study parameters
with open("./src/study_params.json") as json_file:
    study_params = json.load(json_file)


#########################################################################
# EXECUTION
#########################################################################


if study_params["controls"]["load data"] == "False":

    # load parameters
    interpolation_method = study_params["controls"]["interpolation method"]
    func = getattr(functions, study_params["controls"]["function to approximate"])
    grid_size = study_params["controls"]["grid size"]
    grid_method = study_params["controls"]["grid_method"]
    iterations = study_params[interpolation_method]["iterations"]
    interpolator = getattr(
        interpolators, study_params[interpolation_method]["interpolator"]
    )
    n_interpolation_points = study_params["controls"][
        "number of points for accuracy check"
    ]
    accuracy_check_seed = study_params["controls"]["seed for accuracy check"]

    # initiate dict to store results
    results = {}
    results[interpolation_method] = {"rmse": {}, "runtime": {}, "gridpoints": {}}

    for n_vars in study_params["controls"]["variables"]:

        # generate grid
        n_gridpoints = study_params["grid"]["zhou"]["n_gridpoints"][grid_size]
        grid_min = np.array(
            object=study_params["grid"]["zhou"]["lower bounds"][:n_vars]
        )
        grid_max = np.array(
            object=study_params["grid"]["zhou"]["upper bounds"][:n_vars]
        )
        dims_state_grid = np.array(object=[n_gridpoints] * n_vars, dtype=int,)
        n_states = dims_state_grid.prod()
        grid = get_grid(dims_state_grid, grid_min, grid_max)
        index = np.array(object=range(n_states))

        # get interpolation points
        interpolation_points = get_interpolation_points(
            n_interpolation_points, grid, accuracy_check_seed,
        )

        # get results on interpolation points
        results_calc = func(interpolation_points)

        # initiate objects to store results
        rmse_tmp = []
        runtime_tmp = []
        n_gridpoints_effective_tmp = []

        # iterate over settings
        for iteration in range(iterations):
            print(f"dimension: {n_vars}; iteration: {iteration + 1}")

            # adjust interpolation parameters
            interp_params[interpolation_method]["grid_method"] = grid_method
            if study_params["controls"]["interpolation method"] == "linear":
                interp_params["linear"]["interpolation_points"] = study_params[
                    "linear"
                ]["interpolation_points"][iteration]
                interp_params["linear"]["sparse_grid_levels"] = study_params["linear"][
                    "sparse_grid_levels"
                ][iteration]
            elif study_params["controls"]["interpolation method"] == "spline":
                interp_params["spline"]["interpolation_points"] = study_params[
                    "spline"
                ]["interpolation_points"][iteration]
            elif study_params["controls"]["interpolation method"] == "smolyak":
                interp_params["smolyak"]["mu"] = study_params["smolyak"]["mu"][
                    iteration
                ]

            # interpolate and capture computation time
            start = time()
            results_interp, n_gridpoints_effective = interpolator(
                interpolation_points, grid, func, interp_params
            )
            stop = time()

            # assess interpolation accuracy
            rmse_iter = root_mean_squared_error(results_interp, results_calc)

            # print and store results
            print("root mean squared error: " + interpolation_method + f" {rmse_iter}")
            print(
                "computation time: " + interpolation_method + " {}".format(stop - start)
            )
            print(f"gridpoints: {n_gridpoints_effective}")

            rmse_tmp.append(rmse_iter)
            runtime_tmp.append(stop - start)
            n_gridpoints_effective_tmp.append(n_gridpoints_effective)

        results[interpolation_method]["rmse"][n_vars] = np.array(object=rmse_tmp)
        results[interpolation_method]["runtime"][n_vars] = np.array(object=runtime_tmp)
        results[interpolation_method]["gridpoints"][n_vars] = np.array(
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

elif study_params["controls"]["load data"] == "True":
    interpolation_method = study_params["controls"]["method"]
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
    plot_x.append(results[interpolation_method]["gridpoints"][n_vars])
    plot_y.append(results[interpolation_method]["rmse"][n_vars])

for idx in range(len(study_params["controls"]["variables"])):
    plt.plot(plot_x[idx], plot_y[idx])

plt.xscale("log")
plt.yscale("log")
plt.xlabel("number of interpolation points (log axis)")
plt.ylabel("root mean squared error (log axis)")
plt.legend(plot_legend)
plt.title("Interpolation accuracy (" + grid_size + " grid)")
plt.show()


# if coord.shape[1] == 2:
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#
#     xs = coord[:, 0]
#     ys = coord[:, 1]
#     ax.scatter(xs, ys, marker="o")
#
#     ax.set_xlabel('X Label')
#     ax.set_ylabel('Y Label')
#
#     plt.show()
# elif coord.shape[1] == 3:
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#
#     xs = coord[:, 0]
#     ys = coord[:, 1]
#     zs = coord[:, 2]
#     ax.scatter(xs, ys, zs, marker="o")
#
#     ax.set_xlabel('X Label')
#     ax.set_ylabel('Y Label')
#     ax.set_zlabel('Z Label')
#
#     plt.show()
