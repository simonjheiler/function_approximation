import json
import os
import pdb  # noqa:F401
import pickle  # noqa:F401
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


def execute_study(study_params):

    # load parameters
    interpolation_method = study_params["controls"]["interpolation method"]
    func_name = study_params["controls"]["function to approximate"]
    func_name_short = func_name[: func_name.find("_")]
    interpolator_name = study_params[interpolation_method]["interpolator"]
    grid_size = study_params["controls"]["grid size"]
    grid_method = study_params["controls"]["grid method"]
    iterations = study_params[interpolation_method]["iterations"]
    n_interpolation_points = study_params["controls"][
        "number of points for accuracy check"
    ]
    accuracy_check_seed = study_params["controls"]["seed for accuracy check"]

    # set grid parameters
    grid_params = {}
    grid_params["orders"] = study_params["grid"]["orders"][grid_size]
    grid_params["lower bounds"] = study_params["grid"]["lower bounds"][func_name_short]
    grid_params["upper bounds"] = study_params["grid"]["upper bounds"][func_name_short]

    # set functionals for function to approximate and interpolator
    func = getattr(functions, func_name)
    interpolator = getattr(interpolators, interpolator_name)

    # initiate dict to store results
    results = {"rmse": {}, "runtime": {}, "gridpoints": {}}

    for dims in study_params["controls"]["dims"]:

        # generate grid
        grid, index = get_grid(grid_params, dims)

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
            print(f"dimension: {dims}; iteration: {iteration + 1}")

            # adjust interpolation parameters
            interp_params[interpolation_method]["grid method"] = grid_method
            if interpolation_method == "linear":
                interp_params["linear"]["sparse grid levels"] = study_params["linear"][
                    "sparse grid levels"
                ][iteration]
            elif interpolation_method == "spline":
                interp_params["spline"]["interpolation points"] = study_params[
                    "spline"
                ]["interpolation points"][iteration]
            elif interpolation_method == "smolyak":
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

            # store results
            rmse_tmp.append(rmse_iter)
            runtime_tmp.append(stop - start)
            n_gridpoints_effective_tmp.append(n_gridpoints_effective)

        results["rmse"][dims] = np.array(object=rmse_tmp)
        results["runtime"][dims] = np.array(object=runtime_tmp)
        results["gridpoints"][dims] = np.array(object=n_gridpoints_effective_tmp)

    return results


#########################################################################
# PLOTS
#########################################################################


def plot_results(results, study_params):
    plot_legend = []
    plot_x = []
    plot_y = []
    for dims in study_params["controls"]["dims"]:
        plot_legend.append(dims)
        plot_x.append(results["gridpoints"][dims])
        plot_y.append(results["rmse"][dims])

    for idx in range(len(study_params["controls"]["dims"])):
        plt.plot(plot_x[idx], plot_y[idx])

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("number of interpolation points (log axis)")
    plt.ylabel("root mean squared error (log axis)")
    plt.legend(plot_legend)
    plt.title(
        "Interpolation accuracy (" + study_params["controls"]["grid size"] + " grid)"
    )
    plt.show()

    return


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


if __name__ == "__main__":
    results = execute_study(study_params)
    plot_results(results, study_params)
