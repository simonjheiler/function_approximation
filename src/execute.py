"""Collection of functions to run simulation studies and plot results"""
from time import time

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa:F401

import src.functions_to_approximate as functions
import src.interpolate as interpolators
from src.auxiliary import get_grid
from src.auxiliary import get_interpolation_points
from src.auxiliary import rmse as root_mean_squared_error


def execute_study(study_params, interpolation_params):
    """Run the simulation study with parameters *study_params*.

    ...

    Parameters
    ----------
    study_params: dict
        ...
    interpolation_params: dict
        ...

    Returns
    -------
    results: dict
        ...

    """

    # load parameters
    interpolation_method = study_params["controls"]["interpolation method"]
    func_name = study_params["controls"]["function to approximate"]
    func_name_short = func_name[: func_name.find("_")]
    interpolator_name = study_params[interpolation_method]["interpolator"]
    grid_density = study_params["controls"]["grid density"]
    grid_method = study_params["controls"]["grid method"]
    iterations = study_params["controls"]["iterations"]
    n_interpolation_points = study_params["controls"][
        "number of points for accuracy check"
    ]
    accuracy_check_seed = study_params["controls"]["seed for accuracy check"]

    # set grid parameters
    grid_params = {}
    grid_params["orders"] = study_params["grid"]["orders"][grid_density]
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
            # print(f"dimension: {dims}; iteration: {iteration + 1}")

            # adjust interpolation parameters
            interpolation_params[interpolation_method]["grid method"] = grid_method
            interpolation_params[interpolation_method][
                "evaluate off-grid"
            ] = study_params["controls"]["evaluate off-grid"]
            if interpolation_method == "linear":
                interpolation_params["linear"]["sparse grid level"] = study_params[
                    "linear"
                ]["sparse grid levels"][iteration]
                interpolation_params["linear"]["interpolation points"] = study_params[
                    "linear"
                ]["interpolation points"][iteration]
            elif interpolation_method == "spline":
                interpolation_params["spline"]["interpolation points"] = study_params[
                    "spline"
                ]["interpolation points"][iteration]
            elif interpolation_method == "smolyak":
                interpolation_params["smolyak"]["sparse grid level"] = study_params[
                    "smolyak"
                ]["sparse grid levels"][iteration]
            elif interpolation_method == "sparse":
                interpolation_params["sparse"]["sparse grid level"] = study_params[
                    "sparse"
                ]["sparse grid levels"][iteration]

            # interpolate and capture computation time
            start = time()
            results_interp, n_gridpoints_effective = interpolator(
                interpolation_points, grid, func, interpolation_params
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


def plot_results(results, study_params):
    """Plot results for a single simulation study.

    ...

    Parameters
    ----------
    results: dict
        ...
    study_params: dict
        ...

    Returns
    -------
    None

    """
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
        "Interpolation accuracy (" + study_params["controls"]["grid density"] + " grid)"
    )
    plt.show()

    return


def compare_fit_2d_iter(study_params, interpolation_params, iteration):
    """Plot fit for 2D function with givel level of accuracy.

    ...

    Parameters
    ----------
    study_params: dict
        ...
    iteration: int
        ...

    Returns
    -------
    None

    """

    # set interpolation parameters
    dims = 2
    interpolation_method = study_params["controls"]["interpolation method"]
    func_name = study_params["controls"]["function to approximate"]
    func_name_short = func_name[: func_name.find("_")]
    interpolator_name = study_params[interpolation_method]["interpolator"]
    grid_density = study_params["controls"]["grid density"]
    grid_method = study_params["controls"]["grid method"]
    n_interpolation_points = study_params["controls"][
        "number of points for accuracy check"
    ]
    accuracy_check_seed = study_params["controls"]["seed for accuracy check"]

    # set grid parameters
    grid_params = {}
    grid_params["orders"] = study_params["grid"]["orders"][grid_density]
    grid_params["lower bounds"] = study_params["grid"]["lower bounds"][func_name_short]
    grid_params["upper bounds"] = study_params["grid"]["upper bounds"][func_name_short]

    # set functionals for function to approximate and interpolator
    func = getattr(functions, func_name)
    interpolator = getattr(interpolators, interpolator_name)

    # generate grid / state space
    grid, index = get_grid(grid_params, dims)

    # generate grid for plotting
    n_plot_x = n_plot_y = 100
    n_plot = n_plot_x * n_plot_y
    x = np.linspace(
        grid_params["lower bounds"][0], grid_params["upper bounds"][0], n_plot_x
    )
    y = np.linspace(
        grid_params["lower bounds"][1], grid_params["upper bounds"][1], n_plot_y
    )
    X, Y = np.meshgrid(x, y)
    plot_grid = np.asarray([X.reshape(n_plot), Y.reshape(n_plot)]).T
    func_on_plot_grid_actual = func(plot_grid)
    func_on_plot_grid_actual = np.asarray(func_on_plot_grid_actual).reshape(
        (n_plot_x, n_plot_y)
    )

    # adjust interpolation parameters
    interpolation_params[interpolation_method]["grid method"] = grid_method
    interpolation_params[interpolation_method]["evaluate off-grid"] = study_params[
        "controls"
    ]["evaluate off-grid"]
    if interpolation_method == "linear":
        interpolation_params["linear"]["sparse grid level"] = study_params["linear"][
            "sparse grid levels"
        ][iteration]
        interpolation_params["linear"]["interpolation points"] = study_params["linear"][
            "interpolation points"
        ][iteration]
    elif interpolation_method == "spline":
        interpolation_params["spline"]["interpolation points"] = study_params["spline"][
            "interpolation points"
        ][iteration]
    elif interpolation_method == "smolyak":
        interpolation_params["smolyak"]["sparse grid level"] = study_params["smolyak"][
            "sparse grid levels"
        ][iteration]
    elif interpolation_method == "sparse":
        interpolation_params["sparse"]["sparse grid level"] = study_params["sparse"][
            "sparse grid levels"
        ][iteration]

    # interpolate and capture computation time
    start = time()
    func_on_plot_grid_interpolated, n_gridpoints_effective = interpolator(
        plot_grid, grid, func, interpolation_params
    )
    stop = time()
    runtime = stop - start
    func_on_plot_grid_interpolated = np.asarray(func_on_plot_grid_interpolated).reshape(
        (n_plot_x, n_plot_y)
    )

    # calculate approximation error
    interpolation_points = get_interpolation_points(
        n_interpolation_points, grid, accuracy_check_seed,
    )
    results_interp, n_gridpoints_effective = interpolator(
        interpolation_points, grid, func, interpolation_params
    )
    results_calc = func(interpolation_points)
    rmse = root_mean_squared_error(results_interp, results_calc)

    # plot results
    print(f"grid method: {grid_method}")
    print(f"total number of interpolation points: {n_gridpoints_effective}")
    print(f"runtime for interpolation: {runtime}")
    print(f"root mean squared error: {rmse}")

    fig = plt.figure(figsize=(16, 6))

    ax = fig.add_subplot(
        131, projection="3d", title=f"{func_name_short} function, dim=2 (calculated)"
    )
    ax.plot_surface(
        X, Y, func_on_plot_grid_actual, rstride=1, cstride=1, cmap=plt.cm.magma
    )
    ax = fig.add_subplot(
        132, projection="3d", title=f"{func_name_short} function, dim=2 (interpolated)"
    )
    ax.plot_surface(
        X, Y, func_on_plot_grid_interpolated, rstride=1, cstride=1, cmap=plt.cm.magma
    )
    ax = fig.add_subplot(
        133, projection="3d", title=f"{func_name_short} function, dim=2 (error)"
    )
    ax.plot_surface(
        X,
        Y,
        func_on_plot_grid_actual - func_on_plot_grid_interpolated,
        rstride=1,
        cstride=1,
        cmap=plt.cm.magma,
    )
    ax.set_zlim(0.0, ax.get_zlim()[1] * 2)

    plt.show()
    print("-------------------------------------------------------------------")

    return


def compare_results(results_1, results_2, study_params_1, study_params_2):
    """Plot results for a two simulation studies for comparison.

    ...

    Parameters
    ----------
    results_1: dict
        ...
    study_params_1: dict
        ...
    results_2: dict
        ...
    study_params_2: dict
        ...

    Returns
    -------
    None

    """

    plot_legend = []

    dims = list(
        set(study_params_1["controls"]["dims"]).intersection(
            study_params_2["controls"]["dims"]
        )
    )
    # pdb.set_trace()
    for dim in dims:
        plt.plot(
            results_1["gridpoints"][dim],
            results_1["rmse"][dim],
            "-",
            results_2["gridpoints"][dim],
            results_2["rmse"][dim],
            ":",
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("number of interpolation points (log axis)")
    plt.ylabel("root mean squared error (log axis)")
    plt.legend(plot_legend)
    plt.title("Interpolation accuracy")
    plt.show()

    return
