import numpy as np

from src.functions_to_approximate import borehole_numba as borehole  # noqa:F401
from src.functions_to_approximate import zhou_readable as zhou  # noqa:F401
from src.interpolate import interpolate_linear
from src.interpolate import interpolate_smolyak
from src.interpolate import interpolate_spline

# set defaults for interpolation parameters
interp_params = {}
interp_params["linear"] = {}
interp_params["linear"]["seed"] = 123
interp_params["linear"]["grid_method"] = "regular"
interp_params["linear"]["interpolation_points"] = 3
interp_params["linear"]["n_interpolation_points"] = 3
interp_params["smolyak"] = {}
interp_params["smolyak"]["mu"] = 1
interp_params["spline"] = {}
interp_params["spline"]["interpolation_points"] = 3

# set study parameters
study_params = {}

study_params["controls"] = {
    "load data": False,
    "method": "spline",
    "grid size": "large",
    "variables": [2],
    "function to approximate": zhou,
}


study_params["grid"] = {}
study_params["grid"]["borehole"] = {
    "lower bounds": np.array(
        object=[63070.0, 990.0, 700.0, 100.0, 0.05, 1120.0, 1500.0, 63.1], dtype=float,
    ),
    "upper bounds": np.array(
        object=[115600.0, 1110.0, 820.0, 50000.0, 0.15, 1680.0, 15000.0, 116.0],
        dtype=float,
    ),
    "input default": np.array(
        object=[89335.0, 1050.0, 760.0, 25050.0, 0.1, 1400.0, 8250.0, 89.55],
        dtype=float,
    ),
    "n_gridpoints": {"small": 5, "medium": 10, "large": 100},
}
study_params["grid"]["zhou"] = {
    "lower bounds": np.array(
        object=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float,
    ),
    "upper bounds": np.array(
        object=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float,
    ),
    "input default": np.array(
        object=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=float,
    ),
    "n_gridpoints": {"small": 5, "medium": 10, "large": 100},
}

study_params["linear"] = {
    "interpolator": interpolate_linear,
    "share_not_interpolated": [0.0, 0.2, 0.4],
    "interpolation_points": [2, 3, 5, 7, 9, 11],
    "iterations": 3,
}
study_params["smolyak"] = {
    "interpolator": interpolate_smolyak,
    "mu": [1, 2, 3],
    "iterations": 3,
}
study_params["spline"] = {
    "interpolator": interpolate_spline,
    "interpolation_points": [2, 3, 5, 7, 9, 11],
    "iterations": 3,
}
