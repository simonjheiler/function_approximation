import numpy as np

from src.functions_to_approximate import borehole_numba as borehole
from src.interpolate import interpolate_linear
from src.interpolate import interpolate_smolyak
from src.interpolate import interpolate_spline

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
    "method": "spline",
    "grid size": "large",
    "variables": [2],
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
