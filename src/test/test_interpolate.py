import json
import os
import pdb  # noqa:F401
import sys

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root)

import numpy as np
import pytest

from numpy.testing import assert_array_equal
from src.interpolate import get_interpolation_grid_regular
from src.interpolate import get_interpolation_grid_sparse

# load default interpolation parameters
with open("./src/interp_params.json") as json_file:
    interp_params = json.load(json_file)

#########################################################################
# FIXTURES
#########################################################################


@pytest.fixture
def setup_get_interpolation_grid():
    out = {}
    out["n_dims"] = 2
    out["grid_min"] = np.array(object=[1.0, 1.0], dtype=float)
    out["grid_max"] = np.array(object=[9.0, 9.0], dtype=float)
    out["interp_params"] = interp_params
    return out


#########################################################################
# TESTS
#########################################################################


def test_get_interpolation_grid_regular_2D_small(setup_get_interpolation_grid):
    setup_get_interpolation_grid["interp_params"]["linear"]["interpolation_points"] = 2
    grid_interp_expected = np.array(
        object=[[1.0, 1.0], [1.0, 9.0], [9.0, 1.0], [9.0, 9.0]], dtype=float,
    )
    n_gridpoints_expected = 4
    grid_interp_actual, n_gridpoints_actual = get_interpolation_grid_regular(
        **setup_get_interpolation_grid
    )
    assert_array_equal(grid_interp_actual, grid_interp_expected)
    assert n_gridpoints_actual == n_gridpoints_expected


def test_get_interpolation_grid_regular_2D_medium(setup_get_interpolation_grid):
    setup_get_interpolation_grid["interp_params"]["linear"]["interpolation_points"] = 5
    grid_interp_expected = np.array(
        object=[
            [1.0, 1.0],
            [1.0, 3.0],
            [1.0, 5.0],
            [1.0, 7.0],
            [1.0, 9.0],
            [3.0, 1.0],
            [3.0, 3.0],
            [3.0, 5.0],
            [3.0, 7.0],
            [3.0, 9.0],
            [5.0, 1.0],
            [5.0, 3.0],
            [5.0, 5.0],
            [5.0, 7.0],
            [5.0, 9.0],
            [7.0, 1.0],
            [7.0, 3.0],
            [7.0, 5.0],
            [7.0, 7.0],
            [7.0, 9.0],
            [9.0, 1.0],
            [9.0, 3.0],
            [9.0, 5.0],
            [9.0, 7.0],
            [9.0, 9.0],
        ],
        dtype=float,
    )
    n_gridpoints_expected = 25
    grid_interp_actual, n_gridpoints_actual = get_interpolation_grid_regular(
        **setup_get_interpolation_grid
    )
    assert_array_equal(grid_interp_actual, grid_interp_expected)
    assert n_gridpoints_actual == n_gridpoints_expected


def test_get_interpolation_grid_regular_3D_small(setup_get_interpolation_grid):
    setup_get_interpolation_grid["n_dims"] = 3
    setup_get_interpolation_grid["grid_min"] = np.array(
        object=[1.0, 1.0, 1.0], dtype=float
    )
    setup_get_interpolation_grid["grid_max"] = np.array(
        object=[9.0, 9.0, 9.0], dtype=float
    )
    setup_get_interpolation_grid["interp_params"]["linear"]["interpolation_points"] = 2
    grid_interp_expected = np.array(
        object=[
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 9.0],
            [1.0, 9.0, 1.0],
            [1.0, 9.0, 9.0],
            [9.0, 1.0, 1.0],
            [9.0, 1.0, 9.0],
            [9.0, 9.0, 1.0],
            [9.0, 9.0, 9.0],
        ],
        dtype=float,
    )
    n_gridpoints_expected = 8
    grid_interp_actual, n_gridpoints_actual = get_interpolation_grid_regular(
        **setup_get_interpolation_grid
    )
    assert_array_equal(grid_interp_actual, grid_interp_expected)
    assert n_gridpoints_actual == n_gridpoints_expected


def test_get_interpolation_grid_sparse_2D_2L(setup_get_interpolation_grid):
    setup_get_interpolation_grid["interp_params"]["linear"]["sparse_grid_levels"] = 2
    grid_interp_expected = np.array(
        object=[[5.0, 5.0], [5.0, 7.0], [5.0, 3.0], [7.0, 5.0], [3.0, 5.0]],
        dtype=float,
    )
    n_gridpoints_expected = 5
    grid_interp_actual, n_gridpoints_actual = get_interpolation_grid_sparse(
        **setup_get_interpolation_grid
    )
    assert_array_equal(grid_interp_actual, grid_interp_expected)
    assert n_gridpoints_actual == n_gridpoints_expected
