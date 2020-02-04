import os
import pdb  # noqa:F401
import sys

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root)

import numpy as np
import pytest

from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from src.interpolate import get_interpolation_grid_regular
from src.interpolate import get_interpolation_grid_sparse
from src.interpolate import interpolate_locally_step
from src.interpolate import interpolate_locally_batch

# from src.functions_to_approximate import zhou_readable


#########################################################################
# FIXTURES
#########################################################################


@pytest.fixture
def setup_get_interpolation_grid():
    out = {}
    out["dims"] = 2
    out["grid_min"] = np.array(object=[1.0, 1.0], dtype=float)
    out["grid_max"] = np.array(object=[9.0, 9.0], dtype=float)
    return out


@pytest.fixture
def setup_interpolate_locally_step():
    out = {}
    out["point"] = np.array(object=[0.5, 0.5], dtype=float)
    out["local_grid"] = np.array(
        object=[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=float,
    )
    out["func"] = linear_test_function
    return out


@pytest.fixture
def setup_interpolate_locally_batch():
    out = {}
    out["points"] = np.array(object=[[0.5, 0.5], [0.7, 0.2], [0.2, 0.8]], dtype=float,)
    out["grid"] = {
        0: np.array(object=[0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        1: np.array(object=[0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
    }
    out["func"] = linear_test_function
    return out


#########################################################################
# FUNCTIONS FOR TESTING
#########################################################################


def linear_test_function(input):

    output = np.mean(input, axis=1)

    return output


#########################################################################
# TESTS
#########################################################################


def test_get_interpolation_grid_regular_2D_small(setup_get_interpolation_grid):
    setup_get_interpolation_grid["orders"] = [2, 2]
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
    setup_get_interpolation_grid["orders"] = [5, 5]
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
    setup_get_interpolation_grid["dims"] = 3
    setup_get_interpolation_grid["grid_min"] = np.array(
        object=[1.0, 1.0, 1.0], dtype=float
    )
    setup_get_interpolation_grid["grid_max"] = np.array(
        object=[9.0, 9.0, 9.0], dtype=float
    )
    setup_get_interpolation_grid["orders"] = [2, 2, 2]
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
    setup_get_interpolation_grid["level"] = 2
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


def test_get_interpolation_grid_sparse_2D_3L(setup_get_interpolation_grid):
    setup_get_interpolation_grid["level"] = 3
    grid_interp_expected = np.array(
        object=[
            [5.0, 5.0],
            [5.0, 7.0],
            [5.0, 3.0],
            [5.0, 8.0],
            [5.0, 6.0],
            [5.0, 4.0],
            [5.0, 2.0],
            [7.0, 5.0],
            [7.0, 7.0],
            [7.0, 3.0],
            [3.0, 5.0],
            [3.0, 7.0],
            [3.0, 3.0],
            [8.0, 5.0],
            [6.0, 5.0],
            [4.0, 5.0],
            [2.0, 5.0],
        ],
        dtype=float,
    )
    n_gridpoints_expected = 17
    grid_interp_actual, n_gridpoints_actual = get_interpolation_grid_sparse(
        **setup_get_interpolation_grid
    )
    assert_array_equal(grid_interp_actual, grid_interp_expected)
    assert n_gridpoints_actual == n_gridpoints_expected


def test_get_interpolation_grid_sparse_3D_2L(setup_get_interpolation_grid):
    setup_get_interpolation_grid["dims"] = 3
    setup_get_interpolation_grid["grid_min"] = np.array(
        object=[1.0, 1.0, 1.0], dtype=float
    )
    setup_get_interpolation_grid["grid_max"] = np.array(
        object=[9.0, 9.0, 9.0], dtype=float
    )
    setup_get_interpolation_grid["level"] = 2
    grid_interp_expected = np.array(
        object=[
            [5.0, 5.0, 5.0],
            [5.0, 5.0, 7.0],
            [5.0, 5.0, 3.0],
            [5.0, 7.0, 5.0],
            [5.0, 3.0, 5.0],
            [7.0, 5.0, 5.0],
            [3.0, 5.0, 5.0],
        ],
        dtype=float,
    )
    n_gridpoints_expected = 7
    grid_interp_actual, n_gridpoints_actual = get_interpolation_grid_sparse(
        **setup_get_interpolation_grid
    )
    assert_array_equal(grid_interp_actual, grid_interp_expected)
    assert n_gridpoints_actual == n_gridpoints_expected


def test_interpolate_locally_step(setup_interpolate_locally_step):
    expected = 0.5
    actual = interpolate_locally_step(**setup_interpolate_locally_step)
    assert actual == expected


def test_interpolate_locally_batch(setup_interpolate_locally_batch):
    expected = np.array(object=[0.5, 0.45, 0.5])
    actual = interpolate_locally_batch(**setup_interpolate_locally_batch)
    assert_array_almost_equal(actual, expected, decimal=12)
