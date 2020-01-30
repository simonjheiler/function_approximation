import os
import pdb  # noqa F401
import sys

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root)

import numpy as np
import pytest

from src.auxiliary import get_corner_points
from src.auxiliary import get_grid
from src.auxiliary import get_local_grid_step
from src.auxiliary import inputs_from_ids_batch
from src.auxiliary import inputs_from_state
from src.auxiliary import mse
from src.auxiliary import msre
from src.auxiliary import rmse
from src.auxiliary import rmsre
from src.auxiliary import state_from_id
from src.auxiliary import state_to_id
from src.auxiliary import states_from_ids_batch
from src.auxiliary import states_to_ids_batch
from numpy.testing import assert_array_equal
from numpy.testing import assert_equal
from numpy.testing import assert_almost_equal


#########################################################################
# FIXTURES
#########################################################################


@pytest.fixture
def setup_inputs_from_state():
    out = {}
    out["state"] = np.array([0, 0, 0])
    out["grid"] = np.array(
        object=[[1.0, 2.0, 3.0], [4.0, 5.0], [3.0, 5.0, 6.0, 8.0, 10.0]]
    )
    return out


@pytest.fixture
def setup_state_from_id():
    out = {}
    out["index"] = 0
    out["dims_state_grid"] = np.array(object=[3, 2, 5])
    return out


@pytest.fixture
def setup_states_from_ids_batch():
    out = {}
    out["index"] = np.array(object=[0, 14, 29])
    out["dims_state_grid"] = np.array(object=[3, 2, 5])
    return out


@pytest.fixture
def setup_inputs_from_ids_batch():
    out = {}
    out["index"] = np.array(object=[0, 14, 29])
    out["dims_state_grid"] = np.array(object=[3, 2, 5])
    out["grid"] = np.array(
        object=[[1.0, 2.0, 3.0], [4.0, 5.0], [3.0, 5.0, 6.0, 8.0, 10.0]]
    )
    return out


@pytest.fixture
def setup_state_to_id():
    out = {}
    out["dims_state_grid"] = np.array(object=[3, 2, 5])
    return out


@pytest.fixture
def setup_get_grid():
    out = {}
    out["grid_params"] = {}
    out["grid_params"]["orders"] = [3, 2, 5]
    out["grid_params"]["lower bounds"] = [1.0, 1.0, 1.0]
    out["grid_params"]["upper bounds"] = [2.0, 2.0, 5.0]
    out["dims"] = 3
    return out


@pytest.fixture
def setup_get_local_grid_step():
    out = {}
    out["point"] = np.array(object=[1.7, 1.5, 2.5])
    out["grid"] = {
        0: np.array(object=[1.0, 1.5, 2.0]),
        1: np.array(object=[1.0, 2.0]),
        2: np.array(object=[1.0, 2.0, 3.0, 4.0, 5.0]),
    }
    return out


@pytest.fixture
def setup_approximation_error():
    out = {}
    out["x1"] = np.array(object=[0.1, 0.2, 0.3], dtype=float)
    out["x2"] = np.array(object=[0.2, 0.4, 0.5], dtype=float)
    return out


#########################################################################
# TESTS
#########################################################################


def test_inputs_from_state(setup_inputs_from_state):
    expected = np.array(object=[1.0, 4.0, 3.0])
    actual = inputs_from_state(**setup_inputs_from_state)
    assert_array_equal(actual, expected)


def test_state_from_id_first(setup_state_from_id):
    expected = np.array(object=[0, 0, 0])
    actual = state_from_id(**setup_state_from_id)
    assert_array_equal(actual, expected)


def test_state_from_id_middle(setup_state_from_id):
    setup_state_from_id["index"] = 14
    expected = np.array(object=[1, 0, 4])
    actual = state_from_id(**setup_state_from_id)
    assert_array_equal(actual, expected)


def test_state_from_id_last(setup_state_from_id):
    setup_state_from_id["index"] = 29
    expected = np.array(object=[2, 1, 4])
    actual = state_from_id(**setup_state_from_id)
    assert_array_equal(actual, expected)


def test_state_from_id_high_dim(setup_state_from_id):
    setup_state_from_id["index"] = 242
    setup_state_from_id["dims_state_grid"] = np.array(object=[3, 7, 4, 3])
    expected = np.array(object=[2, 6, 0, 2])
    actual = state_from_id(**setup_state_from_id)
    assert_array_equal(actual, expected)


def test_states_from_ids_batch(setup_states_from_ids_batch):
    expected = np.array(object=[[0, 0, 0], [1, 0, 4], [2, 1, 4]])
    actual = states_from_ids_batch(**setup_states_from_ids_batch)
    assert_array_equal(actual, expected)


def test_inputs_from_ids_batch(setup_inputs_from_ids_batch):
    expected = np.array(object=[[1.0, 4.0, 3.0], [2.0, 4.0, 10.0], [3.0, 5.0, 10.0]])
    actual = inputs_from_ids_batch(**setup_inputs_from_ids_batch)
    assert_array_equal(actual, expected)


def test_state_to_id_first(setup_state_to_id):
    setup_state_to_id["state"] = np.array(object=[0, 0, 0])
    expected = 0
    actual = state_to_id(**setup_state_to_id)
    assert actual == expected


def test_state_to_id_middle(setup_state_to_id):
    setup_state_to_id["state"] = np.array(object=[1, 0, 4])
    expected = 14
    actual = state_to_id(**setup_state_to_id)
    assert actual == expected


def test_state_to_id_last(setup_state_to_id):
    setup_state_to_id["state"] = np.array(object=[2, 1, 4])
    expected = 29
    actual = state_to_id(**setup_state_to_id)
    assert actual == expected


def test_states_to_ids_batch(setup_state_to_id):
    setup_state_to_id["states"] = np.array(object=[[0, 0, 0], [1, 0, 4], [2, 1, 4]])
    expected = [0, 14, 29]
    actual = states_to_ids_batch(**setup_state_to_id)
    assert actual == expected


def test_get_corner_points():
    setup_get_corner_points = {}
    setup_get_corner_points["grid"] = {
        0: np.array(object=[0.0, 2.0]),
        1: np.array(object=[0.0, 1.0]),
        2: np.array(object=[0.0, 4.0]),
    }
    expected = np.array(
        object=[
            [0, 0, 0],
            [0, 0, 4],
            [0, 1, 0],
            [0, 1, 4],
            [2, 0, 0],
            [2, 0, 4],
            [2, 1, 0],
            [2, 1, 4],
        ]
    )
    actual = get_corner_points(**setup_get_corner_points)
    assert_array_equal(actual, expected)


def test_get_grid(setup_get_grid):
    grid_expected = {
        0: np.array([1.0, 1.5, 2.0]),
        1: np.array([1.0, 2.0]),
        2: np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    }
    index_expected = np.array(object=list(range(2 * 3 * 5)))
    grid_actual, index_actual = get_grid(**setup_get_grid)
    assert_equal(grid_actual, grid_expected)
    assert_equal(index_actual, index_expected)


def test_mse(setup_approximation_error):
    expected = np.mean([0.01, 0.04, 0.04])
    actual = mse(**setup_approximation_error)
    assert_almost_equal(actual, expected, decimal=12)


def test_msre(setup_approximation_error):
    expected = np.mean([0.01 / 0.1, 0.04 / 0.2, 0.04 / 0.3])
    actual = msre(**setup_approximation_error)
    assert_almost_equal(actual, expected, decimal=12)


def test_rmse(setup_approximation_error):
    expected = np.sqrt(0.03)
    actual = rmse(**setup_approximation_error)
    assert_almost_equal(actual, expected, decimal=12)


def test_rmsre(setup_approximation_error):
    expected = np.sqrt(2.6 / 18)
    actual = rmsre(**setup_approximation_error)
    assert_almost_equal(actual, expected, decimal=12)


def test_get_local_grid_step(setup_get_local_grid_step):
    expected = np.array(
        object=[
            [1.5, 1.0, 2.0],
            [1.5, 1.0, 3.0],
            [1.5, 2.0, 2.0],
            [1.5, 2.0, 3.0],
            [2.0, 1.0, 2.0],
            [2.0, 1.0, 3.0],
            [2.0, 2.0, 2.0],
            [2.0, 2.0, 3.0],
        ],
        dtype=float,
    )
    actual = get_local_grid_step(**setup_get_local_grid_step)
    assert_array_equal(actual, expected)
