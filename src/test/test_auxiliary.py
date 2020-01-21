import os
import sys

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root)

import numpy as np
import pytest

from src.auxiliary import get_corner_states
from src.auxiliary import get_grid
from src.auxiliary import inputs_from_ids_batch
from src.auxiliary import inputs_from_state
from src.auxiliary import state_from_id
from src.auxiliary import state_to_id
from src.auxiliary import states_from_ids_batch
from src.auxiliary import states_to_ids_batch
from numpy.testing import assert_array_equal
from numpy.testing import assert_equal

# from src.auxiliary import get_dims_state_grid
# import pandas as pd
# from pandas.testing import assert_frame_equal
# from pandas.testing import assert_series_equal


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
    out["dims_state_grid"] = np.array(object=[3, 2, 5])
    out["grid_min"] = np.array([1.0, 1.0, 1.0])
    out["grid_max"] = np.array([2.0, 2.0, 5.0])
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


def test_get_corner_states():
    setup_get_corner_states = {}
    setup_get_corner_states["dims_state_grid"] = np.array(object=[3, 2, 5])
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
    actual = get_corner_states(**setup_get_corner_states)
    assert_array_equal(actual, expected)


def test_get_grid(setup_get_grid):
    expected = {
        0: np.array([1.0, 1.5, 2.0]),
        1: np.array([1.0, 2.0]),
        2: np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    }
    actual = get_grid(**setup_get_grid)
    assert_equal(actual, expected)
