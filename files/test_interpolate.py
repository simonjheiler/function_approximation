# import sys
import numpy as np
import pytest
from interpolate import inputs_from_ids_batch
from interpolate import inputs_from_state
from interpolate import state_from_id
from interpolate import states_from_ids_batch
from numpy.testing import assert_array_equal

# import pandas as pd
# from pandas.testing import assert_frame_equal
# from pandas.testing import assert_series_equal

# from interpolate import get_grids_values
# from interpolate import get_dims_state_grid
# from interpolate import get_grids_values
# from interpolate import get_grids_indices
# from interpolate import get_states_grid_dense
# from interpolate import get_corner_states
# from interpolate import evaluation_batch
# from interpolate import state_to_id
# from interpolate import states_to_ids_batch
# from interpolate import get_not_interpolated_indicator_random
# from interpolate import get_data
# from interpolate import interpolate_linear


#########################################################################
# FIXTURES
#########################################################################


@pytest.fixture
def setup_inputs_from_state():
    out = {}
    out["state"] = np.array([0, 0, 0])
    out["grids_values"] = np.array(
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
    out["grids_values"] = np.array(
        object=[[1.0, 2.0, 3.0], [4.0, 5.0], [3.0, 5.0, 6.0, 8.0, 10.0]]
    )
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


def test_states_from_ids_batch(setup_states_from_ids_batch):
    expected = np.array(object=[[0, 0, 0], [1, 0, 4], [2, 1, 4]])
    actual = states_from_ids_batch(**setup_states_from_ids_batch)
    assert_array_equal(actual, expected)


def test_inputs_from_ids_batch(setup_inputs_from_ids_batch):
    expected = np.array(object=[[1.0, 4.0, 3.0], [2.0, 4.0, 10.0], [3.0, 5.0, 10.0]])
    actual = inputs_from_ids_batch(**setup_inputs_from_ids_batch)
    assert_array_equal(actual, expected)
