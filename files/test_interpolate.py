# import sys
import numpy as np
import pytest

from files.interpolate import inputs_from_state

# import pandas as pd
# from pandas.testing import assert_frame_equal
# from pandas.testing import assert_series_equal

# from files.interpolate import get_grids_values
# from files.interpolate import get_dims_state_grid
# from files.interpolate import get_grids_values
# from files.interpolate import get_grids_indices
# from files.interpolate import get_states_grid_dense
# from files.interpolate import get_corner_states
# from files.interpolate import state_from_id
# from files.interpolate import states_from_ids_batch
# from files.interpolate import inputs_from_ids_batch
# from files.interpolate import evaluation_batch
# from files.interpolate import state_to_id
# from files.interpolate import states_to_ids_batch
# from files.interpolate import get_not_interpolated_indicator_random
# from files.interpolate import get_data
# from files.interpolate import interpolate_linear


@pytest.fixture
def setup_inputs_from_state():
    out = {}
    out["state"] = np.array([0, 0, 0])
    out["grids_values"] = np.array(object=[3.0, 2.0, 4.0][4.0, 3.0, 5.0],)
    return out


def test_inputs_from_state(setup_inputs_from_state):

    actual = inputs_from_state(**setup_inputs_from_state)
    assert actual == 23
