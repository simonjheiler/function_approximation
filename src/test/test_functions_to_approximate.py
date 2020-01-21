import os
import pdb  # noqa:F401
import sys

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root)

import numpy as np
import pytest

from src.functions_to_approximate import borehole_numba
from src.functions_to_approximate import borehole_readable
from src.functions_to_approximate import borehole_vectorize
from src.parameters import study_params
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal

# from numpy.testing import assert_equal


#########################################################################
# FIXTURES
#########################################################################


@pytest.fixture
def setup_borehole_on_domain():
    out = {}
    out["input"] = np.array(
        object=[
            [100000, 1000, 800, 1000, 0.1, 1500, 10000, 100],
            [100000, 1000, 800, 1000, 0.1, 1500, 10000, 100],
        ],
        dtype=float,
    )
    return out


@pytest.fixture
def setup_borehole_truncated_input():
    out = {}
    out["input"] = np.array(
        object=[[100000, 1000, 800, 1000, 0.1], [100000, 1000, 800, 1000, 0.1]],
        dtype=float,
    )
    return out


@pytest.fixture
def setup_borehole_large_set():

    grid_min = study_params["grid"]["lower bounds"]
    grid_max = study_params["grid"]["upper bounds"]

    np.random.seed(121)

    input = []
    for _ in range(10000):
        input_tmp = np.random.uniform(0.0, 1.0, len(grid_min))
        input.append(input_tmp)

    out = {}
    out["input"] = (
        input * grid_min + (np.ones((10000, len(grid_min))) - input) * grid_max
    )

    return out


#########################################################################
# TESTS
#########################################################################


def test_borehole_readable_on_domain(setup_borehole_on_domain):
    expected = np.array(
        object=[
            np.pi * 40000000 / (1001 * np.log(10000) + 3000000),
            np.pi * 40000000 / (1001 * np.log(10000) + 3000000),
        ],
        dtype=float,
    )
    actual = borehole_readable(**setup_borehole_on_domain)
    assert_array_equal(actual, expected)


def test_borehole_numba_on_domain(setup_borehole_on_domain):
    expected = np.array(
        object=[
            np.pi * 40000000 / (1001 * np.log(10000) + 3000000),
            np.pi * 40000000 / (1001 * np.log(10000) + 3000000),
        ],
        dtype=float,
    )
    actual = borehole_numba(**setup_borehole_on_domain)
    assert_array_equal(actual, expected)


def test_borehole_readable_truncated_input(setup_borehole_truncated_input):
    expected = np.array(
        object=[
            np.pi
            * 40000000
            / ((1 + (100000 / 89.55)) * np.log(10000) + (280000000 / 82.50)),
            np.pi
            * 40000000
            / ((1 + (100000 / 89.55)) * np.log(10000) + (280000000 / 82.50)),
        ],
        dtype=float,
    )
    actual = borehole_readable(**setup_borehole_truncated_input)
    assert_array_almost_equal(actual, expected)


def test_borehole_vectorize_truncated_input(setup_borehole_truncated_input):
    expected = np.array(
        object=[
            np.pi
            * 40000000
            / ((1 + (100000 / 89.55)) * np.log(10000) + (280000000 / 82.50)),
            np.pi
            * 40000000
            / ((1 + (100000 / 89.55)) * np.log(10000) + (280000000 / 82.50)),
        ],
        dtype=float,
    )
    actual = borehole_vectorize(**setup_borehole_truncated_input)
    assert_array_almost_equal(actual, expected)


def test_borehole_numba_truncated_input(setup_borehole_truncated_input):
    expected = np.array(
        object=[
            np.pi
            * 40000000
            / ((1 + (100000 / 89.55)) * np.log(10000) + (280000000 / 82.50)),
            np.pi
            * 40000000
            / ((1 + (100000 / 89.55)) * np.log(10000) + (280000000 / 82.50)),
        ],
        dtype=float,
    )
    actual = borehole_numba(**setup_borehole_truncated_input)
    assert_array_almost_equal(actual, expected, decimal=12)


def test_borehole_readable_equals_numba(setup_borehole_large_set):
    actual_readable = borehole_readable(**setup_borehole_large_set)
    actual_numba = borehole_numba(**setup_borehole_large_set)
    assert_array_almost_equal(actual_readable, actual_numba, decimal=12)


def test_borehole_numba_equals_vectorize(setup_borehole_large_set):
    actual_numba = borehole_numba(**setup_borehole_large_set)
    actual_vectorize = borehole_vectorize(**setup_borehole_large_set)
    assert_array_almost_equal(actual_numba, actual_vectorize, decimal=12)
