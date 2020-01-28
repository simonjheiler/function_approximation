import os
import pdb  # noqa:F401
import sys

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root)

import numpy as np
import pytest

from src.functions_to_approximate import borehole_wrapper_numba_iter
from src.functions_to_approximate import borehole_wrapper_numba_vectorize
from src.functions_to_approximate import borehole_readable

# from src.functions_to_approximate import borehole_step_numba_iter
# from src.functions_to_approximate import borehole_step_numba_vectorize
from src.functions_to_approximate import zhou_phi
from src.functions_to_approximate import zhou_phi_vectorize
from src.functions_to_approximate import zhou_readable
from src.functions_to_approximate import zhou_vectorize

from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal


#########################################################################
# FIXTURES
#########################################################################


# BOREHOLE FUNCTION


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

    grid_min = np.array(
        object=[63070.0, 990.0, 700.0, 100.0, 0.05, 1120.0, 1500.0, 63.1], dtype=float,
    )
    grid_max = np.array(
        object=[115600.0, 1110.0, 820.0, 50000.0, 0.15, 1680.0, 15000.0, 116.0],
        dtype=float,
    )

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


# ZHOU (1998) FUNCTION


@pytest.fixture
def setup_zhou_on_domain():
    out = {}
    out["input"] = np.array(
        object=[
            [1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3],
            [2 / 3, 2 / 3, 2 / 3, 2 / 3, 2 / 3, 2 / 3, 2 / 3, 2 / 3],
        ],
        dtype=float,
    )
    return out


@pytest.fixture
def setup_zhou_phi_on_domain():
    out = {}
    out["input"] = np.array(
        object=[0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25], dtype=float,
    )
    return out


@pytest.fixture
def setup_zhou_large_set():

    grid_min = np.array(object=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float,)
    grid_max = np.array(object=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float,)

    np.random.seed(123)
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


# BOREHOLE FUNCTION


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


def test_borehole_numba_iter_on_domain(setup_borehole_on_domain):
    expected = np.array(
        object=[
            np.pi * 40000000 / (1001 * np.log(10000) + 3000000),
            np.pi * 40000000 / (1001 * np.log(10000) + 3000000),
        ],
        dtype=float,
    )
    actual = borehole_wrapper_numba_iter(**setup_borehole_on_domain)
    assert_array_equal(actual, expected)


def test_borehole_numba_vectorize_on_domain(setup_borehole_on_domain):
    expected = np.array(
        object=[
            np.pi * 40000000 / (1001 * np.log(10000) + 3000000),
            np.pi * 40000000 / (1001 * np.log(10000) + 3000000),
        ],
        dtype=float,
    )
    actual = borehole_wrapper_numba_vectorize(**setup_borehole_on_domain)
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


def test_borehole_numba_iter_truncated_input(setup_borehole_truncated_input):
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
    actual = borehole_wrapper_numba_iter(**setup_borehole_truncated_input)
    assert_array_almost_equal(actual, expected)


def test_borehole_numba_vectorize_truncated_input(setup_borehole_truncated_input):
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
    actual = borehole_wrapper_numba_vectorize(**setup_borehole_truncated_input)
    assert_array_almost_equal(actual, expected, decimal=12)


def test_borehole_readable_equals_numba_iter(setup_borehole_large_set):
    actual_readable = borehole_readable(**setup_borehole_large_set)
    actual_numba = borehole_wrapper_numba_iter(**setup_borehole_large_set)
    assert_array_almost_equal(actual_readable, actual_numba, decimal=12)


def test_borehole_numba_iter_equals_numba_vectorize(setup_borehole_large_set):
    actual_numba = borehole_wrapper_numba_iter(**setup_borehole_large_set)
    actual_vectorize = borehole_wrapper_numba_vectorize(**setup_borehole_large_set)
    assert_array_almost_equal(actual_numba, actual_vectorize, decimal=12)


# ZHOU (1998) FUNCTION


def test_zhou_phi_on_domain(setup_zhou_phi_on_domain):
    expected = (2 * np.pi) ** (-4) * np.exp(-0.25)
    actual = zhou_phi(**setup_zhou_phi_on_domain)
    assert actual == expected


def test_zhou_readable_on_domain(setup_zhou_on_domain):
    expected = np.array(
        object=[
            0.5
            * 10 ** 8
            * (2 * np.pi) ** (-4)
            * (1 + np.exp(-0.5 * 8 * (10 / 3) ** 2)),
            0.5
            * 10 ** 8
            * (2 * np.pi) ** (-4)
            * (np.exp(-0.5 * 8 * (10 / 3) ** 2) + 1),
        ],
        dtype=float,
    )
    actual = zhou_readable(**setup_zhou_on_domain)
    assert_array_equal(actual, expected)


def test_zhou_phi_vectorize(setup_zhou_on_domain):
    expected = np.array(
        object=[
            (2 * np.pi) ** (-4) * np.exp(-0.5 * 8 / 9),
            (2 * np.pi) ** (-4) * np.exp(-0.5 * 32 / 9),
        ],
        dtype=float,
    )
    actual = zhou_phi_vectorize(**setup_zhou_on_domain)
    assert_array_equal(actual, expected)


def test_zhou_vectorize(setup_zhou_on_domain):
    expected = np.array(
        object=[
            0.5
            * 10 ** 8
            * (2 * np.pi) ** (-4)
            * (1 + np.exp(-0.5 * 8 * (10 / 3) ** 2)),
            0.5
            * 10 ** 8
            * (2 * np.pi) ** (-4)
            * (np.exp(-0.5 * 8 * (10 / 3) ** 2) + 1),
        ],
        dtype=float,
    )
    actual = zhou_vectorize(**setup_zhou_on_domain)
    assert_array_equal(actual, expected)


def test_zhou_readable_equals_vectorize(setup_zhou_large_set):
    actual_readable = zhou_readable(**setup_zhou_large_set)
    actual_vectorize = zhou_vectorize(**setup_zhou_large_set)
    assert_array_almost_equal(actual_readable, actual_vectorize, decimal=12)
