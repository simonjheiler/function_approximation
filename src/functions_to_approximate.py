"""Library of test functions to evaluate alternative multidimensional
    approximation methods.
"""
# import pdb
import numba as nb
import numpy as np

# import pandas as pd
# import respy as rp

#########################################################################
# FUNCTIONS
#########################################################################


def borehole_readable(input):
    """Calculate the flow through a borehole given *inputs*.

    Args:
        inputs (np.ndarray): inputs with entries

    Returns:
        output (float)

    """

    input_default = np.array(
        object=[89335.0, 1050.0, 760.0, 25050.0, 0.1, 1400.0, 8250.0, 89.55],
        dtype=float,
    )

    input = np.append(input, input_default[len(input) :])

    x1 = input[0]
    x2 = input[1]
    x3 = input[2]
    x4 = input[3]
    x5 = input[4]
    x6 = input[5]
    x7 = input[6]
    x8 = input[7]

    output = (
        (2 * np.pi * x1 * (x2 - x3))
        / np.log(x4 / x5)
        / (1 + ((2 * x6 * x1) / (np.log(x4 / x5) * x5 ** 2 * x7)) + (x1 / x8))
    )

    return output


def borehole_vectorize(input):
    """Calculate the flow through a borehole given *inputs*.

    Args:
    inputs (np.ndarray): inputs with entries

    Returns:
    output (float)

    """

    input_default = np.array(
        object=[89335.0, 1050.0, 760.0, 25050.0, 0.1, 1400.0, 8250.0, 89.55],
        dtype=float,
    )

    input = np.append(input, [input_default[input.shape[1] :]] * input.shape[0], axis=1)

    x1 = input[..., 0]
    x2 = input[..., 1]
    x3 = input[..., 2]
    x4 = input[..., 3]
    x5 = input[..., 4]
    x6 = input[..., 5]
    x7 = input[..., 6]
    x8 = input[..., 7]

    output = (
        (2 * np.pi * x1 * (x2 - x3))
        / np.log(x4 / x5)
        / (1 + ((2 * x6 * x1) / (np.log(x4 / x5) * x5 ** 2 * x7)) + (x1 / x8))
    )

    return output


def borehole_numba(input):

    input_default = np.array(
        object=[89335.0, 1050.0, 760.0, 25050.0, 0.1, 1400.0, 8250.0, 89.55],
        dtype=float,
    )

    input = np.append(input, input_default[len(input) :])

    output = borehole_jit(input)

    return output


@nb.jit(nopython=True)
def borehole_jit(input):
    """Calculate the flow through a borehole given *inputs*.

    Args:
        inputs (np.ndarray): inputs with entries

    Returns:
        output (float)

    """

    x1 = input[0]
    x2 = input[1]
    x3 = input[2]
    x4 = input[3]
    x5 = input[4]
    x6 = input[5]
    x7 = input[6]
    x8 = input[7]

    output = (
        (2 * np.pi * x1 * (x2 - x3))
        / np.log(x4 / x5)
        / (1 + ((2 * x6 * x1) / (np.log(x4 / x5) * x5 ** 2 * x7)) + (x1 / x8))
    )

    return output
