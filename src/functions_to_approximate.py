"""Library of test functions to evaluate alternative multidimensional
    approximation methods.
"""
import pdb  # noqa:F401

import numba as nb
import numpy as np


#########################################################################
# FUNCTIONS
#########################################################################


# BOREHOLE FUNCTION


def borehole_readable(input):
    """Calculate the flow through a borehole given *inputs*.

    Args:
        inputs (np.array): inputs with entries

    Returns:
        output (np.array)


    """
    input_default = [89335.0, 1050.0, 760.0, 25050.0, 0.1, 1400.0, 8250.0, 89.55]

    output = []
    for idx in range(input.shape[0]):

        input_tmp = input[idx, ...]
        input_tmp = np.append(input_tmp, input_default[len(input_tmp) :])

        x1 = input_tmp[0]
        x2 = input_tmp[1]
        x3 = input_tmp[2]
        x4 = input_tmp[3]
        x5 = input_tmp[4]
        x6 = input_tmp[5]
        x7 = input_tmp[6]
        x8 = input_tmp[7]

        output_tmp = (
            (2 * np.pi * x1 * (x2 - x3))
            / np.log(x4 / x5)
            / (1 + ((2 * x6 * x1) / (np.log(x4 / x5) * x5 ** 2 * x7)) + (x1 / x8))
        )
        output.append(output_tmp)

    output = np.array(object=output)

    return output


# @nb.njit
def borehole_wrapper_numba_vectorize(input):
    """Calculate the flow through a borehole given *inputs*.

    Args:
        inputs (np.ndarray): inputs with entries

    Returns:
        output (float)

    """
    input_default = np.array(
        object=[89335.0, 1050.0, 760.0, 25050.0, 0.1, 1400.0, 8250.0, 89.55]
    )

    points = np.full((input.shape[0], len(input_default)), np.nan)
    if input.shape[1] == len(input_default):
        points = input
    else:
        input_fill = np.full(
            (input.shape[0], len(input_default[input.shape[1] :])), np.nan
        )
        for col in range(len(input_default[input.shape[1] :])):
            input_fill[:, col] = np.repeat(
                input_default[input.shape[1] + col], input.shape[0]
            )
        points[:, : input.shape[1]] = input
        points[:, input.shape[1] :] = input_fill

    output = borehole_step_numba_vectorize(points)

    return output


@nb.njit
def borehole_step_numba_vectorize(input):
    """Calculate the flow through a borehole given *inputs*.

    Args:
    inputs (np.ndarray): inputs with entries

    Returns:
    output (float)

    """
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


# @nb.njit
def borehole_wrapper_numba_iter(input):
    """Calculate the flow through a borehole given *inputs*.

    Args:
        inputs (np.ndarray): inputs with entries

    Returns:
        output (float)

    """
    input_default = np.array(
        object=[89335.0, 1050.0, 760.0, 25050.0, 0.1, 1400.0, 8250.0, 89.55],
    )

    points = np.full((input.shape[0], len(input_default)), np.nan)
    if input.shape[1] == len(input_default):
        points = input
    else:
        input_fill = np.full(
            (input.shape[0], len(input_default[input.shape[1] :])), np.nan
        )
        for col in range(len(input_default[input.shape[1] :])):
            input_fill[:, col] = np.repeat(
                input_default[input.shape[1] + col], input.shape[0]
            )
        points[:, : input.shape[1]] = input
        points[:, input.shape[1] :] = input_fill

    output = np.full(input.shape[0], np.nan)
    for idx in range(points.shape[0]):
        point = points[idx, :]
        output[idx] = borehole_step_numba_iter(point)

    return output


@nb.njit
def borehole_step_numba_iter(input):
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


# ZHOU (1998) FUNCTION


@nb.njit
def zhou_phi(input):

    d = len(input)
    phi = (2 * np.pi) ** (-d / 2) * np.exp(-0.5 * np.linalg.norm(input) ** 2)

    return phi


def zhou_readable(input):

    d = input.shape[1]

    output = []
    for idx in range(input.shape[0]):
        x = input[idx, :]
        output_tmp = (
            10 ** d
            / 2
            * (zhou_phi(10 * (x - [1 / 3] * d)) + zhou_phi(10 * (x - [2 / 3] * d)))
        )
        output.append(output_tmp)

    output = np.array(object=output)

    return output


# @nb.njit
def zhou_phi_vectorize(input):

    d = input.shape[1]
    phi = (2 * np.pi) ** (-d / 2) * np.exp(-0.5 * np.linalg.norm(input, axis=1) ** 2)

    return phi


# @nb.njit
def zhou_phi_vectorize_jit(input):

    d = input.shape[1]
    phi = (2 * np.pi) ** (-d / 2) * np.exp(-0.5 * np.linalg.norm(input, axis=1) ** 2)

    return phi


def zhou_vectorize(input):

    d = input.shape[1]

    output = (
        10 ** d
        / 2
        * (
            zhou_phi_vectorize(10 * (input - [1 / 3] * d))
            + zhou_phi_vectorize(10 * (input - [2 / 3] * d))
        )
    )

    # pdb.set_trace()
    output = np.array(object=output)

    return output


# @nb.njit
def zhou_numba(input):

    d = input.shape[1]

    output = (
        10 ** d
        / 2
        * (
            zhou_phi_vectorize(10 * (input - [1 / 3] * d))
            + zhou_phi_vectorize(10 * (input - [2 / 3] * d))
        )
    )

    output = np.array(object=output)

    return output
