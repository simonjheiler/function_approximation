"""Library of test functions to evaluate alternative multidimensional
    approximation methods.
"""
# import pdb
import numpy as np

# import pandas as pd

# import respy as rp


def borehole(input):
    """Calculate the flow through a borehole given *inputs*.

    Args:
        inputs (np.ndarray): inputs with entries

    Returns:
        output (float)

    """

    output = (
        (2 * np.pi * input[0] * (input[1] - input[2]))
        / np.log(input[3] / input[4])
        / (
            1
            + (
                (2 * input[5] * input[0])
                / (np.log(input[3] / input[4]) * input[4] ** 2 * input[6])
            )
            + (input[0] / input[7])
        )
    )

    return output
