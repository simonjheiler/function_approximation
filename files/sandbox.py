# import pdb
import numpy as np
import pandas as pd
from functions_to_approximate import borehole

n_state_variables = 8
n_gridpoints = 5
grid_min = [1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0]
grid_max = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]


grids = {
    0: np.linspace(grid_min[0], grid_max[0], n_gridpoints),
    1: np.linspace(grid_min[1], grid_max[1], n_gridpoints),
    2: np.linspace(grid_min[2], grid_max[2], n_gridpoints),
    3: np.linspace(grid_min[3], grid_max[3], n_gridpoints),
    4: np.linspace(grid_min[4], grid_max[4], n_gridpoints),
    5: np.linspace(grid_min[5], grid_max[5], n_gridpoints),
    6: np.linspace(grid_min[6], grid_max[6], n_gridpoints),
    7: np.linspace(grid_min[7], grid_max[7], n_gridpoints),
}


def get_dims_state_grid(
    n_state_variables, n_gridpoints,
):
    tmp = np.zeros(n_state_variables, dtype=np.int)
    for idx in range(n_state_variables):
        tmp[idx] = n_gridpoints

    dims_state_grid = np.array(object=list(tmp))

    return dims_state_grid


def get_states_grid_dense(dims_state_grid):

    state_grid = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [
                pd.np.arange(dims_state_grid[0]),
                pd.np.arange(dims_state_grid[1]),
                pd.np.arange(dims_state_grid[2]),
                pd.np.arange(dims_state_grid[3]),
                pd.np.arange(dims_state_grid[4]),
                pd.np.arange(dims_state_grid[5]),
                pd.np.arange(dims_state_grid[6]),
                pd.np.arange(dims_state_grid[7]),
            ],
            names=range(dims_state_grid.size),
        )
    ).reset_index()

    return state_grid


def state_from_id(idx, dims_state_grid):
    """Convert *idx* to state series structured according to *dims_state_grid*"""

    entries = [idx] * len(dims_state_grid)
    divisors = []
    for i in range(1, len(dims_state_grid)):
        value = 1
        for j in range(i, len(dims_state_grid)):
            value *= dims_state_grid[j]
        divisors.append(value)
        for k in range(len(dims_state_grid)):
            if k == i - 1:
                entries[k] //= value
            else:
                entries[k] %= value
        out = np.array(object=entries)

    return out


dims_state_grid = get_dims_state_grid(n_state_variables, n_gridpoints)
state_grid = get_states_grid_dense(dims_state_grid)
state_index = state_grid.index

state_test = state_from_id(5, dims_state_grid)

inputs_test = np.array(
    [
        grids[0][state_test[0]],
        grids[1][state_test[1]],
        grids[2][state_test[2]],
        grids[3][state_test[3]],
        grids[4][state_test[4]],
        grids[5][state_test[5]],
        grids[6][state_test[6]],
        grids[7][state_test[7]],
    ]
)

output_test = borehole(inputs_test)

print(output_test)
