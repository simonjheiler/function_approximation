import copy  # noqa:F401
import json  # noqa:F401
import os  # noqa:F401
import pdb  # noqa:F401
import pickle  # noqa:F401
import sys  # noqa:F401

root = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.insert(0, root)

import matplotlib.pyplot as plt  # noqa:F401
from mpl_toolkits.mplot3d import Axes3D  # noqa:F401
import numpy as np  # noqa:F401
import src.functions_to_approximate as functions  # noqa:F401

from src.execute import execute_study  # noqa:F401
from src.execute import plot_results  # noqa:F401
from src.execute import compare_results  # noqa:F401
from src.execute import compare_fit_2d_iter  # noqa:F401

#########################################################################
# PARAMETERS
#########################################################################

# load default interpolation parameters
with open("./src/interp_params.json") as json_file:
    interpolation_params = json.load(json_file)

# load default study parameters
with open("./src/study_params.json") as json_file:
    study_params = json.load(json_file)


#########################################################################
# SCRIPT
#########################################################################

if __name__ == "__main__":

    params_default = copy.deepcopy(study_params)
    interp_params = copy.deepcopy(interpolation_params)

    # create instance for study parameters
    params_sparse_ch = copy.deepcopy(params_default)
    params_sparse_ch["controls"]["interpolation method"] = "sparse"
    params_sparse_ch["controls"]["grid_method"] = "sparse"
    params_sparse_ch["controls"]["evaluate off-grid"] = "True"
    params_sparse_ch["controls"]["iterations"] = 3
    params_sparse_ch["controls"]["grid density"] = "medium"

    # this implementation only works for dim <= 2
    params_sparse_ch["controls"]["dims"] = [2]

    # create instance for interpolation parameters
    interp_params_sparse_ch = copy.deepcopy(interp_params)
    interp_params_sparse_ch["sparse"]["polynomial family"] = "CH"

    # create instance for zhou function
    params_zhou_sparse_ch = copy.deepcopy(params_sparse_ch)
    params_zhou_sparse_ch["controls"]["function to approximate"] = "zhou_vectorize"

    # create instance for borehole function
    params_borehole_sparse_ch = copy.deepcopy(params_sparse_ch)
    params_borehole_sparse_ch["controls"][
        "function to approximate"
    ] = "borehole_wrapper_vectorize"

    # results_zhou_sparse_ch = execute_study(params_zhou_sparse_ch, interp_params_sparse_ch)
    results_borehole_sparse_ch = execute_study(
        params_borehole_sparse_ch, interp_params_sparse_ch
    )

    # print(results_zhou_sparse_ch["gridpoints"])
    print(results_borehole_sparse_ch["gridpoints"])
