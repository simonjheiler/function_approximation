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

    # create instance for study parameters
    params = copy.deepcopy(study_params)
    params["controls"]["interpolation method"] = "sparse"
    params["controls"]["grid_method"] = "sparse"
    params["controls"]["evaluate off-grid"] = "True"
    params["controls"]["iterations"] = 3
    params["controls"]["grid density"] = "medium"

    # this implementation only works for dim <= 2
    params["controls"]["dims"] = [2]

    # create instance for interpolation parameters
    interp_params = copy.deepcopy(interpolation_params)
    interp_params["sparse"]["polynomial family"] = "CH"

    # create instance for zhou function
    params_zhou = copy.deepcopy(params)
    params_zhou["controls"]["function to approximate"] = "zhou_vectorize"

    # create instance for borehole function
    params_borehole = copy.deepcopy(params)
    params_borehole["controls"][
        "function to approximate"
    ] = "borehole_wrapper_vectorize"

    results_zhou = execute_study(params_zhou, interp_params)
    # results_borehole = execute_study(params_borehole, interp_params)

    print(results_zhou)
