import json
import os
import sys

root = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.insert(0, root)

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

    params = study_params

    # for iteration in [0, 1, 2, 3, 4, 5]:
    #
    #     interp_points = study_params["linear"]["interpolation points"][iteration]
    #     print(f"iteration: {iteration}")
    #     compare_fit_2d_iter(params, interpolation_params, iteration)

    results = execute_study
