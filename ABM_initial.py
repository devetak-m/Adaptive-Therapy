import numpy as np
import matplotlib.pyplot as plt
from ABM_model import ABM_model
from default_parameters import *

# load model and save initial conditons
test_names = ["resistant_core","resistant_rim","multiple_resistant_cores","multiple_resistant_rims","uniform_ball_0.8","uniform_ball_0.9","uniform_ball_1","uniform"]
parameters["domain_size"] = 500
parameters["S0"] = 50000
parameters["R0"] = 5000
parameters["core_locations"] = np.array([[domain_size//4,domain_size//4],[domain_size//4*3,domain_size//4*3]])
for test_name in test_names:
    print("Saving initial condition for {}".format(test_name))
    if "uniform_ball" in test_name:
        initial_condition_type = "uniform_ball"
        parameters["fill_factor"] = float(test_name.split("_")[-1])
    else:
        parameters["fill_factor"] = 1
        initial_condition_type = test_name
    parameters["initial_condition_type"] = initial_condition_type
    model = ABM_model(parameters=parameters)
    grid = model.grid
    plt.imsave("shared_media/initial_conditions/{}.png".format(test_name),grid, cmap=model.get_cmap(),vmin=0,vmax=2)
