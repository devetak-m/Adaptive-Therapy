from time_to_progression import time_to_progression,ttp_ode
import os 
import numpy as np
from ABM_model import ABM_model
default_parameters = {
        "domain_size": 100,
        "T": 10,
        "dt": 1,    
        "S0": 0,
        "R0": 0,
        "N0": 0,
        "grS": 0.023,
        "grR": 0.023,
        "grN": 0.0,
        "drS": 0.013,
        "drR": 0.013,
        "drN": 0.0,
        "divrS": 0.75,
        "divrN": 0.5,
        "therapy": "adaptive",
        "initial_condition_type": None,
        "save_locations": False,
        "dimension": 2,
        "seed": 4,
        "foldername": None,
        "save_frequency": 100,
    }

initial_conditions = [ic for ic in os.listdir("initial_conditions") if "0.2" in ic]
for ic in initial_conditions:
    # calculate time to progression and mean field
    filename = "data/"+ic[:-4]
    os.mkdir(filename)
    parameters = default_parameters.copy()
    parameters["initial_condition_type"] = "initial_conditions/"+ic
    # parameters["foldername"] = "data/"+ic[:-4]
    time_to_progression(parameters,nruns=10,threshold=2,filename=filename)
# ttp_ode(parameters,threshold=2)