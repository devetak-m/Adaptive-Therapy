import os 
import numpy as np
from ABM_model import ABM_model
import matplotlib.pyplot as plt
default_parameters = {
        "domain_size": 1000,
        "T": 10000,
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
        "save_locations": True,
        "dimension": 2,
        "seed": 4,
        "foldername": None,
        "save_frequency": 1000,}



path = "data/resistant_rim_0.2_0.1_trajectory"
parameters = default_parameters.copy()
parameters["initial_condition_type"] = "initial_conditions/resistant_core_0.2_0.01.png"
model = ABM_model(parameters,verbose=True)
model.load_locations(path)

fig,ax = plt.subplots()
model.plot_celltypes_density(ax)
plt.show()
fig,ax = plt.subplots()
fig,ax,anim = model.animate_cells([fig,ax],stride=1)
plt.show()
# anim.save(f"data/check_up/cells.mp4",fps=10)