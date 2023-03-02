import os 
import numpy as np
from ABM_model import ABM_model
import matplotlib.pyplot as plt
domain_size = 200
default_parameters = {
        "domain_size": domain_size,
        "T": 2000,
        "dt": 1,    
        "S0": 8000,
        "R0": 800,
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
        "core_locations": np.array([[domain_size//4,domain_size//4],[3*domain_size//4,3*domain_size//4]]),
        "seed": 0,
        "foldername": None,
        "save_frequency": 2000,
    }

# initial_conditions = [ic for ic in os.listdir("initial_conditions") if "0.2" in ic]
# initial_conditions = ["resistant_core_0.2_0.1.png","resistant_rim_0.2_0.1.png"]
# initial_conditions = ["resistant_rim_0.2_0.1.png"]
initial_conditions = ["multiple_resistant_cores","multiple_resistant_rims"]
for ic in initial_conditions:
    # calculate time to progression and mean field
    foldername = f"data/{ic}_trajectory"
    parameters = default_parameters.copy()
    parameters["initial_condition_type"] = ic
    parameters["foldername"] = foldername
    model = ABM_model(parameters,verbose=True)   
    print(f"Running model for {ic[:-4]}")
    model.run(parameters["therapy"])
    fig,ax = plt.subplots()
    model.plot_celltypes_density(ax)
    plt.show()
    fig,ax = plt.subplots()
    fig,ax,anim = model.animate_cells([fig,ax],stride=10)
    anim.save(f"{foldername}/cells.mp4",fps=30)

# path = "data/resistant_core_0.2_0.01_trajectory"
# parameters = default_parameters.copy()
# parameters["initial_condition_type"] = "initial_conditions/resistant_core_0.2_0.01.png"
# model = ABM_model(parameters,verbose=True)
# model.load_locations(path)

# fig,ax = plt.subplots()
# model.plot_celltypes_density(ax)
# plt.show(
# fig,ax = plt.subplots()
# fig,ax,anim = model.animate_cells([fig,ax],stride=1)
# anim.save(f"{path}/cells.mp4",fps=10)