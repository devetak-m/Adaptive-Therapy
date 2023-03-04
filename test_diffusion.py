from ABM_model import *
import numpy as np
import matplotlib.pyplot as plt

# set up parameters
domain_size = 100
parameters = {
    "domain_size": domain_size,
    "T": 500,
    "dt": 1,    
    "S0": 2000,
    "R0": 200,
    "N0": 0,
    "grS": 0.023,
    "grR": 0.023,
    "grN": 0.00,
    "drS": 0.01,
    "drR": 0.01,
    "drN": 0.0,
    "divrS": 0.75,
    "divrN": 0.5,
    "therapy": "adaptive",
    "initial_condition_type": "resistant_core",
    "fill_factor": 0.8,
    "core_locations": np.array([[domain_size//3,domain_size//3],[2*domain_size//3,2*domain_size//3],[1*domain_size//3,2*domain_size//3],[2*domain_size//3,1*domain_size//3]]),
    "save_locations": True,
    "dimension": 2,
    "diffusion_rate": 0.1,
    "seed": 0,
    "foldername": "data/diffusion_test",
    "save_frequency": 100,
}
diffusion_rates = [0.0]
for diffusion_rate in diffusion_rates:
    parameters["diffusion_rate"] = diffusion_rate
    parameters["foldername"] = f"data/diffusion_test_{diffusion_rate}_dynamics"
    model = ABM_model(parameters,True)
    model.run(parameters["therapy"])
    fig,ax,anim = model.animate_cells_graph(stride=2)
    anim.save(f"media/diffusion_test{diffusion_rate}_dynamics.mp4")