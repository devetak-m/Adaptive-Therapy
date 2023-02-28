from ABM_model import ABM_model
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
domain_size = 400
parameters_ABM = {
    "domain_size" : domain_size,
    "T" : 2,
    "dt" : 1,
    "S0" : 8000,
    "R0" : 800,
    "N0" : 0,
    "grS" : 0.023,
    "grR" : 0.023,
    "grN" : 0.005,
    "drS" : 0.01,
    "drR" : 0.01,
    "drN" : 0.00,
    "divrS" : 0.75,
    "divrN" : 0.5,
    "therapy" : "continuous",
    "initial_condition_type" : "uniform",
    "fill_factor":1,
    "core_locations": np.array([[domain_size//4,domain_size//4],[3*domain_size//4,3*domain_size//4]]),
    "save_locations" : False,
    "dimension" : 2,
    "seed" : 1}
new_size = 20
def save_initial_conditions(parameters_ABM,filename= None, show = True):
    S0 = parameters_ABM["S0"]
    R0 = parameters_ABM["R0"]
    total = S0+R0
    model = ABM_model(parameters_ABM)
    grid = model.grid
    sensitive_grid = np.zeros(grid.shape)
    sensitive_grid[grid==1] = 255
    resistant_grid = np.zeros(grid.shape)
    resistant_grid[grid==2] = 255
    sensitive_image = Image.fromarray(sensitive_grid)
    sensitive_resized = sensitive_image.resize((new_size,new_size))
    sensitive_array = np.array(sensitive_resized)/255
    sensitive_array *= (S0/(total))/np.sum(sensitive_array)
    resistant_image = Image.fromarray(resistant_grid)
    resistant_resized = resistant_image.resize((new_size,new_size))
    resistant_array = np.array(resistant_resized)/255
    # BUG for some reason negative. Just rounding up for now
    resistant_array[resistant_array<0] = 0
    sensitive_array[sensitive_array<0] = 0
    
    resistant_array *= (R0/(total))/np.sum(resistant_array)
    np.sum(sensitive_array+resistant_array)
    print(np.sum(sensitive_array)/np.sum(resistant_array))
    np.save(f"{filename}_sensitive.npy",sensitive_array)
    np.save(f"{filename}_resistant.npy",resistant_array)
    if show:
        fig,ax = plt.subplots(1,2)
        ax[0].axis("off")
        ax[1].axis("off")
        ax[0].imshow(sensitive_array)
        ax[1].imshow(resistant_array)
        plt.show()

initial_condition_types  = ["resistant_core","resistant_rim","multiple_resistant_cores","multiple_resistant_rims"]
for initial_condition_type in initial_condition_types:
    parameters_ABM["initial_condition_type"] = initial_condition_type
    filename = "pde_initial_conditions/"+initial_condition_type
    save_initial_conditions(parameters_ABM,filename)

for fill_factor in [0.8,0.9,1]:
    parameters_ABM["fill_factor"] = fill_factor
    parameters_ABM["initial_condition_type"] = "uniform_ball"
    filename = f"pde_initial_conditions/uniform_ball_{fill_factor}"
    save_initial_conditions(parameters_ABM,filename)



parameters_ABM["initial_condition_type"] = "uniform"
parameters_ABM["domain_size"] = 200
filename = "pde_initial_conditions/uniform"
save_initial_conditions(parameters_ABM,filename)
