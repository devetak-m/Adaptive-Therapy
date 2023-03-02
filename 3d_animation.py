from ABM_model import *

# set up parameters
domain_size = 40
parameters = {
    "domain_size": domain_size,
    "T": 400,
    "dt": 2,    
    "S0": 20000,
    "R0": 1000,
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
    "initial_condition_type": "cluster_3d",
    "fill_factor": 0.8,
    "core_locations": np.array([[domain_size//3,domain_size//3],[2*domain_size//3,2*domain_size//3],[1*domain_size//3,2*domain_size//3],[2*domain_size//3,1*domain_size//3]]),
    "save_locations": True,
    "dimension": 3,
    "seed": 4,
    "foldername": "data/new_adaptive_data3",
    "save_frequency": 100,
}

    # set up model
model = ABM_model(parameters,True)
model.run(parameters["therapy"])
print("Model run complete.")
fig1,ax1 = plt.subplots()
model.plot_celltypes_density(ax1)
plt.show()
fig,ax,anim = model.animate_cells_graph(stride=1)
anim.save("media/large_adaptive2.mp4")