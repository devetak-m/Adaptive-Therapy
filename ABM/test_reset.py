from ABM_model import ABM_model
import matplotlib.pyplot as plt
import numpy as np
parameters = {"domain_size" : 40,
    "T" : 400,
    "dt" : 1,
    "S0" : 200,
    "R0" : 10,
    "N0" : 0,
    "grS" : 0.023,
    "grR" : 0.023,
    "grN" : 0.005,
    "drS" : 0.01,
    "drR" : 0.01,
    "drN" : 0.00,
    "divrS" : 0.75,
    "divrN" : 0.5,
    "therapy" : "adaptive",
    "initial_condition_type" : "uniform",
    "save_locations" : False,
    "dimension" : 0,
    "seed" : 100}
# seed 100 has one peak, 1000 has two peaks
model = ABM_model(parameters)
model.run(parameters["therapy"])

# Using reset() to reset the model
fig,axs = plt.subplots(1,2)
fig.set_size_inches(10,5)
model.plot_celltypes_density(axs[0])
axs[0].set(xlabel="x", ylabel="y", title="Before reset")
model.set_seed(1000)
model.reset(hard=True)
model.run(parameters["therapy"])
model.plot_celltypes_density(axs[1])
axs[1].set(xlabel="x", ylabel="y", title="After reset")
plt.show()

# Using two instances of model
model1 = ABM_model(parameters)
model1.run(parameters["therapy"])
parameters2 = parameters.copy()
parameters2["seed"] = 1000
model2 = ABM_model(parameters2)
model2.run(parameters["therapy"])
fig,axs = plt.subplots(1,2)
fig.set_size_inches(10,5)
model1.plot_celltypes_density(axs[0])
axs[0].set(xlabel="x", ylabel="y", title="Seed 100")
model2.plot_celltypes_density(axs[1])
axs[1].set(xlabel="x", ylabel="y", title="Seed 1000")
plt.show()
# These two are equivalent. Not true for soft reset.

# Comparing soft reset with 
fig,axs = plt.subplots(1,2) 
fig.set_size_inches(10,5)
# soft reset
model3 = ABM_model(parameters)
model3.set_seed(1000)
model3.run(parameters["therapy"])
model3.plot_celltypes_density(axs[0])
axs[0].set(xlabel="x", ylabel="y", title="Initial seed 100, set_seed 1000")

# running once then soft reset
model4 = ABM_model(parameters)
model4.run(parameters["therapy"])
model4.reset(hard=False)
model4.set_seed(1000)
model4.run(parameters["therapy"])
model4.plot_celltypes_density(axs[1])
axs[1].set(xlabel="x", ylabel="y", title="Initial seed 100, set_seed 1000")
plt.show()


