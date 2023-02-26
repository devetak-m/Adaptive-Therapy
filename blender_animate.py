# import bleneder modules
import bpy
from ABM_model import ABM_model
import numpy as np

# set up model
parameters = {

    "domain_size" : 40,
    "T" : 500,
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
    "initial_condition_type" : "cluster_3d",
    "save_locations" : True,
    "dimension" : 3,
    "seed" : 0}

model = ABM_model(parameters)
model.load_locations_from_file("data/locations_cluster_3d_0.txt")
location_data = model.location_data.copy()
# sort location_data by type, the last column is the type

