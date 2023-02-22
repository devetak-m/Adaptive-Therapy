from ABM_model import ABM_model
import numpy as np

# definine initial grid
def initial_grid(model, initial_occupancy):

    S0 = model.S0
    R0 = model.R0

    # square lenght
    L = int(np.sqrt(S0 + R0)/np.sqrt(initial_occupancy)) + 1

    # create grid
    grid = np.zeros((model.domain_size, model.domain_size))

    # select S0 + R0 random locations in the center of the grid
    # of size L x L without replacement
    locations = np.random.choice(L**2, S0 + R0, replace=False)
    locations = np.unravel_index(locations, (L, L))
    locations = np.array(locations)
    locations[0] += int((model.domain_size - L)/2)
    locations[1] += int((model.domain_size - L)/2)

    # shuffle the locations
    np.random.shuffle(locations.T)
    # assign the first S0 locations to S
    grid[locations[0, :S0], locations[1, :S0]] = 1
    # assign the next R0 locations to R
    grid[locations[0, S0:], locations[1, S0:]] = 2

    model.grid = grid.copy()
    model.initial_grid = grid.copy()
    return






if __name__ == "__main__":

    # set up parameters
    parameters = {"domain_size" : 40,
    "T" : 400,
    "dt" : 1,
    "S0" : 30,
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
    "dimension" : 2,
    "seed" : 0}

    # set up model
    model = ABM_model(parameters)
    # set up initial condition
    model.grid = np.zeros((model.domain_size, model.domain_size))
    initial_grid(model, 0.9)
    model.initial_grid = model.grid.copy()
    # show grid of initial conditions
    model.plot_grid()