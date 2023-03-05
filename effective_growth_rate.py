from default_parameters import parameters
from ABM_model import ABM_model
from neighbour_counting import count_neighbours,calculate_stats,bar_chart
import numpy as np
import matplotlib.pyplot as plt

def get_grid(location_data_t):
    """Returns a grid of the locations of the cells."""
    # Create a grid of zeros
    grid = np.zeros((parameters["domain_size"],parameters["domain_size"]))
    # Loop over the locations
    for i in range(location_data_t.shape[0]):
        # Get the location
        location = location_data_t[i,:].astype(int)
        # Set the grid value to the cell type
        grid[tuple(location[:-1])] = location[-1]
    return grid

# Set up the model
parameters["fill_factor"] = 1
parameters["save_locations"] = True
model = ABM_model(parameters,True)
model.run("adaptive")
R_data = model.data[:,1]

# quadratic fit of R_data using np.polyfit
t = np.arange(0,parameters["T"],parameters["dt"])
quadratic = np.poly1d(np.polyfit(t, R_data, 2))
coeff = quadratic.c
growth_rate = lambda T: 2*coeff[0]*T + coeff[1]

# calculate RE for each time
location_data = model.location_data 
RE_data = np.zeros(len(location_data))
stats_data = np.zeros((len(location_data),6))
for k in range(len(location_data)):
    if k % 20 == 0:
        print("Counting neighbours at time {}...".format(k))
    grid = get_grid(location_data[k])
    results = count_neighbours(grid)
    stats = calculate_stats(results)
    stats_data[k,:] = stats
    RE_data[k] = stats[5]

# plot the data
fig,ax = plt.subplots()
ax.plot(t,R_data,label="R")
ax.plot(t,quadratic(t),label="quadratic fit")
ax.plot(t,growth_rate(t)*100,label="growth rate")
ax.plot(t,RE_data[1:]*200,label="RE")
ax.set_xlabel("Time")
ax.set_ylabel("R growth rate")
ax.legend()
plt.show()
