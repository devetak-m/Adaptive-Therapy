from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
from ABM_model import *

def initial_grid_square(model, initial_occupancy):

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

parameters_ABM_1 = {

    "domain_size" : 40,
    "T" : 600,
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
    "initial_condition_type" : "cluster",
    "save_locations" : True,
    "dimension" : 2,
    "seed" : 0}


    # set up model

parameters_ABM_2 = {

    "domain_size" : 40,
    "T" : 600,
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
    "therapy" : "continuous",
    "initial_condition_type" : "cluster",
    "save_locations" : True,
    "dimension" : 2,
    "seed" : 0}

parameters_ABM_3 = {

    "domain_size" : 70,
    "T" : 600,
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
    "initial_condition_type" : "cluster",
    "save_locations" : True,
    "dimension" : 2,
    "seed" : 0}
   


model1 = ABM_model(parameters_ABM_1)
model2 = ABM_model(parameters_ABM_2)
model3 = ABM_model(parameters_ABM_3)

# set up initial condition
model1.set_initial_condition(parameters_ABM_1["initial_condition_type"])
model2.set_initial_condition(parameters_ABM_2["initial_condition_type"])
model3.set_initial_condition(parameters_ABM_3["initial_condition_type"])

initial_grid_square(model1, 0.85)
initial_grid_square(model2, 0.85)
initial_grid_square(model3, 0.7)

# run simulation
model1.run(parameters_ABM_1["therapy"])
model2.run(parameters_ABM_2["therapy"])
model3.run(parameters_ABM_3["therapy"])

# plot data on separate plots
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0] = model1.plot_celltypes_density(ax[0])
ax[1] = model2.plot_celltypes_density(ax[1])
ax[2] = model3.plot_celltypes_density(ax[2])
plt.show()

# animate cells and graph for each model
fig, ax, anim = model1.animate_cells_graph(stride=10,interval=80)
anim.save("media/adaptive_1.mp4")

fig, ax, anim = model2.animate_cells_graph(stride=10,interval=80)
anim.save("media/continuous_1.mp4")

fig, ax, anim = model3.animate_cells_graph(stride=10,interval=80)
anim.save("media/adaptive_2.mp4")

fig, ax = plt.subplots()
fig, ax, anim = model1.animate_cells((fig, ax))
anim.save("media/adaptive_1.mp4")

fig, ax = plt.subplots()
fig, ax, anim = model2.animate_cells((fig, ax))
anim.save("media/continuous_1.mp4")

fig, ax = plt.subplots()
fig, ax, anim = model3.animate_cells((fig, ax))
anim.save("media/adaptive_2.mp4")

fig, ax = plt.subplots()
fig, ax, anim = model1.animate_graph((fig, ax))
anim.save("media/adaptive_1.mp4")

fig, ax = plt.subplots()
fig, ax, anim = model2.animate_graph((fig, ax))
anim.save("media/continuous_1.mp4")

fig, ax = plt.subplots()
fig, ax, anim = model3.animate_graph((fig, ax))
anim.save("media/adaptive_2.mp4")



# fig, ax, anim = model.animate_cells_graph(stride=10,interval=80)
# anim.save("media/nice_abm.mp4")

#     fig,ax = plt.subplots()
#     fig,ax,anim = model.animate_cells((fig,ax))
#     anim.save("test_ABM.mp4")

#     fig,ax = plt.subplots()
#     fig,ax,anim = model.animate_graph((fig,ax))
#     anim.save("test_ABM_graph.mp4")

#     plt.show()
#     anim.save("both_working.mp4")






# model = ABM_model(parameters_ABM)
#     # set up initial condition
# model.set_initial_condition(parameters_ABM["initial_condition_type"])
# initial_grid_square(model, 0.8)
#     # show grid of initial conditions
# model.plot_grid()
#     # run simulation
# model.run(parameters_ABM["therapy"])

#     # plot data
# fig, ax = plt.subplots(1, 1)
# ax = model.plot_celltypes_density(ax)
# t = np.arange(1, model.T)*model.dt
# # ax.plot(t,model.R0*np.pi * np.exp(-model.drS*t), label="ODE Model")
# plt.show()

# if model.save_locations:
#     fig, ax, anim = model.animate_cells_graph(stride=10,interval=80)
#     anim.save("media/nice_abm.mp4")

    # animate data
    # fig,ax = plt.subplots()
    # fig,ax,anim = model.animate_cells((fig,ax))
    # anim.save("test_ABM.mp4")

    # animate graph
    # fig,ax = plt.subplots()
    # fig,ax,anim = model.animate_graph((fig,ax))
    # anim.save("test_ABM_graph.mp4")

    # plt.show()
    # anim.save("both_working.mp4")

    # # do a parameter sweep
    # therapies = ['notherapy', 'continuous', 'adaptive']
    # initial_conditions_types = ['random', 'cluster', 'two_clusters']
    # S0s = [50, 100, 200]
    # for initial_condition_type in initial_conditions_types:
    #     for S0 in S0s:
    #             # set up model
    #         model = ABM_model(N, T, S0, R0, grS, grR, drS, drR, divrS)
    #             # set up initial condition
    #         model.set_initial_condition(initial_condition_type)
    #             # show grid of initial conditions
    #             # model.print_grid()
    #             # run simulation
    #         model.run(therapy)
    #             # # plot data
    #         fig, ax = plt.subplots(1, 1)
    #         ax = model.plot_celltypes_density(ax)
    #         ax.set_title('Therapy: {}, Initial condition: {}, S0: {}'.format(therapy, initial_condition_type, S0))
    #             # # save figure
    #         fig.savefig('elene_{}_{}_{}.png'.format(therapy, initial_condition_type, S0))
    #         plt.close(fig)

