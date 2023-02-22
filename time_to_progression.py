from ABM_model import ABM_model
from ode_model import ode_model
import numpy as np
import matplotlib.pyplot as plt


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
    return

# define a function that takes in the parameters of the model and returns the statistics of the time to progression
# it also stores the densities trough time in a file and plots the average density of the model with error bars
def time_to_progression(params, nruns, threshold, filename):
    """ This function takes in the parameters of the model and returns the statistics of the time to progression
    it also stores the densities trough time in a file and plots the average density of the model with error bars"""

    # initialize the arrays that will store the statistics
    ttp = np.zeros(nruns)

    # initialize the arrays that will store the densities
    densities_S = np.zeros((nruns, int(params["T"] * (1/params["dt"]))))
    densities_R = np.zeros((nruns, int(params["T"] * (1/params["dt"]))))

    # construct the model
    model = ABM_model(params)
    # set the initial condition
    initial_grid_square(model, 0.9)

    # run the model nruns times
    for i in range(nruns):
        model.reset()
        model.run(params["therapy"])
        ttp[i] = model.time_to_progression(threshold)
        densities_S[i] = model.data[:, 0]
        densities_R[i] = model.data[:, 1]
    
    # save the densities in a file
    np.save(filename + f"{filename}_densities_S", densities_S)
    np.save(filename + f"{filename}_densities_R", densities_R)

    # plot the average density of the model with error bars
    plt.errorbar(np.arange(0, model.T), np.mean(densities_S, axis = 0), yerr = np.std(densities_S, axis = 0), label = "S")
    plt.errorbar(np.arange(0, model.T), np.mean(densities_R, axis = 0), yerr = np.std(densities_R, axis = 0), label = "R")
    # plot the total density
    plt.plot(np.arange(0, model.T), np.mean(densities_S, axis = 0) + np.mean(densities_R, axis = 0), label = "Total")
    # plot line indicating the threshold
    plt.axhline(y = threshold * (np.sum(model.data[0,:2])), color = "black", linestyle = "--")
    # label the plot
    plt.title("Average density of the model with error bars")
    # make error bars transparent
    plt.gca().collections[0].set_alpha(0.2)
    plt.gca().collections[1].set_alpha(0.2)
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("density")
    # save the plot
    plt.savefig(filename + f"_average_density" + ".png")
    # clear the plot
    plt.clf()


    print(ttp)
    # normalize the time to progression
    ttp = ttp / (1/params["dt"])

    # compute how many runs had a time to progression
    nttp = np.sum(ttp > 0)

    # compute the statistics of the time to progression
    mean_ttp = np.mean(ttp[ttp > 0])

    # compute the standard deviation of the time to progression
    std_ttp = np.std(ttp[ttp > 0])

    # print the statistics of the time to progression
    print("Number of runs with time to progression: ", nttp)
    print("Mean time to progression: ", mean_ttp)
    print("Standard deviation of time to progression: ", std_ttp)

    # return the statistics of the time to progression
    return nttp, mean_ttp, std_ttp

def ttp_ode(parameters, threshold):

    parameters["therapy_type"] = "adaptive"

    S, R, N, T, D = ode_model(parameters, verbose=False)

    plt.plot(T, S, label='S')
    plt.plot(T, R, label='R')
    plt.plot(T, N, label='N', linestyle='--')
    plt.plot(T, D, label='D')
    plt.legend()
    plt.show()

    # compute the time to progression
    ttpa = 0
    for i in range(len(D)):
        if S[i] + R[i] > threshold * (S[0] + R[0]):
            ttpa = i
            break
    
    # print the time to progression
    print("Time to progression with adaptive therapy: ", ttpa * parameters["time_step"])

    # change the parameters to simulate the model with continous therapy
    parameters["therapy_type"] = "continuous"

    S, R, N, T, D = ode_model(parameters, verbose=False)

    plt.plot(T, S, label='S')
    plt.plot(T, R, label='R')
    plt.plot(T, N, label='N', linestyle='--')
    plt.plot(T, D, label='D')
    plt.legend()
    plt.show()

    # compute the time to progression
    ttpc = 0
    for i in range(len(D)):
        if S[i] + R[i] > threshold * (S[0] + R[0]):
            ttpc = i
            break
    
    # print the time to progression
    print("Time to progression with continuous therapy: ", ttpc * parameters["time_step"])

    return ttpa, ttpc

if __name__ == "__main__":

    parameters_ABM = {
    "domain_size" : 40,
    "T" : 1500,
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
    "save_locations" : False,
    "dimension" : 2,
    "seed" : 0}

    time_to_progression(parameters_ABM, 5, 2, "continuous")

    parameters_ABM["therapy"] = "adaptive"

    time_to_progression(parameters_ABM, 5, 2, "adaptive")

    parameters_ode = {
        'time_start': 0,                                  
        'time_end': 2000,
        'time_step': 0.1,
        'tolerance': 100,
        'S0': 0.125,
        'R0': 0.00625,
        'growth_rate_S': 0.023,
        'growth_rate_R': 0.023,
        'carrying_capacity': 1,
        'maximum_tollerated_dose': 1,
        'death_rate_S': 0.01,
        'death_rate_R': 0.01,
        'division_rate': 0.75,
        'therapy_type': 'adaptive',
        'current_state': 1,
    }

    # ttp_ode(parameters_ode, 2)





