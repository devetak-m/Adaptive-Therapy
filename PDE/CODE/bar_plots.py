import numpy as np
import matplotlib.pyplot as plt
import itertools as it
from helping_functions import unpack_solution

initial_conditions_names = ['uniform', 'uniform_ball_1', 'uniform_ball_0.9', 'uniform_ball_0.8', 'resistant_rim', 'resistant_core', 'multiple_resistant_rims','multiple_resistant_cores']
therapy_types = ['adaptive', 'continuous']

# combine the two lists into a list of tuples
initial_conditions_and_therapy_types = list(it.product(initial_conditions_names, therapy_types))


parameters = {
        'time_start': 0,
        'time_end': 3,
        'time_step': 1,
        'space_start': 0,
        'space_end': 1,
        'space_points': 20,
        'S0': 100,
        'R0': 1,
        'S0_distribution': 'uniform',
        'R0_distribution': 'uniform',
        'S0_extra_parameters': ['circle', 0, 0,1],
        'R0_extra_parameters': [0.1, 0.1],
        'growth_rate_S': 0.023,
        'growth_rate_R': 0.023,
        'carrying_capacity': 50,
        'diffusion_coefficient_S': 0.0001,
        'diffusion_coefficient_R': 0.0001,
        'diffusion_coefficient_N': 0.0001,
        'standard_deviation_S': 0.01,
        'standard_deviation_R': 0.01,
        'maximum_tolerated_dose': 1,
        'death_rate_S': 0.03,
        'death_rate_R': 0.03,
        'division_rate_S': 0.75,
        'division_rate_N': 0.013,
        'therapy_type': 'adaptive',
        'current_state': 1,
        'time_boundary_conditions': 'Periodic',
        'S0_left': 0,
        'R0_left': 0,
        'S0_right': 0,
        'R0_right': 0,
        'diffusion_type': 'standard',
        'initial_condition_name': 'uniform',

        #masking option - check this 
        'cut': 'off', #can be on or off depending on cutout 
        'cut_tolerence': 1e-2
        }

# iterate through the list of tuples
for initial_condition_name, therapy_type in initial_conditions_and_therapy_types:
    # final_arrray_multiple_resistant_rims_adaptive.txt
    file_name = f"final_arrray_{initial_condition_name}_{therapy_type}.txt"
    parameters['initial_condition_name'] = initial_condition_name
    parameters['therapy_type'] = therapy_type
    # read the file
    with open(file_name, 'r') as f:
        # load the file as a numpy array
        final_arrray = np.loadtxt(f)
        # unpack the solution
        S, R, D, X, T = unpack_solution(final_arrray, parameters)
        # calculate the mean density
        S_density = np.mean(S, axis=1)
        S_density = np.mean(S_density, axis=1)
        R_density = np.mean(R, axis=1)
        R_density = np.mean(R_density, axis=1)
        # compute the time to progression defined as the first time S + R > 1.5 * starting density
        # find the index of the first time S + R > 1.5 * starting density
        index = np.argmax(S_density + R_density > 1.5 * S_density[0] + R_density[0])
        # compute the time to progression
        time_to_progression = T[index]
        # print the time to progression
        print(f"Time to progression for {initial_condition_name} and {therapy_type} is {time_to_progression}")







    # S, R, D, X, T = unpack_solution(final_arrray, parameters)
    # S_density = np.mean(S, axis=1)
    # S_density = np.mean(S_density, axis=1)
    # R_density = np.mean(R, axis=1)
    # R_density = np.mean(R_density, axis=1)