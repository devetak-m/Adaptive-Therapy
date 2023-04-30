import numpy as np
import matplotlib.pyplot as plt
from initial_conditions import set_up_initial_condition
from helping_functions import unpack_solution, draw_solution
from broyden_methods import broyden_method_good
from construct_implicit import construct_F
from joblib import Parallel, delayed
import itertools as it

def pde_model(parameters):

    time_step = parameters['time_step']
    time_start = parameters['time_start']
    time_end = parameters['time_end']

    space_points = parameters['space_points']
    space_start = parameters['space_start']
    space_end = parameters['space_end']

    initial_condition_name = parameters['initial_condition_name']

    # S0 = parameters['S0']
    # S0_distribution = parameters['S0_distribution']
    # S0_extra_parameters = parameters['S0_extra_parameters']
    # initial_S = set_up_initial_condition(S0, S0_distribution, space_points, space_start, space_end, S0_extra_parameters)
    
    # R0 = parameters['R0']
    # R0_distribution = parameters['R0_distribution']
    # R0_extra_parameters = parameters['R0_extra_parameters']
    # initial_R = set_up_initial_condition(R0, R0_distribution, space_points, space_start, space_end, R0_extra_parameters)

    # read the initial conditions from uniform_ball_0.9_resistant.npy in initial_conditions folder
    initial_conditions_R = np.load(f'/Users/mitjadevetak/Desktop/implicit_pde_final/initial_conditions/{initial_condition_name}_resistant.npy')
    initial_conditions_S = np.load(f'/Users/mitjadevetak/Desktop/implicit_pde_final/initial_conditions/{initial_condition_name}_sensitive.npy')
    # print("Initial Conditions Loaded")
    # print(f"Initial Conditions Shape: {initial_conditions_S.shape}")
    # print(f"Initial Conditions Shape: {initial_conditions_R.shape}")
    initial_S = initial_conditions_S.reshape(space_points**2)
    initial_R = initial_conditions_R.reshape(space_points**2)
    initial_S[initial_S < 0] = 0
    initial_R[initial_R < 0] = 0
    initial_S = parameters["S0"]* initial_S/np.sum(initial_S)
    initial_R = parameters["R0"]* initial_R/np.sum(initial_R)
    number_of_time_points = int((time_end - time_start)/time_step)

    final_arrray = np.zeros((number_of_time_points, 2*space_points**2))
    final_arrray[0, :] = np.concatenate((initial_S, initial_R))

    indices = np.linspace(0, space_points**2-1, space_points**2, dtype=int)

    indices_shape = indices.reshape(space_points, space_points)

    left = np.roll(indices_shape, 1, axis=0).reshape(space_points**2)
    right = np.roll(indices_shape, -1, axis=0).reshape(space_points**2)
    up = np.roll(indices_shape, 1, axis=1).reshape(space_points**2)
    down = np.roll(indices_shape, -1, axis=1).reshape(space_points**2)

    left = np.concatenate((left, left + space_points**2)).astype(int)
    right = np.concatenate((right, right + space_points**2)).astype(int)
    up = np.concatenate((up, up + space_points**2)).astype(int)
    down = np.concatenate((down, down + space_points**2)).astype(int)

    parameters['left'] = left
    parameters['right'] = right
    parameters['up'] = up
    parameters['down'] = down

    J_inverse_old = np.eye(2*space_points**2)

    for i in range(number_of_time_points-1):
    
            if i%10 ==0 :
                print(f"Reached Time Step {i}")
            
            F = construct_F(final_arrray[i,:], parameters)
            final_arrray[i+1,:] , error, it_count, J_inverse_old = broyden_method_good(F, final_arrray[i,:], J_inverse_old)
            # print(f"Error: {error}, Iterations: {it_count} for cc: {parameters['carrying_capacity']}")
            # clean the final array from negative and small values
            final_arrray[i+1, final_arrray[i+1,:] < 1e-10] = 0
            # np.savetxt('final_arrray.txt', final_arrray)

    # # adding the cut off parameter       
    # cut = parameters['cut']
    # cut_tolerence = parameters['cut_tolerence']

    # if cut == 'on':
    #      final_arrray_cut = final_arrray.copy()
    #      mask = final_arrray_cut <= cut_tolerence
    #      final_arrray_cut[mask] = 0
    #      return final_arrray_cut

    # if cut == 'off':
    #      return final_arrray
    # else:
    #      return ValueError('cut parameter should either be on or off')
    return final_arrray
    


def density_plot(S,R,D,T, filename):

    # plot the evolution of the average density trough time
    # clear plot
    plt.figure()
    plt.plot(T, S, label='S')
    plt.plot(T, R, label='R')
    plt.plot(T, D, label='D')
    plt.xlabel('Time')
    plt.ylabel('Average Density')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def test_diffusion(parameters, diffusion_constant):
         
    # test the diffusion constant
    parameters['diffusion_coefficient_S'] = diffusion_constant
    parameters['diffusion_coefficient_R'] = diffusion_constant
    
    final_arrray = pde_model(parameters)
    S, R, D, X, T = unpack_solution(final_arrray, parameters)
    S_density = np.mean(S, axis=1)
    S_density = np.mean(S_density, axis=1)
    R_density = np.mean(R, axis=1)
    R_density = np.mean(R_density, axis=1)

    density_plot(S_density,R_density,D,T, f"diffusion_test_{diffusion_constant}.png")


def test_cc(parameters, cc):
         
    # test the carrying capacity
    parameters['carrying_capacity'] = cc
    
    final_arrray = pde_model(parameters)
    S, R, D, X, T = unpack_solution(final_arrray, parameters)
    S_density = np.mean(S, axis=1)
    S_density = np.mean(S_density, axis=1)
    R_density = np.mean(R, axis=1)
    R_density = np.mean(R_density, axis=1)

    density_plot(S_density,R_density,D,T, f"cc_test_{cc}.png")

def test_final_ic_therapy(parameters, final_ic_therapy):
         

    # unpack final ic therapy
    parameters['initial_condition_name'] = final_ic_therapy[0]
    parameters['therapy_type'] = final_ic_therapy[1]
    print(f"Testing final ic therapy: {final_ic_therapy[0]} and {final_ic_therapy[1]}")

    final_arrray = pde_model(parameters)
    # save the final array
    np.savetxt(f"final_arrray_{final_ic_therapy[0]}_{final_ic_therapy[1]}.txt", final_arrray)


    S, R, D, X, T = unpack_solution(final_arrray, parameters)
    S_density = np.mean(S, axis=1)
    S_density = np.mean(S_density, axis=1)
    R_density = np.mean(R, axis=1)
    R_density = np.mean(R_density, axis=1)

    density_plot(S_density,R_density,D,T, f"final_ic_therapy_test_{final_ic_therapy[0]}_{final_ic_therapy[1]}.png")


if __name__ == "__main__":
     
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
    
    # final_arrray = pde_model(parameters)
    # S, R, D, X, T = unpack_solution(final_arrray, parameters)
    # S_density = np.mean(S, axis=1)
    # S_density = np.mean(S_density, axis=1)
    # R_density = np.mean(R, axis=1)
    # R_density = np.mean(R_density, axis=1)

    # density_plot(S_density,R_density,D,T, 'adaptive_therapy.png')

    # cc_arr = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # Parallel(n_jobs=8)(delayed(test_cc)(parameters.copy(), cc) for cc in cc_arr)

    initial_conditions_names = ['uniform', 'uniform_ball_1', 'uniform_ball_0.9', 'uniform_ball_0.8', 'resistant_rim', 'resistant_core', 'multiple_resistant_rims','multiple_resistant_cores']
    therapy_types = ['adaptive', 'continuous']
    # make all possible combinations of initial conditions and therapy types
    initial_conditions_and_therapy_types = list(it.product(initial_conditions_names, therapy_types))

    Parallel(n_jobs=1)(delayed(test_final_ic_therapy)(parameters.copy(), ic_thp) for ic_thp in initial_conditions_and_therapy_types)


    
            