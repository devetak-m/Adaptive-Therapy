import numpy as np
import matplotlib.pyplot as plt
from PDE_model.initial_conditions import set_up_initial_condition
from PDE_model.helping_functions import unpack_solution, draw_solution
from PDE_model.broyden_methods import broyden_method_good
from PDE_model.construct_implicit import construct_F

def pde_model(parameters):

    time_step = parameters['time_step']
    time_start = parameters['time_start']
    time_end = parameters['time_end']

    space_points = parameters['space_points']
    space_start = parameters['space_start']
    space_end = parameters['space_end']

    S0 = parameters['S0']
    S0_distribution = parameters['S0_distribution']
    S0_extra_parameters = parameters['S0_extra_parameters']
    initial_S = set_up_initial_condition(S0, S0_distribution, space_points, space_start, space_end, S0_extra_parameters)
    
    R0 = parameters['R0']
    R0_distribution = parameters['R0_distribution']
    R0_extra_parameters = parameters['R0_extra_parameters']
    initial_R = set_up_initial_condition(R0, R0_distribution, space_points, space_start, space_end, R0_extra_parameters)

    N0 = parameters['N0']
    N0_distribution = parameters['N0_distribution']
    N0_extra_parameters = parameters['N0_extra_parameters']
    initial_N = set_up_initial_condition(N0, N0_distribution, space_points, space_start, space_end, N0_extra_parameters)
    # show grid in plot
    # plt.grid()
    # plt.spy(np.reshape(initial_S, (space_points, space_points)))
    # plt.show()
    number_of_time_points = int((time_end - time_start)/time_step)

    final_arrray = np.zeros((number_of_time_points, 3*space_points**2))
    final_arrray[0, :] = np.concatenate((initial_S, initial_R, initial_N))

    for i in range(number_of_time_points-1):
    #        if i%10 ==0 :
    #            print(f"Reached Time Step {i}")
            F = construct_F(final_arrray[i,:], parameters)

            final_arrray[i+1,:] , error, it_count = broyden_method_good(F, final_arrray[i,:])
            if it_count > 10:
                print(it_count)
            # np.savetxt('final_arrray.txt', final_arrray)

    #adding the cut off parameter       
    cut = parameters['cut']
    cut_tolerence = parameters['cut_tolerence']

    if cut == 'on':
        final_arrray_cut = final_arrray.copy()
        mask = final_arrray_cut <= cut_tolerence
        final_arrray_cut[mask] = 0
        return final_arrray_cut

    if cut == 'off':
        return final_arrray
    else:
        return ValueError('cut parameter should either be on or off')



def pde_3D_model_implicit(parameters):

    print("pde_3D_model_implicit")

    U = pde_model(parameters)

    S, R, N, D, X, T = unpack_solution(U, parameters)

    return S, R, N, D, X, T