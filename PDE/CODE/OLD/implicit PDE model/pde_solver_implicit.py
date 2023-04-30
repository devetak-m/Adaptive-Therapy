import numpy as np
import matplotlib.pyplot as plt
from initial_conditions import set_up_initial_condition
from helping_functions import unpack_solution, draw_solution


def set_initial_conditions(parameters):

    S0 = parameters['S0']
    R0 = parameters['R0']
    N0 = parameters['N0']

    S0_distribution = parameters['S0_distribution']
    R0_distribution = parameters['R0_distribution']
    N0_distribution = parameters['N0_distribution']

    S0_extra_parameters = parameters['S0_extra_parameters']
    R0_extra_parameters = parameters['R0_extra_parameters']
    N0_extra_parameters = parameters['N0_extra_parameters']

    space_points = parameters['space_points']
    space_start = parameters['space_start']
    space_end = parameters['space_end']

    S = set_up_initial_condition(S0, S0_distribution, space_points, space_start, space_end, S0_extra_parameters)
    R = set_up_initial_condition(R0, R0_distribution, space_points, space_start, space_end, R0_extra_parameters)
    N = set_up_initial_condition(N0, N0_distribution, space_points, space_start, space_end, N0_extra_parameters)

    return np.vstack(S, R, N)

def pde_3D_model_implicit(parameters):

    # set up time grid
    time_points = parameters['time_points']
    time_start = parameters['time_start']
    time_end = parameters['time_end']
    time_points = np.linspace(time_start, time_end, time_points + 1)

    initial_conditions = set_initial_conditions(parameters)

    
    solver.set_initial_condition(initial_conditions)

    rhs = set_rhs(parameters)
    K = set_jacobian(parameters)

    solver = odespy.BackwardEuler(rhs, f_is_linear=True, jac=K)
    solver = odespy.ThetaRule(rhs, f_is_linear=True, jac=K, theta=0.5)

    u, t = solver.solve(time_points)  

    S, R, N, D, X, T = unpack_solution(u, t, parameters)   

    return S, R, N, D, X, T



if __name__ == " __main__ ":

    parameters = {
        'time_start': 0,
        'time_end': 100,
        'time_points': 10_000,
        'space_start': 0,
        'space_end': 0.5,
        'space_points': 100,
        'S0': 1.4,
        'R0': 0.02,
        'N0': 1.42,
        'S0_distribution': 'normal',
        'R0_distribution': 'normal',
        'N0_distribution': 'uniform',
        'S0_extra_parameters': [0.1, 0.1],
        'R0_extra_parameters': [0.1, 0.1],
        'N0_extra_parameters': [0.1, 0.1],
        'growth_rate_S': 0.03,
        'growth_rate_R': 0.03,
        'growth_rate_N': 0.03,
        'carrying_capacity': 2,
        'diffusion_coefficient_S': 0.0001,
        'diffusion_coefficient_R': 0.0001,
        'diffusion_coefficient_N': 0.0001,
        'standard_deviation_S': 0.01,
        'standard_deviation_R': 0.01,
        'standard_deviation_N': 0.01,
        'maximum_tollerated_dose': 1,
        'death_rate_S': 0.03,
        '': 0.03,
        'death_rate_N': 0.03,
        'division_rate_S': 0.4,
        'division_rate_N': 0.4,
        'therapy_type': 'continuous',
        'time_boundary_conditions': 'Periodic',
        'S0_left': 0,
        'R0_left': 0,
        'S0_right': 0,
        'R0_right': 0,
        'diffusion_type': 'standard'
    }

    S, R, N, D, X, T = pde_3D_model_implicit(parameters)

    draw_solution(S, R, N, D, X, T, parameters, show = True, save = True, save_name = 'implicit_3D_model', save_path = 'implicit_3D_model')