import helping_functions as hf
from pde_model_implicit import pde_3D_model_implicit



parameters = {
    'time_start': 0,
    'time_end': 1,
    'time_step': 0.1,
    'space_start': 0,
    'space_end': 10,
    'space_points': 10,
    'S0': 1,
    'R0': 0,
    'N0': 0,
    'S0_distribution': 'patch',
    'R0_distribution': 'uniform',
    'N0_distribution': 'uniform',
    'S0_extra_parameters': ['circle', 3, 3, 0],
    'R0_extra_parameters': [0.1, 0.1],
    'N0_extra_parameters': [0.1, 0.1],
    'growth_rate_S': 0.04,
    'growth_rate_R': 0.04,
    'growth_rate_N': 0.04,
    'carrying_capacity': 4.9,
    'diffusion_coefficient_S': 0.1,
    'diffusion_coefficient_R': 0.0001,
    'diffusion_coefficient_N': 0.0001,
    'standard_deviation_S': 0.01,
    'standard_deviation_R': 0.01,
    'standard_deviation_N': 0.01,
    'maximum_tolerated_dose': 1,
    'death_rate_S': 0,
    'death_rate_R': 0.03,
    'death_rate_N': 0.03,
    'division_rate_S': 0.3,
    'division_rate_N': 0.3,
    'therapy_type': 'notherapy',
    'current_state': 1,
    'time_boundary_conditions': 'Periodic',
    'S0_left': 0,
    'R0_left': 0,
    'S0_right': 0,
    'R0_right': 0,
    'diffusion_type': 'standard'
}

S, R, N, D, X, T = pde_3D_model_implicit(parameters)

hf.draw_solution(S, R, N, D, X, T, parameters, show = True, save = True, save_name = 'implicit_3D_model', save_path = 'implicit_3D_model')