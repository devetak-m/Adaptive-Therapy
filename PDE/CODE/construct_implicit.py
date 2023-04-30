import numpy as np
from initial_conditions import *

def unpack_parameters(parameters):

    separated_parameters = [
        {
            'type': 0,
            'extra_parameters': parameters['S0_extra_parameters'],
            'growth_rate': parameters['growth_rate_S'],
            'diffusion_coefficient': parameters['diffusion_coefficient_S'],
            'death_rate': parameters['death_rate_S'],
            'division_rate': parameters['division_rate_S'],
            'carrying_capacity': parameters['carrying_capacity'],
            'diffusion_type': parameters['diffusion_type'],
            'maximum_tolerated_dose': parameters['maximum_tolerated_dose'],
            'space_start': parameters['space_start'],
            'space_end': parameters['space_end'],
            'space_points': parameters['space_points']
        }, 
        {
            'type': 1,
            'extra_parameters': parameters['R0_extra_parameters'],
            'growth_rate': parameters['growth_rate_R'],
            'diffusion_coefficient': parameters['diffusion_coefficient_R'],
            'death_rate': parameters['death_rate_R'],
            'division_rate': 0,
            'carrying_capacity': parameters['carrying_capacity'],
            'diffusion_type': parameters['diffusion_type'],
            'maximum_tolerated_dose': parameters['maximum_tolerated_dose'],
            'space_start': parameters['space_start'],
            'space_end': parameters['space_end'],
            'space_points': parameters['space_points']
        }
    ]

    return separated_parameters

def compute_denisity_grid(U, parameters):

    space_points = parameters['space_points']
    return_grid = np.zeros((space_points, space_points))
    
    for i in range(space_points):
        for j in range(space_points):
            k1 = i * space_points + j
            k2 = k1 + space_points**2
            return_grid[i][j] = U[k1] + U[k2]
    return return_grid
    
def compute_diffusion_coefficient(value, parameters):

    if parameters['diffusion_type'] == 'standard':
        return parameters['diffusion_coefficient']
    else:
        raise ValueError('Invalid diffusion type')


def compute_scheme(parameters, density, sum_diffusion, point_value):

    drug_concentration = parameters['drug_concentration']
    diffusion_coefficient = compute_diffusion_coefficient(point_value, parameters)
    maximum_tolerated_dose = parameters['maximum_tolerated_dose']
    carrying_capacity = parameters['carrying_capacity']
    growth_rate = parameters['growth_rate']
    death_rate = parameters['death_rate']
    space_end = parameters['space_end']
    space_start = parameters['space_start']
    space_points = parameters['space_points']
    space_step = (space_end - space_start) / space_points
    division_rate = parameters['division_rate']


    effectivie_growth_rate = growth_rate * (1 - density/carrying_capacity)*(1 - 2*division_rate*(drug_concentration/maximum_tolerated_dose)) - death_rate
    effective_diffusion_coefficient = diffusion_coefficient/(space_step**2)

    return effectivie_growth_rate * point_value + effective_diffusion_coefficient * (sum_diffusion)

def therapy_drug_concentration(density, parameters):

    therapy_type = parameters['therapy_type']
    maximum_tolerated_dose = parameters['maximum_tolerated_dose']

    if therapy_type == 'continuous':
        return maximum_tolerated_dose
    if therapy_type == 'notherapy':
        return 0
    elif therapy_type == 'adaptive':
        initial_size = parameters['S0'] + parameters['R0']
        current_size = np.sum(density)
        current_state = parameters['current_state']
        # on therapy
        if current_state == 1:
            # if shrunk sufficiently, turn off therapy
            if current_size < initial_size/2:
                parameters['current_state'] = 0
                return 0
            else:
            # else, keep on therapy
                return maximum_tolerated_dose
        # off therapy
        else:
            # if grown sufficiently, turn on therapy
            if current_size > initial_size:
                parameters['current_state'] = 1
                return maximum_tolerated_dose
            else:
            # else, keep off therapy
                return 0
    else:
        raise ValueError(f'Invalid therapy type. Spelling? {therapy_type}')

def construct_F(U, parameters):

    density_grid = compute_denisity_grid(U, parameters)
    density_grid = density_grid.reshape((parameters['space_points']**2))
    parameters_by_type = unpack_parameters(parameters)
    time_step = parameters['time_step']
    space_points = parameters['space_points']
    drug_concentration = therapy_drug_concentration(density_grid, parameters)

    for dic in parameters_by_type:
        dic['drug_concentration'] = drug_concentration
    
    def F(solution):
        
        result = solution[parameters['left']] + solution[parameters['right']] + solution[parameters['up']] + solution[parameters['down']] - 4 * solution

        for k in range(len(solution)):

            time_derivative = (solution[k] - U[k]) / time_step
            # i, j, type = unpack_index(k, parameters)
            type = k // (space_points**2)
            # neighbourhood_coordinates = get_neighbourhood(k, space_points)
            # neighbourhood = [solution[int(h)] for h in neighbourhood_coordinates]
            result[k] = compute_scheme(parameters_by_type[type], density_grid[k - type * space_points**2], result[k], solution[k]) - time_derivative

        return result

    return F