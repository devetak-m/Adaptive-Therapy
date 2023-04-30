import numpy as np

def compute_denisity_grid(U, parameters):
    x = int(np.sqrt(len(U)//3))
    return np.reshape(U[:x**2], (x,x)) + np.reshape(U[x**2:2*x**2], (x,x)) + np.reshape(U[2*x**2:], (x,x))

def unpack_parameters(parameters):

    array = [
        {
            'type': 'S',
            'distribution': parameters['S0_distribution'],
            'extra_parameters': parameters['S0_extra_parameters'],
            'growth_rate': parameters['growth_rate_S'],
            'diffusion_coefficient': parameters['diffusion_coefficient_S'],
            'standard_deviation': parameters['standard_deviation_S'],
            'death_rate': parameters['death_rate_S'],
            'division_rate': parameters['division_rate_S'],
            'carrying_capacity': parameters['carrying_capacity'],
            'diffusion_type': parameters['diffusion_type'],
            'maximum_tolerated_dose': parameters['maximum_tolerated_dose'],
            'kill_probability': parameters['kill_probability']
        }, 
        {
            'type': 'R',
            'distribution': parameters['R0_distribution'],
            'extra_parameters': parameters['R0_extra_parameters'],
            'growth_rate': parameters['growth_rate_R'],
            'diffusion_coefficient': parameters['diffusion_coefficient_R'],
            'standard_deviation': parameters['standard_deviation_R'],
            'death_rate': parameters['death_rate_R'],
            'division_rate': None,
            'carrying_capacity': parameters['carrying_capacity'],
            'diffusion_type': parameters['diffusion_type'],
            'maximum_tolerated_dose': parameters['maximum_tolerated_dose'],
            'kill_probability': parameters['kill_probability']
        },
        {
            'type': 'N',
            'distribution': parameters['N0_distribution'],
            'extra_parameters': parameters['N0_extra_parameters'],
            'growth_rate': parameters['growth_rate_N'],
            'diffusion_coefficient': parameters['diffusion_coefficient_N'],
            'standard_deviation': parameters['standard_deviation_N'],
            'death_rate': parameters['death_rate_N'],
            'division_rate': parameters['division_rate_N'],
            'carrying_capacity': parameters['carrying_capacity'],
            'diffusion_type': parameters['diffusion_type'],
            'maximum_tolerated_dose': parameters['maximum_tolerated_dose'],
            'kill_probability': parameters['kill_probability']
        }
    ]

    return array

def therapy_drug_concentration(density, parameters):

    therapy_type = parameters['therapy_type']
    maximum_tollerated_dose = parameters['maximum_tollerated_dose']

    if therapy_type == 'continuous':
        return maximum_tollerated_dose
    if therapy_type == 'notherapy':
        return 0
    elif therapy_type == 'adaptive':
        initial_size = parameters['S0'] + parameters['R0'] + parameters['N0']
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
                return maximum_tollerated_dose
        # off therapy
        else:
            # if grown sufficiently, turn on therapy
            if current_size > initial_size:
                parameters['current_state'] = 1
                return maximum_tollerated_dose
            else:
            # else, keep off therapy
                return 0
            
def compute_diffusion_coefficient(value, parameters):
    if parameters['diffusion_type'] == 'standard':
        return parameters['diffusion_coefficient']
    else:
        raise ValueError('Invalid diffusion type')
    
def compute_scheme(i, j, parameters, density, neighbourhood):

    point_value = neighbourhood[0]
    drug_concentration = therapy_drug_concentration(density,parameters)
    diffusion_coefficient = compute_diffusion_coefficient(point_value, parameters)
    maximum_tollerated_dose = parameters['maximum_tolerated_dose']
    carrying_capacity = parameters['carrying_capacity']
    growth_rate = parameters['growth_rate']
    death_rate = parameters['death_rate']
    space_step = (parameters['space_end'] - parameters['space_start']) / parameters['space_points']
    division_rate = parameters['division_rate']

    if parameters['type'] == 'R':
        return growth_rate * (1 - density/carrying_capacity) * point_value\
        - death_rate * point_value\
        + diffusion_coefficient/(space_step**2) * (sum(neighbourhood[1:]-point_value))

    elif parameters['type'] == 'S' or parameters['type'] == 'N':
        return growth_rate * (1 - density/carrying_capacity)\
        * (1 - 2*division_rate*(drug_concentration/maximum_tollerated_dose)) * point_value\
        - death_rate * point_value\
        + diffusion_coefficient/(space_step**2) * (sum(neighbourhood[1:]-point_value))
    
    else:
        raise ValueError('Type must be S, R or N')
