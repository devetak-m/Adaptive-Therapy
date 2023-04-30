import numpy as np
from helping_functions import from_2D_to_1D

def gaussian_cdf(x, y, extra_parameters = None):
        if extra_parameters is None:
            center_x = 0
            center_y = 0
            sigma_x = 1
            sigma_y = 1
        else:
            center_x = extra_parameters[0]
            center_y = extra_parameters[1]
            sigma_x = extra_parameters[2]
            sigma_y = extra_parameters[3]
    
        return np.exp(-((x - center_x) ** 2 / (2 * sigma_x ** 2) + (y - center_y) ** 2 / (2 * sigma_y ** 2)))

def is_in_patch(x, y, patch_type, center_x, center_y, side_length):
    if patch_type == 'square':
        if center_x - side_length / 2 <= x <= center_x + side_length / 2 and center_y - side_length / 2 <= y <= center_y + side_length / 2:
            return 1
        else:
            return 0
    elif patch_type == 'circle':
        if (x - center_x) ** 2 + (y - center_y) ** 2 <= (side_length / 2) ** 2:
            return 1
        else:
            return 0
    else:
        raise ValueError('Patch type not recognized')

def patch_cdf(x,y, extra_parameters = None):
    if extra_parameters is None:
        patch_type = 'square'
        center_x = 0
        center_y = 0
        side_length = 10
    else:
        patch_type = extra_parameters[0]
        center_x = extra_parameters[1]
        center_y = extra_parameters[2]
        side_length = extra_parameters[3]
    return is_in_patch(x, y, patch_type, center_x, center_y, side_length)

def set_value(x, y, X0, X0_distribution, extra_parameters = None):

    if X0_distribution == 'uniform':
        return X0
    elif X0_distribution == 'gaussian':
        return X0 * gaussian_cdf(x, y, extra_parameters)
    elif X0_distribution == 'patch':
        return X0 * patch_cdf(x, y, extra_parameters)

def set_up_initial_condition(X0, X0_distribution, space_points, space_start, space_end, extra_parameters):

    x = np.arange(space_points)

    initial_condition_array = np.zeros(space_points * space_points)
    scaling_factor = (space_end - space_start) / space_points

    for x_coordinate in x:
        for y_coordinate in x:
            array_coordinate = from_2D_to_1D(x_coordinate, y_coordinate, space_points)
            x_real = x_coordinate * scaling_factor + space_start
            y_real = y_coordinate * scaling_factor + space_start
            initial_condition_array[array_coordinate] = set_value(x_real, y_real, X0, X0_distribution, extra_parameters)

    return initial_condition_array