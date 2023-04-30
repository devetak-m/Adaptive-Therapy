import numpy as np

def from_2D_to_1D(x, y, space_points):
    return y * space_points + x

def from_1D_to_2D(index, space_points):
    return index % space_points, index // space_points

def unpack_solution(u, t, parameters):

    return None

def draw_solution(S, R, N, D, X, T, parameters, show = True, save = True, save_name = 'implicit_3D_model', save_path = 'implicit_3D_model'):

    return None

