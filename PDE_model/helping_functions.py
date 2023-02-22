import numpy as np

def from_2D_to_1D(x, y, space_points):
    return y * space_points + x

def from_1D_to_2D(index, space_points):
    return index % space_points, index // space_points

# S, R, N, D, X, T = unpack_solution(U, parameters)

def unpack_solution(u, parameters):

    space_points = parameters['space_points']
    space_start = parameters['space_start']
    space_end = parameters['space_end']
    time_start = parameters['time_start']
    time_end = parameters['time_end']
    time_step = parameters['time_step']
    time_points = int((time_end - time_start)/time_step)

    X = np.linspace(space_start, space_end, space_points)
    T = np.linspace(time_start, time_end, time_points)
    D = np.zeros(time_points)
    S  = np.zeros((time_points, space_points, space_points))
    R  = np.zeros((time_points, space_points, space_points))
    N  = np.zeros((time_points, space_points, space_points))
    
    for i in range(time_points):
        S[i, :, :] = np.reshape(u[i, 0:space_points**2], (space_points, space_points))
        R[i, :, :] = np.reshape(u[i, space_points**2:2*space_points**2], (space_points, space_points))
        N[i, :, :] = np.reshape(u[i, 2*space_points**2:3*space_points**2], (space_points, space_points))

    return S, R, N, D, X, T

def draw_solution(S, R, N, D, X, T, parameters, show = True, save = True, save_name = 'implicit_3D_model', save_path = 'implicit_3D_model'):

    print("draw_solution")
    # print(S.shape)
    # print(R.shape)
    # print(N.shape)
    # print(D)
    # print(X)
    # print(T)

    # print("draw_solution")

    # print(S)



    return None

