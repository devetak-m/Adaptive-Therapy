import numpy as np
import matplotlib.pyplot as plt

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
    S = np.zeros((time_points, space_points, space_points))
    R = np.zeros((time_points, space_points, space_points))
    
    for i in range(time_points):
        S[i, :, :] = np.reshape(u[i, 0:space_points**2], (space_points, space_points))
        R[i, :, :] = np.reshape(u[i, space_points**2:2*space_points**2], (space_points, space_points))

        therapy_type = parameters['therapy_type']
        maximum_tolerated_dose = parameters['maximum_tolerated_dose']

        if therapy_type == 'continuous':
            D[i] = maximum_tolerated_dose
        elif therapy_type == 'notherapy':
            D[i] = 0
        elif therapy_type == 'adaptive':
            initial_size = np.mean(S[0, :, :]) + np.mean(R[0, :, :])
            current_size = np.mean(S[i, :, :])+ np.mean(R[i,:,:])
            current_state = parameters['current_state']
            # on therapy
            if current_state == 1:
                # if shrunk sufficiently, turn off therapy
                if current_size < initial_size/2:
                    parameters['current_state'] = 0
                    D[i] = 0
                else:
                # else, keep on therapy
                    D[i] = maximum_tolerated_dose
            # off therapy
            else:
                # if grown sufficiently, turn on therapy
                if current_size > initial_size:
                    parameters['current_state'] = 1
                    D[i] = maximum_tolerated_dose
                else:
                # else, keep off therapy
                    D[i] = 0
        else:
            raise ValueError('Invalid therapy type. Spelling?')

    return S, R, D, X, T

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

#added by kylie for the data_function file 
def density(data, time, parameters):
    #time frames
    #T = int((parameters['time_end']-parameters["time_start"])/parameters['time_step'])
    #area
    x = parameters['space_end']- parameters['space_start']
    A = x*x
    
    meandensity = []

    for i in range(len(time)):
        rho = np.mean(data[i,:,:])
        meandensity.append(rho)
    return meandensity 

def densityplot(S,R,D,T):

    plt.figure()
    plt.plot(T, S, label='S')
    plt.plot(T, R, label='R')
    plt.plot(T, S+R, label='total number of cells', linestyle = '--')
    plt.plot(T, D, label = 'D')
    # naming the x axis
    plt.xlabel('time')
    # naming the y axis
    plt.ylabel('average number density PDE model')
    plt.title('cell density plot')
    plt.legend()
    # show a legend on the plot
    #plt.legend()
    plt.grid()
    # function to show the plot
    plt.show()


""" scaling laws for PDE model

recall that PDE model with uniform will distribute stuff all around
make sure ratio of cells to carrying capacity is identical to the ODE model

"""