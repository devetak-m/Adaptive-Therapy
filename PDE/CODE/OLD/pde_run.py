import numpy as np
import matplotlib.pyplot as plt
from initial_conditions import set_up_initial_condition
from helping_functions import unpack_solution, draw_solution
from broyden_methods import broyden_method_good
from construct_implicit import construct_F
from joblib import Parallel, delayed
import os
import itertools as it
import pandas as pd
import matplotlib.animation as animation

def pde_model(parameters):

    time_step = parameters['time_step']
    time_start = parameters['time_start']
    time_end = parameters['time_end']

    space_points = parameters['space_points']
    space_start = parameters['space_start']
    space_end = parameters['space_end']

    initial_condition_name = parameters['initial_condition_name']
    grid_size = parameters['grid_size']

    folder_name = f"initial_conditions"

    if grid_size != 20:
        path_S = f'{folder_name}/{initial_condition_name}_sensitive_{grid_size}.npy'
        path_R = f'{folder_name}/{initial_condition_name}_resistant_{grid_size}.npy'
    else:
        path_S = f'{folder_name}/{initial_condition_name}_sensitive.npy'
        path_R = f'{folder_name}/{initial_condition_name}_resistant.npy'
    
    initial_conditions_S = np.load(path_S)
    initial_conditions_R = np.load(path_R)

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
            
        F = construct_F(final_arrray[i,:], parameters)
        final_arrray[i+1,:] , error, it_count, J_inverse_old = broyden_method_good(F, final_arrray[i,:], J_inverse_old)
        final_arrray[i+1, final_arrray[i+1,:] < 1e-10] = 0

    return final_arrray
    


def compute_data(parameters, test):
    # set up the test
    parameters['initial_condition_name'] = test[0]
    parameters['therapy_type'] = test[1]
    # change if different for resistant and sensitive
    parameters['diffusion_coefficient_S'] = test[2]
    parameters['diffusion_coefficient_R'] = test[2]
    parameters['grid_size'] = test[3]
    parameters['space_points'] = test[3]

    folder_name = f"data/data_diffusion_{test[2]}_grid_{test[3]}"
    
    # compute the data
    final_arrray = pde_model(parameters)

    # save the data
    np.save(f'{folder_name}/{test[0]}_{test[1]}.npy', final_arrray)
    print(f"Data Saved for {test[0]}_{test[1]}_diffusion_{test[2]}_grid_{test[3]}")

def density_plot(S,R,D,T, filename):

    # plot the evolution of the average density trough time
    # clear plot
    plt.figure()
    plt.plot(T, S, label='S')
    plt.plot(T, R, label='R')
    
    plt.plot(T, S+R, label='N', linestyle='--',linewidth=2)
    plt.plot(T, D, label='D')
    plt.xlabel('Time')
    plt.ylabel('Average Density')
    plt.xlim=(0, T[-1])
    plt.title("Density of S and R")
    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_data(parameters, test):
    # set up the test
    parameters['initial_condition_name'] = test[0]
    parameters['therapy_type'] = test[1]
    # change if different for resistant and sensitive
    parameters['diffusion_coefficient_S'] = test[2]
    parameters['diffusion_coefficient_R'] = test[2]
    parameters['grid_size'] = test[3]
    parameters['space_points'] = test[3]

    folder_name = f"data/data_diffusion_{test[2]}_grid_{test[3]}"
    
    # load the data
    final_arrray = np.load(f'{folder_name}/{test[0]}_{test[1]}.npy')

    # plot the data
    S, R, D, X, T = unpack_solution(final_arrray, parameters)
    S_density = np.mean(S, axis=1)
    S_density = np.mean(S_density, axis=1)
    R_density = np.mean(R, axis=1)
    R_density = np.mean(R_density, axis=1)

    folder_name = f"plots/plots_diffusion_{test[2]}_grid_{test[3]}"

    density_plot(S_density,R_density,D,T, f"{folder_name}/final_ic_therapy_test_{test[0]}_{test[1]}.png")

def plot_bar_chart(parameters, diffusion_type, initial_conditions_names, therapy_types, grid_size):

    # compute the time to progression
    time_to_progression = np.zeros((len(initial_conditions_names), len(therapy_types)))
    for i in range(len(initial_conditions_names)):
        for j in range(len(therapy_types)):
            final_arrray = np.load(f"data/data_diffusion_{diffusion_type}_grid_{grid_size}/{initial_conditions_names[i]}_{therapy_types[j]}.npy")
            parameters['initial_condition_name'] = initial_conditions_names[i]
            parameters['therapy_type'] = therapy_types[j]
            parameters['space_points'] = grid_size
            S, R, D, X, T = unpack_solution(final_arrray, parameters)
            S_density = np.mean(S, axis=1)
            S_density = np.mean(S_density, axis=1)
            R_density = np.mean(R, axis=1)
            R_density = np.mean(R_density, axis=1)
            starting_density = S_density[0] + R_density[0]
            time_to_progression[i,j] = 3000
            for k in range(len(S_density)):
                if S_density[k] + R_density[k] > starting_density*1.5:
                    time_to_progression[i,j] = T[k]
                    break
    percentage_increase = np.zeros(len(initial_conditions_names))
    for i in range(len(initial_conditions_names)):
        percentage_increase[i] = 100 * (time_to_progression[i,0] - time_to_progression[i,1])/(time_to_progression[i,1])
    #percentage_increase[-1] = 8.12 # ODE
    #initial_conditions_names.append('ODE')
    fig,ax = plt.subplots()
    bar = ax.barh(initial_conditions_names,percentage_increase)
    #bar[list(initial_conditions_names).index("ODE")].set_color("red")
    ax.set_xlabel("Percentage increase in adaptive TTP over continuous T")
    ax.vlines(0,-0.5,len(initial_conditions_names)-0.5,linestyle="--",color="black")

    # label_locations = np.arange(len(initial_conditions_names))
    # width = 0.2
    # multiplier = 0
    # colors = ["#bf0202","#0b2eb0","#00940c"]
    # fig, ax = plt.subplots(constrained_layout=True)
    # for i,therapy_type in enumerate(therapy_types):
        # offset = width * multiplier
        # rects = ax.barh(label_locations + offset, time_to_progression[:,i], width, label=therapy_type,color=colors[i])
        # multiplier += 1

    #ax.set_title('Time to progression comparison')
    #ax.set_yticks(label_locations + width, initial_conditions_names)
    #ax.invert_yaxis()
    ax.legend()
    plt.tight_layout()

    plt.savefig(f"time_to_progression_bar_chart.png")
    plt.close()

def make_video(parameters, test):
    # set up the test
    parameters['initial_condition_name'] = test[0]
    parameters['therapy_type'] = test[1]
    # change if different for resistant and sensitive
    parameters['diffusion_coefficient_S'] = test[2]
    parameters['diffusion_coefficient_R'] = test[2]
    parameters['grid_size'] = test[3]
    parameters['space_points'] = test[3]

    folder_name = f"data/data_diffusion_{test[2]}_grid_{test[3]}"
    save_folder_name = f"videos/videos_diffusion_{test[2]}_grid_{test[3]}"

    # load the data
    final_arrray = np.load(f'{folder_name}/{test[0]}_{test[1]}.npy')
    S, R, D, X, T = unpack_solution(final_arrray, parameters)

    # Create a figure and a set of subplots
    fig, ax = plt.subplots()
    im = ax.contourf(S[0,:, :], cmap='plasma')

    # Method to change the contour data points
    def update(i):
        ax.clear()
        im = ax.contourf(S[i * 10,:, :], cmap='plasma')

    # Call animate method
    anim = animation.FuncAnimation(fig, update, 150)
    save_path = f"video/{test[0]}_{test[1]}_{test[2]}_grid_{test[3]}_susceptible.mp4"
    #anim.save(save_path)
    # anim.save(f"{save_folder_name}/{test[0]}_{test[1]}_susceptible.mp4")

###
    fig, ax = plt.subplots()
    im = ax.contourf(R[0,:, :], cmap='plasma')
    # Method to change the contour data points
    def update(i):
        ax.clear()
        im = ax.contourf(R[i * 10,:, :], cmap='plasma')
    # Call animate method
    #anim = animation.FuncAnimation(fig, update, 150)
    #save_path = f"video/{test[0]}_{test[1]}_{test[2]}_grid_{test[3]}_resistant.mp4"
    #anim.save(save_path)
###

    fig, ax = plt.subplots()
    ax.set_facecolor('xkcd:salmon')
    ax.set_facecolor((1.0, 1, 1))
    ax.axis('off')
    im = ax.imshow(S[0,:, :] + R[0,:,:], vmax=np.log(15))
    # Method to change the contour data points
    def update(i):
        ax.clear()
        ax.set_facecolor('xkcd:salmon')
        ax.set_facecolor((1.0, 1, 1))
        ax.axis('off')
        im = ax.imshow(np.log(S[i * 10,:, :] + R[i * 10,:,:]), vmax=np.log(15))
    # Call animate method
    anim = animation.FuncAnimation(fig, update, 150)
    save_path = f"video/{test[0]}_{test[1]}_{test[2]}_grid_{test[3]}_total.mp4"
    anim.save(save_path, fps=20)

if __name__ == "__main__":
     
    parameters = {
        'time_start': 0,
        'time_end': 1500, # 1500
        'time_step': 1,
        'space_start': 0,
        'space_end': 1,
        'space_points': 20,
        'S0': 70,
        'R0': 0.7,
        'S0_distribution': 'uniform',
        'R0_distribution': 'uniform',
        'S0_extra_parameters': ['circle', 0, 0,1],
        'R0_extra_parameters': [0.1, 0.1],
        'growth_rate_S': 0.023,
        'growth_rate_R': 0.023,
        'carrying_capacity': 15,
        'diffusion_coefficient_S': 0.0001,
        'diffusion_coefficient_R': 0.0001,
        'diffusion_coefficient_N': 0.0001,
        'standard_deviation_S': 0.01,
        'standard_deviation_R': 0.01,
        'maximum_tolerated_dose': 1,
        'death_rate_S': 0.01,
        'death_rate_R': 0.01,
        'division_rate_S': 0.75,
        'therapy_type': 'adaptive',
        'current_state': 1,
        'time_boundary_conditions': 'Periodic',
        'S0_left': 0,
        'R0_left': 0,
        'S0_right': 0,
        'R0_right': 0,
        'diffusion_type': 'standard',
        'initial_condition_name': 'uniform'
        }
    
    initial_conditions_names = ['resistant_core']
    therapy_types = ['adaptive' , 'continuous', 'notherapy']
    # diffusion_types = [0] # , 1e-6, 1e-8, 1e-10]
    diffusion_types = [1e-7]
    grid_size = [40]
    N_JOBS = 1

    # make all possible combinations of initial conditions and therapy typesand diffusion
    tests = list(it.product(initial_conditions_names, therapy_types, diffusion_types, grid_size))

    # create folder to store results based on diffusion type
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists("plots"):
        os.makedirs("plots")
    if not os.path.exists("video"):
        os.makedirs("video")
    
    for test in tests:
        if not os.path.exists(f"data/data_diffusion_{test[2]}_grid_{test[3]}"):
            os.makedirs(f"data/data_diffusion_{test[2]}_grid_{test[3]}")
        if not os.path.exists(f"plots/plots_diffusion_{test[2]}_grid_{test[3]}"):
            os.makedirs(f"plots/plots_diffusion_{test[2]}_grid_{test[3]}")
        if not os.path.exists(f"video/video_diffusion_{test[2]}_grid_{test[3]}"):
            os.makedirs(f"video/video_diffusion_{test[2]}_grid_{test[3]}")

    # compute the data
    # Parallel(n_jobs=N_JOBS, verbose=3)(delayed(compute_data)(parameters.copy(), test) for test in tests)

    print('Data Computed')

    # plot the data
    Parallel(n_jobs=N_JOBS, verbose=3)(delayed(plot_data)(parameters.copy(), test) for test in tests)
    
    print('Data Plotted')
    
    # compute bar charts
    # bar_charts = list(it.product(diffusion_types, grid_size))
    #Parallel(n_jobs=N_JOBS, verbose=3)(delayed(plot_bar_chart)(parameters.copy(), bar_chart[0], initial_conditions_names, therapy_types, bar_chart[1]) for bar_chart in bar_charts)
    
    print('Bar Charts Plotted')

    # compute video
    Parallel(n_jobs=N_JOBS, verbose=3)(delayed(make_video)(parameters.copy(), test) for test in tests)
    
    # for test in tests:
        # make_video(parameters.copy(), test)
    

    print('Video Made')


    




    
            
