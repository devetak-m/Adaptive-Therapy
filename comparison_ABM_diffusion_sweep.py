import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from ABM_model import ABM_model
import pandas as pd 
import time
import pickle
import os 
import shutil
def run_ABM(parameters, i, theshold):
    # set the seed
    parameters['seed'] = i
    # construct the model
    model = ABM_model(parameters)
    # run the model
    model.run(parameters["therapy"])
    # get the densities
    density_S = model.data[:, 0]
    density_R = model.data[:, 1]
    # compute the time to progression
    ttp = parameters["T"]
    for j in range(model.T):
        if density_S[j] + density_R[j] > theshold * (parameters['S0'] + parameters['R0']):
            ttp = j
            break
    # return the time to progression and the densities
    return ttp, density_S, density_R


def comparison_ABM(parameters, nruns, theshold, folder_name = None):
    if folder_name is None:
        folder_name = f"results_ABM_diffusion/comparison_ABM_{parameters['therapy']}_initial_condition_{parameters['initial_condition_type']}"

    # print("Running time to progression for", folder_name, "...")

    # initialize the arrays that will store the statistics
    ttp = np.zeros(nruns)

    # initialize the arrays that will store the densities
    densities_S = np.zeros((nruns, parameters["T"]))
    densities_R = np.zeros((nruns, parameters["T"]))


    # run the model nruns in parallel
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap_async(run_ABM, [(parameters.copy(), i, theshold) for i in range(nruns)])
        results = results.get()
    
    # unpack the results
    for i in range(nruns):
        ttp[i] = results[i][0]
        densities_S[i] = results[i][1]
        densities_R[i] = results[i][2]

    # save the densities in a file
    S_mean = np.mean(densities_S, axis=0)
    S_std = np.std(densities_S, axis=0)
    R_mean = np.mean(densities_R, axis=0)
    R_std = np.std(densities_R, axis=0)
    np.savez(f"{folder_name}_densities.npz", S_mean=S_mean, S_std=S_std, R_mean=R_mean, R_std=R_std,densities_S=densities_S,densities_R=densities_R)

    # plot the average density of the model with error bars
    plt.figure()
    plt.errorbar(np.arange(parameters["T"]), np.mean(densities_S, axis=0), yerr=np.std(densities_S, axis=0), label="S")
    plt.errorbar(np.arange(parameters["T"]), np.mean(densities_R, axis=0), yerr=np.std(densities_R, axis=0), label="R")
    # plot total density
    plt.plot(np.arange(parameters["T"]), np.mean(densities_S, axis=0) + np.mean(densities_R, axis=0), label="Total")
    # plot the threshold with a dashed line and transparency
    plt.axhline(y=theshold * (parameters['S0'] + parameters['R0']), color='r', linestyle='--', label="Threshold")
    plt.gca().collections[0].set_alpha(0.2)
    plt.gca().collections[1].set_alpha(0.2)
    plt.title("Average densities of S and R")
    plt.xlabel("Time")
    plt.ylabel("Density")
    plt.legend()
    # save plot in results_ABM folder
    plt.savefig(f"{folder_name}_densities_plot.png")
    plt.close()
    # print(f"Average time to progression of {folder_name}: ", np.mean(ttp), "±", np.std(ttp))
    return np.mean(ttp), np.std(ttp)

def run_test(parameters,test_name):
    imagename = f"results_ABM/{test_name}/initial_condition.png"
    model = ABM_model(parameters,False)
    plt.imsave(imagename, model.grid, cmap=model.get_cmap())


if __name__ == "__main__":

    # print("Starting the simulation...")

    # define the parameters of the model
    domain_size = 100
    parameters_ABM = {
        "domain_size" : domain_size,
        "T" : 2000,
        "dt" : 0.05,
        "S0" : 2000,
        "R0" : 20,
        "N0" : 0,
        "grS" : 0.023,
        "grR" : 0.023,
        "grN" : 0.005,
        "drS" : 0.01,
        "drR" : 0.01,
        "drN" : 0.00,
        "divrS" : 0.75,
        "divrN" : 0.5,
        "therapy" : "continuous",
        "initial_condition_type" : "uniform",
        "fill_factor":0.8,
        "core_locations": np.array([[domain_size//4,domain_size//4],[3*domain_size//4,3*domain_size//4]]),
        "save_locations" : False,
        "dimension" : 2,
        "seed" : 1}
    # NUMBER OF RUNS
    nruns = 10
    theshold = 1.25
    test_names = ["resistant_core","uniform"]
    # test_names = ["resistant_core","resistant_rim","multiple_resistant_cores","multiple_resistant_rims","uniform_ball_0.8","uniform_ball_0.9","uniform_ball_1.0","uniform"]
    # diffusions = [0.01,10]
    diffusions = [0.0]
    therapies = ["continuous","adaptive","notherapy"]
    time_to_progression = np.zeros((len(test_names), len(diffusions),len(therapies)))
    std_time_to_progression = np.zeros((len(test_names), len(diffusions),len(therapies)))
    param_dicts = [] # list of dictionaries with the parameters for each test
    # make parameter dictionaries and file structure
    start = time.perf_counter()
    overwrite = True
    for i,test_name in enumerate(test_names):
        try:
            os.mkdir(f"diffusion_results_ABM/{test_name}")
        except FileExistsError:
            if overwrite:
                shutil.rmtree(f"diffusion_results_ABM/{test_name}")
                os.mkdir(f"diffusion_results_ABM/{test_name}")
            else:
                raise FileExistsError(f"File diffusion_results_ABM/{test_name} already exists. Set overwrite to True to overwrite.")
        print("Starting test: ", test_name)
        print(f"{i/len(test_names)*100:.2f}% completed")
        current = time.perf_counter()
        print(f"Running for {(current-start)/60:.2f} minutes")
        for j,diffusion in enumerate(diffusions):
            print("Starting diffusion: ", diffusion)
            print(f"{i/len(test_names)*100+j/(len(diffusions)*len(test_names))*100:.2f}% completed")
            current = time.perf_counter()
            print(f"Running for {(current-start)/60:.2f} minutes")
            os.mkdir(f"diffusion_results_ABM/{test_name}/diffusion_{diffusion}")
            param_dict = parameters_ABM.copy()
            param_dict["diffusion_rate"] = diffusion
            # param_dict["dt"] = 0.01/diffusion
            # param_dict["T"] = int(parameters_ABM["T"]/param_dict["dt"])
            # param_dict["dt"] = 0.1/diffusion
            if "uniform_ball" in test_name:
                param_dict["initial_condition_type"] = "uniform_ball"
                param_dict["fill_factor"] = float(test_name.split("_")[2])
            else:
                param_dict["initial_condition_type"] = test_name
            param_dicts.append(param_dict)
            # pickle the parameter dictionary in a file
            pickle.dump(param_dict, open(f"diffusion_results_ABM/{test_name}/diffusion_{diffusion}/parameters.p", "wb"))
            for k,therapy in enumerate(therapies):
                print("Starting Therapy: ", therapy)
                param_dict["therapy"] = therapy
                mean,std = comparison_ABM(param_dict, nruns, theshold, f"diffusion_results_ABM/{test_name}/diffusion_{diffusion}/{therapy}")
                time_to_progression[i,j,k] = mean
                std_time_to_progression[i,j,k] = std
            df = pd.DataFrame(time_to_progression[:,j,:],index=test_names,columns=therapies)
            df.to_csv(f"diffusion_results_ABM/comparison_means_diffusion_{diffusion}.csv")
            df = pd.DataFrame(std_time_to_progression[:,j,:],index=test_names,columns=therapies)
            df.to_csv(f"diffusion_results_ABM/comparison_std_diffusion_{diffusion}.csv")

    # store the results in a file
    # for j,diffusion in enumerate(diffusions):
    #     df = pd.DataFrame(time_to_progression[:,j,:],index=test_names,columns=therapies)
    #     df.to_csv(f"results_ABM/comparison_means_diffusion_{diffusion}.csv")
    #     df = pd.DataFrame(std_time_to_progression[:,j,:],index=test_names,columns=therapies)
    #     df.to_csv(f"results_ABM/comparison_std_diffusion_{diffusion}.csv")


