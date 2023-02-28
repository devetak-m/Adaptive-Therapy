import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from ABM_model import ABM_model
import time
import os 
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
        folder_name = f"results_ABM/comparison_ABM_{parameters['therapy']}_initial_condition_{parameters['initial_condition_type']}"

    # print("Running time to progression for", folder_name, "...")

    # initialize the arrays that will store the statistics
    ttp = np.zeros(nruns)

    # initialize the arrays that will store the densities
    densities_S = np.zeros((nruns, int(parameters["T"] * (1/parameters["dt"]))))
    densities_R = np.zeros((nruns, int(parameters["T"] * (1/parameters["dt"]))))


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
    np.savez(f"{folder_name}_densities.npz", S_mean=S_mean, S_std=S_std, R_mean=R_mean, R_std=R_std)

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

if __name__ == "__main__":

    # print("Starting the simulation...")

    # define the parameters of the model
    domain_size = 400
    parameters_ABM = {
        # "domain_size" : domain_size,
        "domain_size" : 20,
        "T" : 2,
        "dt" : 1,
        # "S0" : 8000,
        # "R0" : 800,
        "S0" : 100,
        "R0" : 10,
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
    nruns = 1

    # define parameters to compare
    therapies = ["continuous", "adaptive", "notherapy"]
    n_tests = 8

    # define the threshold
    theshold = 1.5

    # time to progression for the different combinations of parameters
    ttp = np.zeros((n_tests, len(therapies)))
    # standard deviation of the time to progression for the different combinations of parameters
    std = np.zeros((n_tests, len(therapies)))

    # run the model for all the combinations of parameters
    start = time.perf_counter()
    test_num = 0
    initial_condition_types  = ["resistant_core","resistant_rim","multiple_resistant_cores","multiple_resistant_rims"]
    for initial_condition_type in initial_condition_types:
        # print("Running for initial condition type", initial_condition_type, "...")
        test_name = initial_condition_type
        os.mkdir(f"results_ABM/{test_name}")
        print(f"Starting test {test_num+1}/{n_tests}: {test_name}...")
        if test_num>=1:
            remaining = (time.perf_counter() - start) * (n_tests - test_num) / test_num
            print("Time Remaining: ", np.round(remaining//60,1), "minutes")
        imagename = f"results_ABM/{test_name}/initial_condition.png"
        model = ABM_model(parameters_ABM)
        plt.imsave(imagename,model.grid, cmap=model.get_cmap(),vmin=0,vmax=2)
        for therapy in therapies:
            # print("Therapy = ", therapy, "...")
            parameters_ABM["initial_condition_type"] = initial_condition_type
            parameters_ABM["therapy"] = therapy
            model = ABM_model(parameters_ABM)
            folder_name = f"results_ABM/{test_name}/{therapy}"
            mean, std_r = comparison_ABM(parameters_ABM, nruns, theshold,folder_name=folder_name)
            ttp[test_num, therapies.index(therapy)] = mean
            std[test_num, therapies.index(therapy)] = std_r
        test_num += 1
    # uniform_tests
    fill_factors = [0.8,0.9,1]
    parameters_ABM["initial_condition_type"] = "uniform_ball"
    for fill_factor in fill_factors:
        # print("Starting uniform ball test with fill factor", fill_factor, "...")
        test_name = f"uniform_ball_{fill_factor}"
        os.mkdir(f"results_ABM/{test_name}")
        remaining = (time.perf_counter() - start) * (n_tests - test_num) / test_num
        print("Time Remaining: ", np.round(remaining//60,1), "minutes")
        imagename = f"results_ABM/{test_name}/initial_condition.png"
        model = ABM_model(parameters_ABM)
        plt.imsave(imagename,model.grid, cmap=model.get_cmap(),vmin=0,vmax=2)
        for therapy in therapies:
            # print("Therapy = ", therapy, "...")
            parameters_ABM["fill_factor"] = fill_factor
            parameters_ABM["therapy"] = therapy
            model = ABM_model(parameters_ABM)
            folder_name = f"results_ABM/{test_name}/{therapy}"
            mean, std_r = comparison_ABM(parameters_ABM, nruns, theshold,folder_name=folder_name)
            ttp[test_num, therapies.index(therapy)] = mean
            std[test_num, therapies.index(therapy)] = std_r
        test_num += 1

    parameters_ABM["domain_size"] = 200
    parameters_ABM["initial_condition_type"] = "uniform"
    # print("Starting uniform test...")
    test_name = "uniform"
    os.mkdir(f"results_ABM/{test_name}")
    print(f"Starting test {test_num+1}/{n_tests}: {test_name}...")
    remaining = (time.perf_counter() - start) * (n_tests - test_num) / test_num
    print("Time Remaining: ", np.round(remaining//60,1), "minutes")
    imagename = f"results_ABM/{test_name}/initial_condition.png"
    model = ABM_model(parameters_ABM)
    plt.imsave(imagename,model.grid, cmap=model.get_cmap(),vmin=0,vmax=2)
    for therapy in therapies:
        # print("Therapy = ", therapy, "...")
        parameters_ABM["fill_factor"] = fill_factor
        parameters_ABM["therapy"] = therapy
        folder_name = f"results_ABM/{test_name}/{therapy}"
        mean, std_r = comparison_ABM(parameters_ABM, nruns, theshold,folder_name=folder_name)
        ttp[test_num, therapies.index(therapy)] = mean
        std[test_num, therapies.index(therapy)] = std_r


    # store the results in a file
    import pandas as pd 
    test_names = initial_condition_types + [f"uniform_ball_{fill_factor}" for fill_factor in fill_factors] + ["uniform"]
    df = pd.DataFrame(ttp,index=test_names,columns=therapies)
    df.to_csv(f"results_ABM/comparison_means.csv")
    df = pd.DataFrame(std,index=test_names,columns=therapies)
    df.to_csv(f"results_ABM/comparison_std.csv")


