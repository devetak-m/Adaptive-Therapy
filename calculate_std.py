import numpy as np 
import matplotlib.pyplot as plt
import pickle
# load the data
def compute_ttps(data):
    # read in densities S
    densities_S = data["densities_S"]
    densities_R = data["densities_R"]
    # parameters = pickle.load(open(f"data/results_ABM_fr_0.01/{test_names[0]}/{therapies[0]}_parameters.p", "rb"))
    R0 = 2000
    S0 = 20
    threshold = 1.25
    total = densities_S + densities_R
    ttps = np.zeros(len(total))
    for i in range(len(total)):
        density_S = densities_S[i]
        density_R = densities_R[i]
        for j in range(len(density_S)):
            if density_S[j] + density_R[j] > threshold * (R0+S0):
                ttps[i] = j
                break
    return ttps

def plot_densities(data,ttp=None):
    densities_S = data["densities_S"]
    densities_R = data["densities_R"]
    R0,S0 = densities_S[0,0],densities_R[0,0]
    fig,ax = plt.subplots()
    ax.plot(np.mean(densities_S, axis=0))
    ax.plot(np.mean(densities_R, axis=0))
    ax.plot(np.mean(densities_R+densities_S, axis=0))
    ax.hlines((R0+S0)*1.25,0,1500,linestyles="dashed",linewidth=1)
    if ttp:
        ax.vlines(ttp,0,(R0+S0)*2,linestyles="dashed",linewidth=1)
    ax.set(xlabel="Time",ylabel="Density")
    ax.legend(["S","R"])
    plt.show()

def compute_increases(test_names):
    increase_means = np.zeros((len(test_names)))
    increase_stds = np.zeros((len(test_names)))
    for test_num,test_name in enumerate(test_names):
        data_adaptive = np.load(f"data/results_ABM_fr_0.01/{test_name}/adaptive_densities.npz")
        ttps_adaptive = compute_ttps(data_adaptive)
        data_continuous = np.load(f"data/results_ABM_fr_0.01/{test_name}/continuous_densities.npz")
        ttps_continuous = compute_ttps(data_continuous)
        plot_densities(data_adaptive,ttp= np.mean(ttps_adaptive))
        print("Test: ", test_name)
        print("Adaptive TTP: ", np.mean(ttps_adaptive))
        print("Continuous TTP: ", np.mean(ttps_continuous))
        increase = (ttps_adaptive-ttps_continuous)/100
        increase_means[test_num] = np.mean(increase)
        increase_stds[test_num] = np.std(increase)
    return increase_means,increase_stds
            
            
# return the time to progression
test_names = ["resistant_core","resistant_rim","multiple_resistant_cores","multiple_resistant_rims","uniform_ball_0.8","uniform_ball_0.9","uniform_ball_1"]
increase_means,increase_stds = compute_increases(test_names)
plt.barh(test_names,increase_means,xerr=increase_stds)
plt.show()
print(increase_means)
print(increase_stds)