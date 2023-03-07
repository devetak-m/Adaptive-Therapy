import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from neighbour_counting import count_neighbours,calculate_stats,bar_chart

# values bar chart
# load the data

def bar_chart(means,std):
    therapies = ["continuous", "adaptive", "notherapy"]
    names = means.values[:,0]
    mean_data = means.values[:,1:]
    std_data = std.values[:,1:]
    label_locations = np.arange(len(names))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0
    # sort by max adaptive
    sortkey = np.argsort(mean_data[:,1])[::-1]
    sorted_names = names[sortkey]
    print(sorted_names)
    colors = ["#bf0202","#0b2eb0","#00940c"]
    fig, ax = plt.subplots(constrained_layout=True)
    for i,therapy in enumerate(therapies):
        measurement = mean_data[:,i]
        sorted_measurement = measurement[sortkey]
        yerr = std_data[:,i]
        offset = width * multiplier
        rects = ax.barh(x + offset, sorted_measurement, width, xerr=yerr, label=therapy,color=colors[i])
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Time')
    ax.set_title('Time to progression')
    ax.set_yticks(x + width, sorted_names)
    ax.invert_yaxis()
    ax.legend()

def percentage_increase_bar_chart(means,std):
    names = np.array(means.index)
    continous_means = np.array(means.values[:,0])
    adaptive_means = np.array(means.values[:,1])
    percentage_increase = 100*(adaptive_means-continous_means)/continous_means 
    continous_std = np.array(std.values[:,1])
    adaptive_std = np.array(std.values[:,2])
    combined_var = np.array([i for i in (continous_std/continous_means)**2+(adaptive_std/adaptive_means)**2])
    percentage_std = 100*np.sqrt(combined_var)
    sortkey = np.argsort(percentage_increase)
    sorted_names = names[sortkey]
    sorted_percentage = percentage_increase[sortkey]
    sorted_std = percentage_std[sortkey]
    fig,ax = plt.subplots()
    bar = ax.barh(sorted_names,sorted_percentage,xerr=sorted_std)
    bar[list(sorted_names).index("ODE")].set_color("red")
    ax.set_xlabel("Percentage increase in adaptive TTP over continuous TTP")
    ax.vlines(0,-0.5,len(sorted_names)-0.5,linestyle="--",color="black")
    return ax


if __name__ == "__main__":
    foldername = "data/diffusion_results_ABM"
    means = pd.read_csv(f"{foldername}/comparison_means_diffusion_1.csv",index_col="initial_condition")
    std = pd.read_csv(f"{foldername}/comparison_std_diffusion_1.csv",index_col="initial_condition")
    ODE_means = [406,439,23]
    ODE_std = [0.0,0.0,0.0]
    means.loc["ODE"] = ODE_means
    std.loc["ODE"] = ODE_std
    # bar_chart(means,std)
    percentage_increase_bar_chart(means,std)
    plt.tight_layout()
    plt.savefig("shared_media/TTP_increase_with_diffusion_1.png")


    ##  for diffusion
    # mean_dfs = []
    # std_dfs = []
    # diffusions = [0.01,0.05,0.1,0.5,1]
    # mean_values = np.zeros((len(diffusions),8,3))
    # std_values = np.zeros((len(diffusions),8,3))
    # for i,diffusion in enumerate(diffusions):
    #     foldername = "data/diffusion_results_ABM"
    #     means = pd.read_csv(f"{foldername}/comparison_means_diffusion_{diffusion}.csv",index_col="initial_condition")
    #     std = pd.read_csv(f"{foldername}/comparison_std_diffusion_{diffusion}.csv",index_col="initial_condition")
    #     mean_values[i,:,:] = means.values
    #     std_values[i,:,:] = std.values
    # resistant_core_means = mean_values[:,0,:]
    # resistant_core_std = std_values[:,0,:]
    # resistant_core_names = [f"Diffusion = {diffusion}" for diffusion in diffusions]
    # resistant_core_means_df = pd.DataFrame(resistant_core_means,columns=["continuous","adaptive","notherapy"],index=resistant_core_names)
    # resistant_core_std_df = pd.DataFrame(resistant_core_std,columns=["continuous","adaptive","notherapy"],index=resistant_core_names)
    # ODE_means = [406,439,23]
    # ODE_std = [0.0,0.0,0.0]
    # resistant_core_means_df.loc["ODE"] = ODE_means
    # resistant_core_std_df.loc["ODE"] = ODE_std
    # # bar_chart(means,std)
    # ax = percentage_increase_bar_chart(resistant_core_means_df,resistant_core_std_df)
    # ax.set(title="Resistant Core TTP increase with Diffusion")
    # plt.tight_layout()
    # plt.savefig("shared_media/resistant_core_TTP_increase_with_diffusion.png")



