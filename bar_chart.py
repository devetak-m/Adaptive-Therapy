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
    x = np.arange(len(names))  # the label locations
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
    names = means.values[:,0]
    continous_means = np.array(means.values[:,1])
    adaptive_means = np.array(means.values[:,2])
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
    ax.barh(sorted_names,sorted_percentage,xerr=sorted_std)
    ax.set_xlabel("Percentage increase in adaptive TTP over continuous TTP")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    means = pd.read_csv("results_ABM_computed/comparison_means.csv")
    std = pd.read_csv("results_ABM_computed/comparison_std.csv")
    # bar_chart(means,std)
    # todo investigate ordering by ratio of SE to RE
    percentage_increase_bar_chart(means,std)
    plt.show()


