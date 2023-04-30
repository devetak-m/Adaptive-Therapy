import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

# load the data
means = pd.read_csv("results_ABM_computed/comparison_means.csv")
std = pd.read_csv("results_ABM_computed/comparison_std.csv")
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

plt.show()