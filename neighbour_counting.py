import numpy as np
import matplotlib.pyplot as plt
from ABM_model import ABM_model
from default_parameters import parameters

def count_neighbours(grid):
    """Counts the number of sensitive-sensitive, resistant-resistant, 
    and sensitive-resistant neighbours for each cell in the grid."""
    # Create a grid of zeros to store the number of neighbours
    SS = np.zeros(grid.shape)
    SR = np.zeros(grid.shape)
    RR = np.zeros(grid.shape)
    SE = np.zeros(grid.shape)
    RE = np.zeros(grid.shape)
    # Loop over the grid
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            # Count the number of neighbours
            surrounding = grid[max(0,i-1):min(grid.shape[0],i+2),max(0,j-1):min(grid.shape[1],j+2)]
            if grid[i,j] == 1:
                SS[i,j] = np.sum(surrounding==1)-1
                SR[i,j] = np.sum(surrounding==2)
                SE[i,j] = np.sum(surrounding==0)
            elif grid[i,j] == 2:
                RR[i,j] = np.sum(surrounding==2)-1
                RE[i,j] = np.sum(surrounding==0)
    results = [SS,SR,RR,SE,RE]
    return results

def calculate_stats(results):
    SS_stat = np.sum(results[0])/parameters["S0"]
    SR_stat = np.sum(results[1])/(parameters["S0"])
    RR_stat = np.sum(results[2])/parameters["R0"]
    SE_stat = np.sum(results[3])/parameters["S0"]
    RE_stat = np.sum(results[4])/parameters["R0"]
    return [SS_stat,SR_stat,RR_stat,SE_stat,RE_stat]

def bar_chart(grid,stats,ax = None):
    """Plots a bar chart of the stats."""
    # Create a bar chart
    if ax == None:
        fig,ax = plt.subplots()
    ax.bar(["SS","SR","RR","SE","RE"],stats)
    ax.set_ylabel("Number of neighbours")
    ax.set_xlabel("Neighbour type")
    # ax.set(ylim=(0,5e-3))
    return ax

if __name__ == "__main__":
    # Set up the model
    parameters["initial_condition_type"] = "resistant_core"
    parameters["fill_factor"] = 1
    parameters["domain_size"] = 100
    parameters["S0"] = 2000
    parameters["R0"] = 200
    parameters["seed"] = 1
    model = ABM_model(parameters)
    results = count_neighbours(model.grid)
    stats = calculate_stats(results)
    print("S sum:",stats[0]+stats[1]+stats[3])
    print("R sum:",stats[2]+stats[1]*parameters["S0"]/parameters["R0"]+stats[4])
    bar_chart(model.grid,stats)
    print("Ratio of exposed sensitive to exposed resistant:",stats[3]/stats[4])

    # plotting the results
    fig,ax = plt.subplots(1,2)
    ax[0].imshow(model.grid,cmap=model.get_cmap())
    ax[0].axis("off")
    print(results[3].shape)
    neighbours = ax[1].imshow(results[0],cmap="bone")
    ax[1].axis("off")
    fig.colorbar(neighbours,ax=ax[1])
    plt.show()