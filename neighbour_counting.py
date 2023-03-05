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
    RS = np.zeros(grid.shape)
    RR = np.zeros(grid.shape)
    SE = np.zeros(grid.shape)
    RE = np.zeros(grid.shape)
    # Loop over the grid
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            # Count the number of neighbours
            surrounding = grid[max(0,i-1):min(grid.shape[0],i+2),max(0,j-1):min(grid.shape[1],j+2)]
            n_surrounding = np.sum(surrounding!=-1)-1
            if grid[i,j] == 1:
                SS[i,j] = (np.sum(surrounding==1)-1)/n_surrounding
                SR[i,j] = np.sum(surrounding==2)/n_surrounding
                SE[i,j] = np.sum(surrounding==0)/n_surrounding
            elif grid[i,j] == 2:
                RS[i,j] = np.sum(surrounding==1)/n_surrounding
                RR[i,j] = (np.sum(surrounding==2)-1)/n_surrounding
                RE[i,j] = np.sum(surrounding==0)/n_surrounding
    results = [SS,SR,SE,RS,RR,RE]
    return results

def calculate_stats(results):
    SS_stat = np.sum(results[0])/parameters["S0"]
    SR_stat = np.sum(results[1])/(parameters["S0"])
    SE_stat = np.sum(results[2])/parameters["S0"]
    RS_stat = np.sum(results[3])/(parameters["R0"])
    RR_stat = np.sum(results[4])/parameters["R0"]
    RE_stat = np.sum(results[5])/parameters["R0"]
    return [SS_stat,SR_stat,SE_stat,RS_stat,RR_stat,RE_stat]

def bar_chart(grid,stats,ax = None):
    """Plots a bar chart of the stats."""
    # Create a bar chart
    if ax == None:
        fig,ax = plt.subplots()
    ax.bar(["SS","SR","SE","RS","RR","RE"],stats)
    ax.set_ylabel("Number of neighbours")
    ax.set_xlabel("Neighbour type")
    # ax.set(ylim=(0,5e-3))
    return ax

if __name__ == "__main__":
    # Set up the model
    parameters["fill_factor"] = 1
    parameters["domain_size"] = 100
    parameters["S0"] = 2000
    parameters["R0"] = 200
    parameters["seed"] = 1
    initial_condition_types = ["resistant_rim","resistant_core","multiple_resistant_cores","multiple_resistant_rims","uniform_ball","uniform"]
    for initial_condition_type in initial_condition_types:
        parameters["initial_condition_type"] = initial_condition_type
        model = ABM_model(parameters)
        results = count_neighbours(model.grid)
        stats = calculate_stats(results)
        print("S sum:",stats[0]+stats[1]+stats[2])
        print("R sum:",stats[3]+stats[4]+stats[5])
        ax = bar_chart(model.grid,stats)
        ax.set(title=f"{initial_condition_type} Neighbours")
        plt.savefig(f"shared_media/{initial_condition_type}_neighbours.png")
        print("Ratio of exposed sensitive to exposed resistant:",stats[2]/stats[5])

    # # plotting the results
    # fig,ax = plt.subplots(1,2)
    # ax[0].imshow(model.grid,cmap=model.get_cmap())
    # ax[0].axis("off")
    # print(results[4].shape)
    # neighbours = ax[1].imshow(results[2],cmap="bone")
    # ax[1].axis("off")
    # fig.colorbar(neighbours,ax=ax[1])
    # plt.show()