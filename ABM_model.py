from matplotlib import gridspec
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class AMB_model:
    def __init__(self,parameters):
        self.domain_size = parameters["domain_size"]
        self.T = parameters["T"]
        self.S0 = parameters["S0"]
        self.R0 = parameters["R0"]
        self.N0 = parameters["N0"]
        self.domain_size0 = parameters["N0"]
        self.grS = parameters["grS"]
        self.grR = parameters["grR"]
        self.grN = parameters["grN"]
        self.drS = parameters["drS"]
        self.drR = parameters["drR"]
        self.drN = parameters["drN"]
        self.divrS = parameters["divrS"]
        self.divrN = parameters["divrN"]
        self.dimension = parameters["dimension"]
        self.grid = np.zeros((self.domain_size, self.domain_size))
        self.data = np.zeros((self.T, 4))
        self.current_therapy = 1
        # cell types 1 = sensitive, 2 = resistant, 3 = normal
        self.sensitive_type = 1
        self.resistant_type = 2
        self.normal_type = 3
        self.data[0, 0] = np.sum(self.grid == self.sensitive_type)
        self.data[0, 1] = np.sum(self.grid == self.resistant_type)
        self.data[0, 2] = np.sum(self.grid == self.normal_type)
        print(f"Starting with {self.S0} sensitive cells, {self.R0} resistant cells and {self.N0} normal cells.")
        self.data[0, 3] = self.current_therapy

        self.save_locations = parameters["save_locations"]
        if self.save_locations:
            self.location_data = []
            sensitive_location_data = np.append(
                np.argwhere(self.grid == self.sensitive_type),
                np.ones((self.data[0, 0].astype(int), 1)),
                axis=1,
            )
            resistant_location_data = np.append(
                np.argwhere(self.grid == self.resistant_type),
                np.ones((self.data[0, 1].astype(int), 1)) * 2,
                axis=1,
            )
            normal_location_data = np.append(
                np.argwhere(self.grid == self.normal_type),
                np.ones((self.data[0, 2].astype(int), 1)) * 3,
                axis=1,
            )
            initial_location_data = np.append(
                sensitive_location_data, resistant_location_data, axis=0
            )
            initial_location_data = np.append(
                initial_location_data, normal_location_data, axis=0
            )
            self.location_data.append(initial_location_data)

    # def reset_grid(self):
    #     self.grid = np.zeros((self.domain_size, self.domain_size))
    #     for location in self.location_data[0]:
    #         self.grid[location[0], location[1]] = location[2]

    def set_initial_condition(self, initial_condition_type):
        if initial_condition_type == "random":
            # select random 2D coordinates for S0 cells
            # randmoly set S0 grid points to 1[
            S0_idx = [
                divmod(i, self.domain_size) for i in random.sample(range(self.domain_size**2), self.S0)
            ]
            for idx in S0_idx:
                self.grid[idx] = 1
            # randmoly set R0 grid points to 2
            R0_idx = [
                divmod(i, self.domain_size) for i in random.sample(range(self.domain_size**2), self.R0)
            ]
            for idx in R0_idx:
                self.grid[idx] = 2
        elif initial_condition_type == "cluster":
            # make a ball of S0 cells
            for i in range(self.domain_size):
                for j in range(self.domain_size):
                    if (i - self.domain_size / 2) ** 2 + (j - self.domain_size / 2) ** 2 < self.S0:
                        self.grid[i, j] = 1
            # make a ball of R0 cells
            for i in range(self.domain_size):
                for j in range(self.domain_size):
                    if (i - self.domain_size / 2) ** 2 + (j - self.domain_size / 2) ** 2 < self.R0:
                        self.grid[i, j] = 2
        elif initial_condition_type == "cluster_in_normal":
            # make a ball of S0 cells
            for i in range(self.domain_size):
                for j in range(self.domain_size):
                    if (i - self.domain_size / 2) ** 2 + (j - self.domain_size / 2) ** 2 < self.S0:
                        self.grid[i, j] = self.sensitive_type
            # make a ball of R0 cells
            for i in range(self.domain_size):
                for j in range(self.domain_size):
                    if (i - self.domain_size / 2) ** 2 + (j - self.domain_size / 2) ** 2 < self.R0:
                        self.grid[i, j] = self.resistant_type
            # line walls with normal cells
            R = np.floor(((self.domain_size**2 - self.N0) / np.pi)**0.5)
            # fill in the grid with the value 1 for a circle of radius R around the center
            for i in range(self.domain_size):   
                for j in range(self.domain_size):
                    if (i - self.domain_size/2)**2 + (j - self.domain_size/2)**2 >= R**2:
                        self.grid[i, j] = self.normal_type           
            # randomly kill surplus cells so there are exactly N_cells cells using np.random.choice
            N_generated = np.sum(self.grid==self.normal_type)
            print(N_generated)
            cell_locations = np.argwhere(self.grid == self.normal_type)  
            kill_surplus = np.random.choice(cell_locations.shape[0], N_generated - self.N0, replace=False)
            self.grid[cell_locations[kill_surplus, 0], cell_locations[kill_surplus, 1]] = 0
            
        elif initial_condition_type == "two_clusters":
            # make a ball of S0 cells
            for i in range(self.domain_size):
                for j in range(self.domain_size):
                    if (i - self.domain_size / 4) ** 2 + (j - self.domain_size / 4) ** 2 < self.S0:
                        self.grid[i, j] = 1
            # make a ball of R0 cells
            for i in range(self.domain_size):
                for j in range(self.domain_size):
                    if (i - 3 * self.domain_size / 4) ** 2 + (j - 3 * self.domain_size / 4) ** 2 < self.R0:
                        self.grid[i, j] = 2
        elif initial_condition_type == "uniform":
            # uniformly distribution S0 + R0 + N0 cells using numpy
            self.flattened_indicies = np.argwhere(self.grid == 0).reshape(self.domain_size ** 2,2)
            self.N_total = self.S0 + self.R0 + self.N0
            N_grid = self.flattened_indicies.shape[0]
            self.location_indices = self.flattened_indicies[np.random.choice(np.linspace(0,N_grid-1,N_grid,dtype=int),replace=False,size=self.N_total)]
            for i in range(self.S0):
                self.grid[self.location_indices[i,0],self.location_indices[i,1]] = self.sensitive_type
            for i in range(self.S0,self.S0+self.R0):
                self.grid[self.location_indices[i,0],self.location_indices[i,1]] = self.resistant_type
            for i in range(self.S0+self.R0,self.N_total):
                self.grid[self.location_indices[i,0],self.location_indices[i,1]] = self.normal_type
        else:
            print("initial condition type not recognized")

        # # set up grid
        # grid = np.zeros((self.domain_size, self.domain_size))
        # # make a ball of S0 cells
        # for i in range(self.domain_size):
        #     for j in range(self.domain_size):
        #         if (i - self.domain_size / 2) ** 2 + (j - self.domain_size / 2) ** 2 < self.S0:
        #             grid[i, j] = 1
        # # make a ball of R0 cells
        # for i in range(self.domain_size):
        #     for j in range(self.domain_size):
        #         if (i - self.domain_size / 2) ** 2 + (j - self.domain_size / 2) ** 2 < self.R0:
        #             grid[i, j] = 2

        # # select random 2D coordinates for S0 cells
        # # randmoly set S0 grid points to 1[
        # S0_idx = [divmod(i, self.domain_size) for i in random.sample(range(self.domain_size ** 2), self.S0)]
        # for idx in S0_idx:
        #     grid[idx] = 1
        # # randmoly set R0 grid points to 2
        # R0_idx = [divmod(i, self.domain_size) for i in random.sample(range(self.domain_size ** 2), self.R0)]
        # for idx in R0_idx:
        #     grid[idx] = 2
        # # resample points for which grid[i] == 1
        # # while np.any(grid[R0_idx] == 1):
        #     # R0_idx = [divmod(i, self.domain_size) for i in random.sample(range(self.domain_size ** 2), self.R0)]
        # return gridspec

    def run(self, therapy_type):
        # run model for T iterations
        for t in range(1, self.T):
            # if t % 50 == 0:
            #     self.plot_grid()
            self.set_therapy(therapy_type, t)
            self.compute_death()
            self.compute_growth_S()
            self.compute_growth_R()
            self.compute_growth_N()

            # compute number of resistant and sensitive cells
            self.data[t, 0] = np.sum(self.grid == self.sensitive_type)
            self.data[t, 1] = np.sum(self.grid == self.resistant_type)
            self.data[t, 2] = np.sum(self.grid == self.normal_type)
            self.data[t, 3] = self.current_therapy

            if self.save_locations == True:
                sensitive_location_data = np.append(
                    np.argwhere(self.grid == self.sensitive_type),
                    np.ones((self.data[t, 0].astype(int), 1)) * self.sensitive_type,
                    axis=1,
                )
                resistant_location_data = np.append(
                    np.argwhere(self.grid == self.resistant_type),
                    np.ones((self.data[t, 1].astype(int), 1)) * self.resistant_type,
                    axis=1,
                )
                normal_location_data = np.append(
                    np.argwhere(self.grid == self.normal_type),
                    np.ones((self.data[t, 2].astype(int), 1)) * self.normal_type,
                    axis=1,
                )
                current_location_data = np.append(
                    sensitive_location_data, resistant_location_data, axis=0
                )
                current_location_data = np.append(
                    current_location_data, normal_location_data, axis=0
                )
                self.location_data.append(current_location_data)

    def compute_death(self):
        # compute death of cells
        # get all cells with S
        cells = np.argwhere(self.grid == self.sensitive_type)
        for cell in cells:
            if np.random.random() < self.drS:
                # cell dies
                self.grid[cell[0]][cell[1]] = 0
        cells = np.argwhere(self.grid == self.resistant_type)
        for cell in cells:
            if np.random.random() < self.drR:
                # cell dies
                self.grid[cell[0]][cell[1]] = 0
        cells = np.argwhere(self.grid == self.normal_type)
        for cell in cells:
            if np.random.random() < self.drN:
                # cell dies
                self.grid[cell[0]][cell[1]] = 0

    def compute_growth_S(self):
        # get all cells with S
        cells = np.argwhere(self.grid == self.sensitive_type)
        for cell in cells:
            # it grows with probability grS
            if np.random.random() < self.grS:
                # get all neigbours
                neigbours = self.get_neigbours(cell)
                count = 0
                for neigbour in neigbours:
                    if self.grid[neigbour[0]][neigbour[1]] == 0:
                        count += 1
                # check if a neigbour is empty
                if count > 0:
                    # check if therapy is succeful
                    if self.current_therapy and np.random.random() < self.divrS:
                        # cell dies during division
                        self.grid[cell[0]][cell[1]] = 0
                    else:
                        # shuffle neigbours
                        np.random.shuffle(neigbours)
                        # check one by one if they are empy
                        for neigbour in neigbours:
                            if self.grid[neigbour[0]][neigbour[1]] == 0:
                                # if empty, cell divides
                                self.grid[neigbour[0]][neigbour[1]] = 1
                                break

    def compute_growth_R(self):
        # get all cells with R
        cells = np.argwhere(self.grid == self.resistant_type)
        for cell in cells:
            # it grows with probability grR
            if np.random.random() < self.grR:
                # get all neigbours
                neigbours = self.get_neigbours(cell)
                count = 0
                for neigbour in neigbours:
                    if self.grid[neigbour[0]][neigbour[1]] == 0:
                        count += 1
                # check if a neigbour is empty
                if count > 0:
                    # shuffle neigbours
                    np.random.shuffle(neigbours)
                    # check one by one if they are empy
                    for neigbour in neigbours:
                        if self.grid[neigbour[0]][neigbour[1]] == 0:
                            # if empty, cell divides
                            self.grid[neigbour[0]][neigbour[1]] = 2
                            break

    def compute_growth_N(self):
        # get all cells with R
        cells = np.argwhere(self.grid == self.normal_type)
        for cell in cells:
            # it grows with probability grR
            if np.random.random() < self.grN:
                # get all neigbours
                neigbours = self.get_neigbours(cell)
                count = 0
                for neigbour in neigbours:
                    if self.grid[neigbour[0]][neigbour[1]] == 0:
                        count += 1
                # check if a neigbour is empty
                if count > 0:
                    if self.current_therapy and np.random.random() < self.divrN:
                        # cell dies during division
                        self.grid[cell[0]][cell[1]] = 0
                    else:
                        # shuffle neigbours
                        np.random.shuffle(neigbours)
                        # check one by one if they are empy
                        for neigbour in neigbours:
                            if self.grid[neigbour[0]][neigbour[1]] == 0:
                                # if empty, cell divides
                                self.grid[neigbour[0]][neigbour[1]] = 3
                                break

    def get_neigbours(self, cell):
        # get neigbours of cell
        neigbours = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i != 0 or j != 0:
                    neigbours.append([cell[0] + i, cell[1] + j])
        # check if neigbours are in the grid
        neigbours = [
            neigbour
            for neigbour in neigbours
            if neigbour[0] >= 0
            and neigbour[0] < self.domain_size
            and neigbour[1] >= 0
            and neigbour[1] < self.domain_size
        ]
        return neigbours

    def set_therapy(self, therapy_type, t):
        # set current therapy
        if therapy_type == "notherapy":
            self.current_therapy = 0
        elif therapy_type == "continuous":
            self.current_therapy = 0
        elif therapy_type == "adaptive":    
            n_sensitive = np.sum(self.grid == self.sensitive_type)
            n_resistant = np.sum(self.grid == self.resistant_type)
            n_normal = np.sum(self.grid == self.normal_type)
            total = n_sensitive + n_resistant
            initial_number = self.S0 + self.R0
            # total = np.sum(self.grid == self.sensitive_type) + np.sum(self.grid == self.resistant_type) + np.sum(self.grid == self.normal_type)
            # initial_number = self.S0 + self.R0 + self.N0
            if self.current_therapy and total < 0.5 * initial_number:
                self.current_therapy = 0
            elif not self.current_therapy and total > initial_number:
                self.current_therapy = 1
        else:
            raise ValueError("Therapy type not recognized")

    def get_data(self):
        # get data
        return self.data

    # PLOTTING FUNCTIONS
    def plot_grid(self):
        # plot the grid in different colors for S and R
        # make scatter plot
        # plot the grid in different colors for S and R
        # make scatter plot
        fig, ax = plt.subplots()
        sensitiveLocations = np.argwhere(self.grid == self.sensitive_type)
        resistantLocations = np.argwhere(self.grid == self.resistant_type)
        normalLocations = np.argwhere(self.grid == self.normal_type)
        scale = 20000 / self.domain_size**2
        sS = ax.scatter(
            sensitiveLocations[:, 0],
            sensitiveLocations[:, 1],
            c="b",
            marker="s",
            s=scale,
        )
        sR = ax.scatter(
            resistantLocations[:, 0],
            resistantLocations[:, 1],
            c="r",
            marker="s",
            s=scale,
        )
        sN = ax.scatter(
            normalLocations[:, 0],
            normalLocations[:, 1],
            c="g",
            marker="s",
            s=scale,
        )
        ax.set(xlim=(-0.5, self.domain_size + 0.5), ylim=(-0.5, self.domain_size + 0.5))
        ax.vlines(np.linspace(0, self.domain_size - 1, self.domain_size) - 0.5, 0, self.domain_size, linewidth=0.1)
        ax.hlines(np.linspace(0, self.domain_size - 1, self.domain_size) - 0.5, 0, self.domain_size, linewidth=0.1)
        ax.axis("equal")
        ax.axis("off")
        plt.show()


    def plot_celltypes_density(self, ax):
        # plot cell types density
        ax.plot(np.arange(1, self.T), self.data[1:, 0], label="S")
        ax.plot(np.arange(1, self.T), self.data[1:, 1], label="R")
        ax.plot(np.arange(1, self.T), self.data[1:, 2], label="N")
        ax.plot(np.arange(1, self.T), self.data[1:, 3] * 100, label="Therapy")
        ax.set_xlabel("Time")
        ax.set_ylabel("Density")
        ax.legend()
        return ax

    def animate_cells(self, figax):
        if np.all(self.data == 0):
            print("No Data!")
            return None, None, None
        fig, ax = figax
        nFrames = self.T - 1
        print(f"{self.location_data[1].shape=}")
        sensitiveLocations = self.location_data[1][self.location_data[1][:, 2] == 1, :2]
        resistantLocations = self.location_data[1][self.location_data[1][:, 2] == 2, :2]
        scale = 60000 / self.domain_size
        sS = ax.scatter(
            sensitiveLocations[:, 0],
            sensitiveLocations[:, 1],
            c="b",
            marker="s",
            s=scale,
        )
        sR = ax.scatter(
            resistantLocations[:, 0],
            resistantLocations[:, 1],
            c="r",
            marker="s",
            s=scale,
        )
        ax.set(xlim=(-0.5, self.domain_size + 0.5), ylim=(-0.5, self.domain_size + 0.5))
        ax.axis("equal")
        ax.axis("off")

        def update(i):
            sensitiveLocations = self.location_data[i + 1][
                self.location_data[i + 1][:, 2] == 1, :2
            ]
            resistantLocations = self.location_data[i + 1][
                self.location_data[i + 1][:, 2] == 2, :2
            ]
            sS.set_offsets(sensitiveLocations)
            sR.set_offsets(resistantLocations)

        anim = animation.FuncAnimation(
            fig=fig, func=update, frames=nFrames, interval=20
        )
        return fig, ax, anim

    def animate_graph(self, figax, interval=20):
        fig, ax = figax
        i = 2
        (lineS,) = ax.plot(np.arange(1, i), self.data[1:i, 0], label="S")
        (lineR,) = ax.plot(np.arange(1, i), self.data[1:i, 1], label="R")
        (lineN,) = ax.plot(np.arange(1, i), self.data[1:i, 2], label="N")
        (lineD,) = ax.plot(np.arange(1, i), self.data[1:i, 3] * 100, label="Therapy")
        ax.set_xlabel("Time")
        ax.set_ylabel("Density")
        ax.set(xlim=(0, self.T))
        ax.legend()

        def update(i):
            lineS.set_data(np.arange(1, i), self.data[1:i, 0])
            lineR.set_data(np.arange(1, i), self.data[1:i, 1])
            lineN.set_data(np.arange(1, i), self.data[1:i, 2])
            lineD.set_data(np.arange(1, i), self.data[1:i, 3] * 100)

        anim = animation.FuncAnimation(fig, update, self.T - 1, interval=interval)
        return fig, ax, anim

    def animate_cells_graph(self, interval=20):
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(10, 7)
        j = 2
        (lineS,) = ax[0].plot(np.arange(1, j), self.data[1:j, 0], label="S")
        (lineR,) = ax[0].plot(np.arange(1, j), self.data[1:j, 1], label="R")
        (lineN,) = ax[0].plot(np.arange(1, j), self.data[1:j, 2], label="N")
        (lineD,) = ax[0].plot(np.arange(1, j), self.data[1:j, 3] * 100, label="Therapy")
        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("Density")
        ax[0].set(xlim=(0, self.T))
        ax[0].legend()

        nFrames = self.T - 1
        print(f"{self.location_data[1].shape=}")
        sensitiveLocations = self.location_data[1][self.location_data[1][:, 2] == self.sensitive_type, :2]
        resistantLocations = self.location_data[1][self.location_data[1][:, 2] == self.resistant_type, :2]
        normalLocations = self.location_data[1][self.location_data[1][:, 2] == self.normal_type, :2]
        scale = 20000/self.domain_size**2
        # scale = 10
        sS = ax[1].scatter(
            sensitiveLocations[:, 0],
            sensitiveLocations[:, 1],
            c="b",
            marker="s",
            s=scale,
        )
        sR = ax[1].scatter(
            resistantLocations[:, 0],
            resistantLocations[:, 1],
            c="r",
            marker="s",
            s=scale,
        )
        sN = ax[1].scatter(
            normalLocations[:, 0],
            normalLocations[:, 1],
            c="g",
            marker="s",
            s=scale,
        )
        ax[1].set(xlim=(-0.5, self.domain_size + 0.5), ylim=(-0.5, self.domain_size + 0.5))
        ax[1].vlines(np.linspace(0, self.domain_size - 1, self.domain_size) - 0.5, 0, self.domain_size, linewidth=0.1)
        ax[1].hlines(np.linspace(0, self.domain_size - 1, self.domain_size) - 0.5, 0, self.domain_size, linewidth=0.1)
        ax[1].axis("equal")
        ax[1].axis("off")
        # ax[1].set(xlim=(70,130),ylim=(70,130))
        def update(i):
            lineS.set_data(np.arange(1, i), self.data[1:i, 0])
            lineR.set_data(np.arange(1, i), self.data[1:i, 1])
            lineN.set_data(np.arange(1, i), self.data[1:i, 2])
            lineD.set_data(np.arange(1, i), self.data[1:i, 3] * 100)
            sensitiveLocations = self.location_data[i + 1][
                self.location_data[i + 1][:, 2] == self.sensitive_type, :2
            ]
            resistantLocations = self.location_data[i + 1][
                self.location_data[i + 1][:, 2] == self.resistant_type, :2
            ]
            normalLocations = self.location_data[i + 1][
                self.location_data[i + 1][:, 2] == self.normal_type, :2
            ]
            sS.set_offsets(sensitiveLocations)
            sR.set_offsets(resistantLocations)
            sN.set_offsets(normalLocations)

        anim = animation.FuncAnimation(
            fig=fig, func=update, frames=nFrames, interval=interval
        )
        return fig, ax, anim

    def time_to_progression(self, threshold):
        # calculate inital tumor size
        initial_tumor_size = self.S0 + self.R0 + self.N0
        for i in range(self.T):
            total_number = np.sum(self.data[i, :3])
            print(total_number)
            if total_number > threshold * initial_tumor_size:
                return i


if __name__ == "__main__":

    # set up parameters
    parameters = {"domain_size" : 40,
    "T" : 400,
    "S0" : 200,
    "R0" : 10,
    "N0" : 300,
    "grS" : 0.028,
    "grR" : 0.023,
    "grN" : 0.005,
    "drS" : 0.013,
    "drR" : 0.013,
    "drN" : 0.00,
    "divrS" : 0.75,
    "divrN" : 0.5,
    "therapy" : "adaptive",
    "initial_condition_type" : "cluster_in_normal",
    "save_locations" : True,
    "dimension" : 2}

    # set up model
    model = AMB_model(parameters)
    # set up initial condition
    model.set_initial_condition(parameters["initial_condition_type"])
    # show grid of initial conditions
    model.plot_grid()
    # run simulation
    model.run(parameters["therapy"])

    # plot data
    fig, ax = plt.subplots(1, 1)
    ax = model.plot_celltypes_density(ax)
    plt.show()
    if model.save_locations:
        fig, ax, anim = model.animate_cells_graph()
        anim.save("media/nice_abm.mp4")

    # animate data
    # fig,ax = plt.subplots()
    # fig,ax,anim = model.animate_cells((fig,ax))
    # anim.save("test_ABM.mp4")

    # animate graph
    # fig,ax = plt.subplots()
    # fig,ax,anim = model.animate_graph((fig,ax))
    # anim.save("test_ABM_graph.mp4")

    # plt.show()
    # anim.save("both_working.mp4")

    # # do a parameter sweep
    # therapies = ['notherapy', 'continuous', 'adaptive']
    # initial_conditions_types = ['random', 'cluster', 'two_clusters']
    # S0s = [50, 100, 200]
    # for initial_condition_type in initial_conditions_types:
    #     for S0 in S0s:
    #             # set up model
    #         model = AMB_model(N, T, S0, R0, grS, grR, drS, drR, divrS)
    #             # set up initial condition
    #         model.set_initial_condition(initial_condition_type)
    #             # show grid of initial conditions
    #             # model.print_grid()
    #             # run simulation
    #         model.run(therapy)
    #             # # plot data
    #         fig, ax = plt.subplots(1, 1)
    #         ax = model.plot_celltypes_density(ax)
    #         ax.set_title('Therapy: {}, Initial condition: {}, S0: {}'.format(therapy, initial_condition_type, S0))
    #             # # save figure
    #         fig.savefig('elene_{}_{}_{}.png'.format(therapy, initial_condition_type, S0))
    #         plt.close(fig)
