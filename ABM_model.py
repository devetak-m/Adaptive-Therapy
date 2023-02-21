from matplotlib import gridspec
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class ABM_model:
    def __init__(self,parameters):
        self.parameters = parameters
        self.domain_size = parameters["domain_size"]
        self.T = parameters["T"]
        self.dt = parameters["dt"]
        self.S0 = parameters["S0"]
        self.R0 = parameters["R0"]
        self.N0 = parameters["N0"]
        self.grS = parameters["grS"]
        self.grR = parameters["grR"]
        self.grN = parameters["grN"]
        self.drS = parameters["drS"]
        self.drR = parameters["drR"]
        self.drN = parameters["drN"]
        self.divrS = parameters["divrS"]
        self.divrN = parameters["divrN"]
        self.dimension = parameters["dimension"]
        self.data = np.zeros((self.T, 4))
        self.current_therapy = 1
        self.seed = parameters["seed"]
        print(f"Using seed {self.seed}")
        np.random.seed(self.seed)
        # cell types 1 = sensitive, 2 = resistant, 3 = normal
        self.sensitive_type = 1
        self.resistant_type = 2
        self.normal_type = 3
        # print(f"Starting with {self.S0} sensitive cells, {self.R0} resistant cells and {self.N0} normal cells.")

        # set initial condition and store data
        self.set_initial_condition(parameters["initial_condition_type"])
        self.data[0, 0] = np.sum(self.grid == self.sensitive_type)
        self.data[0, 1] = np.sum(self.grid == self.resistant_type)
        self.data[0, 2] = np.sum(self.grid == self.normal_type)
        self.data[0, 3] = self.current_therapy
        # inintialize grid

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

    def set_seed(self,seed):
        self.seed = seed
        np.random.seed(self.seed)

    def reset(self,hard = False):
        """Hard reset will reset the grid to a new initial grid of the same type. 
        Soft reset will reset the grid to the initial grid generated at the start."""
        if hard:
            self.set_initial_condition(self.parameters["initial_condition_type"])
        else:
            self.grid = self.initial_grid.copy()
        return self.seed

    def set_initial_condition(self, initial_condition_type):
        if self.dimension == 2:
            self.grid = np.zeros((self.domain_size, self.domain_size))
        elif self.dimension == 3:
            self.grid = np.zeros((self.domain_size, self.domain_size, self.domain_size))
        else:
            raise ValueError("dimension must be 2 or 3")
        if initial_condition_type == "random":
            # select random 2D coordinates for S0 cells
            # randmoly set S0 grid points to 1[
            S0_idx = [
                divmod(i, self.domain_size) for i in random.sample(range(self.domain_size**self.dimension), self.S0)
            ]
            for idx in S0_idx:
                self.grid[idx] = 1
            # randmoly set R0 grid points to 2
            R0_idx = [
                divmod(i, self.domain_size) for i in random.sample(range(self.domain_size**self.dimension), self.R0)
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
            R = np.floor(((self.domain_size**self.dimension - self.N0) / np.pi)**0.5)
            # fill in the grid with the value 1 for a circle of radius R around the center
            for i in range(self.domain_size):   
                for j in range(self.domain_size):
                    if (i - self.domain_size/2)**2 + (j - self.domain_size/2)**2 >= R**2:
                        self.grid[i, j] = self.normal_type           
            # randomly kill surplus cells so there are exactly N_cells cells using np.random.choice
            N_generated = np.sum(self.grid==self.normal_type)
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
            self.flattened_indicies = np.argwhere(self.grid == 0).reshape(self.domain_size**self.dimension,self.dimension)
            self.N_total = self.S0 + self.R0 + self.N0
            N_grid = self.flattened_indicies.shape[0]
            self.location_indices = self.flattened_indicies[np.random.choice(np.linspace(0,N_grid-1,N_grid,dtype=int),replace=False,size=self.N_total)]
            for i in range(self.S0):
                self.grid[self.location_indices[i,:]] = self.sensitive_type
            for i in range(self.S0,self.S0+self.R0):
                self.grid[self.location_indices[i,:]] = self.resistant_type
            for i in range(self.S0+self.R0,self.N_total):
                self.grid[self.location_indices[i,:]] = self.normal_type
        elif initial_condition_type == "uniform_3d":
            # uniformly distribution S0 + R0 + N0 cells using numpy
            self.flattened_indicies = np.argwhere(self.grid == 0).reshape(self.domain_size**self.dimension,self.dimension)
            self.N_total = self.S0 + self.R0 + self.N0
            N_grid = self.flattened_indicies.shape[0]
            self.location_indices = self.flattened_indicies[np.random.choice(np.linspace(0,N_grid-1,N_grid,dtype=int),replace=False,size=self.N_total)]
            for i in range(self.S0):
                self.grid[tuple(self.location_indices[i,:])] = self.sensitive_type
            for i in range(self.S0,self.S0+self.R0):
                self.grid[tuple(self.location_indices[i,:])] = self.resistant_type
            for i in range(self.S0+self.R0,self.N_total):
                self.grid[tuple(self.location_indices[i,:])] = self.normal_type
        else:
            print("initial condition type not recognized")

        # save initial grid
        self.initial_grid = self.grid.copy()

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
        # S0_idx = [divmod(i, self.domain_size) for i in random.sample(range(self.domain_size**self.dimension), self.S0)]
        # for idx in S0_idx:
        #     grid[idx] = 1
        # # randmoly set R0 grid points to 2
        # R0_idx = [divmod(i, self.domain_size) for i in random.sample(range(self.domain_size**self.dimension), self.R0)]
        # for idx in R0_idx:
        #     grid[idx] = 2
        # # resample points for which grid[i] == 1
        # # while np.any(grid[R0_idx] == 1):
        #     # R0_idx = [divmod(i, self.domain_size) for i in random.sample(range(self.domain_size**self.dimension), self.R0)]
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
            if np.random.random() < self.drS *self.dt:
                # cell dies
                self.grid[cell[0]][cell[1]] = 0
        cells = np.argwhere(self.grid == self.resistant_type)
        for cell in cells:
            if np.random.random() < self.drR *self.dt:
                # cell dies
                self.grid[cell[0]][cell[1]] = 0
        cells = np.argwhere(self.grid == self.normal_type)
        for cell in cells:
            if np.random.random() < self.drN *self.dt:
                # cell dies
                self.grid[cell[0]][cell[1]] = 0

    # TODO make one function for all three types
    def compute_growth_S(self):
        # get all cells with S
        cells = np.argwhere(self.grid == self.sensitive_type)
        for cell in cells:
            # it grows with probability grS
            if np.random.random() < self.grS*self.dt:
                # get all neighbours
                neighbours = self.get_neighbours(cell)
                count = 0
                for neighbour in neighbours:
                    if self.grid[tuple(neighbour)] == 0:
                        count += 1
                # check if a neighbour is empty
                if count > 0:
                    # check if therapy is succeful
                    if self.current_therapy and np.random.random() < self.divrS:
                        # cell dies during division
                        self.grid[cell[0]][cell[1]] = 0
                    else:
                        # shuffle neighbours
                        np.random.shuffle(neighbours)
                        # check one by one if they are empy
                        for neighbour in neighbours:
                            if self.grid[tuple(neighbour)] == 0:
                                # if empty, cell divides
                                self.grid[tuple(neighbour)] = 1
                                break

    def compute_growth_R(self):
        # get all cells with R
        cells = np.argwhere(self.grid == self.resistant_type)
        for cell in cells:
            # it grows with probability grR
            if np.random.random() < self.grR*self.dt:
                # get all neighbours
                neighbours = self.get_neighbours(cell)
                count = 0
                for neighbour in neighbours:
                    if self.grid[tuple(neighbour)] == 0:
                        count += 1
                # check if a neighbour is empty
                if count > 0:
                    # shuffle neighbours
                    np.random.shuffle(neighbours)
                    # check one by one if they are empy
                    for neighbour in neighbours:
                        if self.grid[tuple(neighbour)] == 0:
                            # if empty, cell divides
                            self.grid[tuple(neighbour)] = 2
                            break

    def compute_growth_N(self):
        # get all cells with R
        cells = np.argwhere(self.grid == self.normal_type)
        for cell in cells:
            # it grows with probability grR
            if np.random.random() < self.grN*self.dt:
                # get all neighbours
                neighbours = self.get_neighbours(cell)
                count = 0
                for neighbour in neighbours:
                    if self.grid[tuple(neighbour)] == 0:
                        count += 1
                # check if a neighbour is empty
                if count > 0:
                    if self.current_therapy and np.random.random() < self.divrN:
                        # cell dies during division
                        self.grid[cell[0]][cell[1]] = 0
                    else:
                        # shuffle neighbours
                        np.random.shuffle(neighbours)
                        # check one by one if they are empy
                        for neighbour in neighbours:
                            if self.grid[tuple(neighbour)] == 0:
                                # if empty, cell divides
                                self.grid[tuple(neighbour)] = 3
                                break

    def get_neighbours(self, cell):
        # get neighbours of cell
        if self.dimension == 2:
            neighbours = []
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i != 0 or j != 0:
                        neighbours.append([cell[0] + i, cell[1] + j])
            # check if neighbours are in the grid
            neighbours = [
                neighbour
                for neighbour in neighbours
                if neighbour[0] >= 0
                and neighbour[0] < self.domain_size
                and neighbour[1] >= 0
                and neighbour[1] < self.domain_size
            ]
            return neighbours
        elif self.dimension == 3:
            neighbours = []
            for i in range(-1, 2):
                for j in range(-1, 2):
                    for k in range(-1, 2):
                        if i != 0 or j != 0 or k != 0:
                            neighbours.append([cell[0] + i, cell[1] + j, cell[2] + k])
            # check if neighbours are in the grid
            neighbours = [
                neighbour
                for neighbour in neighbours
                if neighbour[0] >= 0
                and neighbour[0] < self.domain_size
                and neighbour[1] >= 0
                and neighbour[1] < self.domain_size
                and neighbour[2] >= 0
                and neighbour[2] < self.domain_size
            ]
            return neighbours

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
        return self.data

    # PLOTTING FUNCTIONS
    def plot_grid(self):
        fig, ax = plt.subplots()
        sensitiveLocations = np.argwhere(self.grid == self.sensitive_type)
        resistantLocations = np.argwhere(self.grid == self.resistant_type)
        normalLocations = np.argwhere(self.grid == self.normal_type)
        scale = 20000 / self.domain_size**self.dimension
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
        ax.plot(np.arange(1, self.T)*self.dt, self.data[1:, 0], label="S")
        ax.plot(np.arange(1, self.T)*self.dt, self.data[1:, 1], label="R")
        ax.plot(np.arange(1, self.T)*self.dt, self.data[1:, 2], label="N")
        ax.plot(np.arange(1, self.T)*self.dt, self.data[1:, 3] * 100, label="Therapy")
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

    def animate_cells_graph(self, interval=20, stride=1):
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
        # print(f"{self.location_data[1].shape=}")
        sensitiveLocations = self.location_data[1][self.location_data[1][:, 2] == self.sensitive_type, :2]
        resistantLocations = self.location_data[1][self.location_data[1][:, 2] == self.resistant_type, :2]
        normalLocations = self.location_data[1][self.location_data[1][:, 2] == self.normal_type, :2]
        scale = 20000/self.domain_size**self.dimension
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
        def update(j):
            i = j * stride
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
            fig=fig, func=update, frames=nFrames//stride, interval=interval
        )
        return fig, ax, anim

    # 3d Plotting 
    def plot_cells_3d(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        def explode(arr):
            size = np.array(arr.shape)*2
            arr_e = np.zeros(size - 1, dtype=arr.dtype)
            arr_e[::2, ::2, ::2] = arr
            return arr_e
        sensitive_voxels = np.array(self.grid==self.sensitive_type).astype(bool)
        sensitive_facecolors = np.where(sensitive_voxels, '#5692CD', '#7A88CCC0')
        sensitive_edgecolors = np.where(sensitive_voxels, '#000000', '#7D84A6')
        sensitive_filled = sensitive_voxels.copy()
        sensitive_facecolors_exp = explode(sensitive_facecolors)
        sensitive_edgecolors_exp = explode(sensitive_edgecolors)
        sensitive_filled_exp = explode(sensitive_filled)
        resistant_voxels = np.array(self.grid==self.resistant_type).astype(bool)
        resistant_facecolors = np.where(resistant_voxels, '#BF1818', '#7A88CCC0')
        resistant_edgecolors = np.where(resistant_voxels, '#000000', '#7D84A6')
        resistant_filled = resistant_voxels.copy()
        resistant_facecolors_exp = explode(resistant_facecolors)
        resistant_edgecolors_exp = explode(resistant_edgecolors)
        resistant_filled_exp = explode(resistant_filled)
        normal_voxels = np.array(self.grid==self.normal_type).astype(bool)
        normal_facecolors = np.where(normal_voxels, '#11A81F', '#7A88CCC0')
        normal_edgecolors = np.where(normal_voxels, '#000000', '#7D84A6')
        normal_filled = normal_voxels.copy()
        normal_facecolors_exp = explode(normal_facecolors)
        normal_edgecolors_exp = explode(normal_edgecolors)
        normal_filled_exp = explode(normal_filled)

        # generate mesh
        x, y, z = np.indices(np.array(sensitive_filled_exp.shape) + 1).astype(float) // 2
        x[0::2, :, :] += 0.05
        y[:, 0::2, :] += 0.05
        z[:, :, 0::2] += 0.05
        x[1::2, :, :] += 0.95
        y[:, 1::2, :] += 0.95
        z[:, :, 1::2] += 0.95
        ax.voxels(x,y,z,sensitive_filled_exp, facecolors=sensitive_facecolors_exp, edgecolors=sensitive_edgecolors_exp)
        ax.voxels(x,y,z,resistant_filled_exp, facecolors=resistant_facecolors_exp, edgecolors=resistant_edgecolors_exp)
        ax.voxels(x,y,z,normal_filled_exp, facecolors=normal_facecolors_exp, edgecolors=normal_edgecolors_exp)
        ax.set(xlim=(0, self.domain_size), ylim=(0, self.domain_size), zlim=(0, self.domain_size))  
        ax.axis("off")
        plt.show()




    def animate_cells_3d(self, interval=20, stride=1):
        pass

    def time_to_progression(self, threshold):
        # calculate inital tumor size
        initial_tumor_size = self.S0 + self.R0 + self.N0
        for i in range(self.T):
            total_number = np.sum(self.data[i, :3])
            if total_number > threshold * initial_tumor_size:
                return i
        return -1


if __name__ == "__main__":

    # set up parameters
    parameters = {"domain_size" : 10,
    "T" : 400,
    "dt" : 1,
    "S0" : 50,
    "R0" : 50,
    "N0" : 50,
    "grS" : 0.00,
    "grR" : 0.00,
    "grN" : 0.00,
    "drS" : 0.0,
    "drR" : 0.0,
    "drN" : 0.0,
    "divrS" : 0.75,
    "divrN" : 0.5,
    "therapy" : "adaptive",
    "initial_condition_type" : "uniform_3d",
    "save_locations" : True,
    "dimension" : 3,
    "seed" : 0}

    # set up model
    model = ABM_model(parameters)
    # set up initial condition
    model.set_initial_condition(parameters["initial_condition_type"])
    # show grid of initial conditions
    # model.plot_grid()
    model.plot_cells_3d()
    # run simulation
    # model.run(parameters["therapy"])

    # plot data
    fig, ax = plt.subplots(1, 1)
    ax = model.plot_celltypes_density(ax)
    t = np.arange(1, model.T)*model.dt
    # ax.plot(t,model.R0*np.pi * np.exp(-model.drS*t), label="ODE Model")
    plt.show()

    # if model.save_locations:
    #     fig, ax, anim = model.animate_cells_graph(stride=10,interval=80)
    #     anim.save("media/nice_abm.mp4")

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
    #         model = ABM_model(N, T, S0, R0, grS, grR, drS, drR, divrS)
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
