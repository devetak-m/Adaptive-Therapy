import numpy as np 
import matplotlib.pyplot as plt

domain_size = 50
grid = np.zeros((domain_size, domain_size))
N_cells = 2000
R = np.floor((domain_size**2 - N_cells) / np.pi)**0.5
# fill in the grid with the value 1 for a circle of radius R around the center
for i in range(domain_size):   
    for j in range(domain_size):
        if (i - domain_size/2)**2 + (j - domain_size/2)**2 >= R**2:
            grid[i, j] = 1
# randomly kill surplus cells so there are exactly N_cells cells using np.random.choice
N_generated = np.sum(grid!=0)
cell_locations = np.argwhere(grid == 1)  
kill_surplus = np.random.choice(cell_locations.shape[0], N_generated - N_cells, replace=False)
grid[cell_locations[kill_surplus, 0], cell_locations[kill_surplus, 1]] = 0

print(np.sum(grid!=0))
plt.imshow(grid)
plt.show()

