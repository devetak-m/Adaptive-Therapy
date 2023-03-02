import numpy as np
import numpy.linalg as lag
import matplotlib.pyplot as plt
from PIL import Image

def uniform_balls(space_points,radius,S0,R0):
    resolution = 1000
    n_cells = R0+S0
    big_grid = np.zeros((resolution,resolution))
    scaled_radius = resolution*radius/space_points
    center = np.array([500,500])
    for i in range(resolution):
        for j in range(resolution):
            if (i-center[0])**2+(j-center[1])**2<scaled_radius**2:
                big_grid[i,j] = 255
    image = Image.fromarray(big_grid)
    circle = np.array(image.resize((space_points,space_points)))/255
    resistant_cells = circle*R0/n_cells
    sensitive_cells = circle*S0/n_cells
    return resistant_cells,sensitive_cells
radius = 5
R0,S0 = 10,1
max = (R0+S0)/50
r,s = uniform_balls(20,radius,R0,S0)
np.save(f"pde_initial_conditions/uniform_radius_{radius}_resistant.npy",r)
np.save(f"pde_initial_conditions/uniform_radius_{radius}_sensitive.npy",s)
fig,ax = plt.subplots(1,2)
ax[0].imshow(r,vmin=0,vmax=max)
ax[1].imshow(s,vmin=0,vmax=max)
plt.show()


