import matplotlib.pyplot as plt
import numpy as np

# domain_size = 20
# random_cubes = np.random.uniform((0,0,0),(domain_size,domain_size,domain_size),size=(100,3))
# random_cubes = np.round(random_cubes).astype(int)
# random_cubes = np.unique(random_cubes,axis=0)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # ax.scatter(random_cubes[:,0],random_cubes[:,1],random_cubes[:,2])
# # plot using voxels 
# ax.voxels(random_cubes[:,0],random_cubes[:,1],random_cubes[:,2],edgecolor='k',filled=True)
# plt.show()


# numpy logo

def explode(data):
    size = np.array(data.shape)*2
    data_e = np.zeros(size - 1, dtype=data.dtype)
    data_e[::2, ::2, ::2] = data
    return data_e

# build up the numpy logo
# n_voxels = np.zeros((4, 3, 4), dtype=bool)
# n_voxels[0, 0, :] = True
# n_voxels[-1, 0, :] = True
# n_voxels[1, 0, 2] = True
# n_voxels[2, 0, 1] = True

# create a random set of cubes
n_voxels = np.random.randint(0,2,size=(4,3,4)).astype(bool)
facecolors = np.where(n_voxels, '#FFD65DC0', '#7A88CCC0')
edgecolors = np.where(n_voxels, '#BFAB6E', '#7D84A6')
# filled = np.ones(n_voxels.shape)
filled = n_voxels

# upscale the above voxel image, leaving gaps
filled_2 = explode(filled)
fcolors_2 = explode(facecolors)
ecolors_2 = explode(edgecolors)

# Shrink the gaps
x, y, z = np.indices(np.array(filled_2.shape) + 1).astype(float) // 2
x[0::2, :, :] += 0.05
y[:, 0::2, :] += 0.05
z[:, :, 0::2] += 0.05
x[1::2, :, :] += 0.95
y[:, 1::2, :] += 0.95
z[:, :, 1::2] += 0.95

ax = plt.figure().add_subplot(projection='3d')
ax.voxels(x, y, z, filled_2, facecolors=fcolors_2, edgecolors=ecolors_2)
plt.show()