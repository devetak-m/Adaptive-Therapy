import matplotlib.pyplot as plt
import numpy as np
import os

def reverse(filename):
    img = plt.imread(filename)
    img = np.array(img)
    print(img)
    reverse_img = 1 - img
    reverse_img[:,:,3] =1 
    reverse_img = reverse_img
    plt.imsave(filename.split(".png")[0] + "_reversed_.png",reverse_img)

image_names = [filename for filename in os.listdir("initial_conditions") if "0.2" in filename]
for image_name in image_names:
    reverse("initial_conditions/" + image_name)

