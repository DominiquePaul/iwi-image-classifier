
"""
A script that demonstrates how to:
* read and display an image from a local file
* display an image that is stored in a numpy format

"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

# reading an image from disk
img1 = cv2.imread("/Users/dominiquepaul/xCoding/classification_tool/Data/apparel/apparel/RalphLauren_6797246895_10153191295236896_.jpg")
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.show()

# displaying an image from a numpy matrix
imagenet_imgs= np.load("./example_output_folder/imgnet_rocks_x.npy")
img2 = imagenet_imgs[0]
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.show()
