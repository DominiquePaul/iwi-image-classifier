"""
Script that demonstrates how to download images from wordnet
"""

# these first imports are required to tell python where to look for packages
# in this case we link to the directory containing the preprocessing package
import sys
from os.path import dirname
sys.path.append(dirname("../."))

import numpy as np
from preprocessing import create_imagenet_dataset, create_imagenet_dataset_random

# the synset ID tells the module which images to search for
# Other synset IDs can be found here: http://www.image-net.org/explore?wnid=n09416076
# the number following the 'wnid=' part in the link is the synset id
synset_id = "n09416076"
object_name = "rocks"

# this function collects images associated with the synset_id stated
imgs_object = create_imagenet_dataset(synset_id=synset_id, size=20, use_child_synsets=True)
# this function collects random images as 'negative' examples in the dataset
imgs_random = create_imagenet_dataset_random(size=20, max_synset_imgs=2, forbidden_synset=synset_id, exclude_synset_children=True)

# theoretically we could already stop here and save the images
# however, to create a dataset, we want to mix both image types and add labels
# adding labels can be done easily by creating a long array of either 0s or 1s
imgnet_imgs = np.concatenate((imgs_object, imgs_random))
imgnet_labels = np.array([1]*len(imgs_object) + [0]*len(imgs_random))

# to improve training later on we shuffle the dataset. However, we must ensure
# that the images and the labels are shuffled in the same way
random_order = np.random.permutation(len(imgnet_imgs))
imgnet_imgs = imgnet_imgs[random_order]
imgnet_labels = imgnet_labels[random_order]

# we can now save our finished dataset (theoretically we could also create one
#    file that includes image and label data. This is a personal preference)
np.save("./example_output_folder/imgnet_{}_x".format(object_name), imgnet_imgs)
np.save("./example_output_folder/imgnet_{}_y".format(object_name), imgnet_labels)
print("Finished processing the {} dataset".format(object_name))
