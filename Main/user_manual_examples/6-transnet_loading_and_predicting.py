"""
A script that demonstrates how to:
* load a previously saved transfer network model
* make predictions on new data that in an evaluation ready format (names + predictions)
* save our results to the local disk
"""

# these first imports are required to tell python where to look for packages
# in this case we link to the directory containing the preprocessing package
import sys
from os.path import dirname
sys.path.append(dirname("/Users/dominiquepaul/xCoding/classification_tool/Main/"))


from cnn import cnn_model
import numpy as np
import pandas as pd

# load the new data
new_images, names  = np.load("./example_output_folder/unlabelled_data_image_package_no_labels_0.npy")

# instantiate a new empty cnn model
transfer_model_new = cnn_model()
# load the previously saved model from our disk
transfer_model_new.load_model(file_path="./example_output_folder/saved_cnn_model.HDF5")

# again, the data is can be easily used to output the results to format that is nicer
# to read for humans or to be read into a database
df = pd.DataFrame(names[0])
# predict and print the classes for previously unseen images
df["predictions"] = transfer_model_new.predict_classes(new_images)
# we save the results locally
df.to_csv("./example_output_folder/my_transfernet_classification_results.csv")

print("Results:")
print(df)
