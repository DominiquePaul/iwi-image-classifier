"""
A script that demonstrates how to:
* load a previously saved dataset
* instantiate and train a custom neural network
* save a cnn model to the local disk
* make probability and class predictions
"""

# these first imports are required to tell python where to look for packages
# in this case we link to the directory containing the preprocessing package
import sys
from os.path import dirname
sys.path.append(dirname("../."))

from preprocessing import join_npy_data
from cnn import cnn_model
import numpy as np

# load the data that was previously
file_paths=['./example_output_folder/apparel_image_package_train_val_split_0.npy']
x_train, y_train, x_val, y_val, conversion = join_npy_data(file_paths, training_data_only=False)

# as there is only one file to read we could also just use:
file_path='./example_output_folder/apparel_image_package_train_val_split_0.npy'
x_train, y_train, x_val, y_val, conversion = np.load(file_path)

# instantiate the empty cnn model
m1 = cnn_model()

# define the general parameters of the model
config_file = {
    "conv_layers": 2,
    "conv_filters": 32,
    "dense_layers": 5,
    "dense_neurons": 10,
    "dropout_rate_dense": 0.2,
    "learning_rate": 1e-04}

# construct a new model and the configuration file and data to be usef for training
m1.new_model(x_data=x_train, y_data=y_train, config=config_file)
# start training and state where to save the logs
m1.train(epochs=2, batch_size=8, on_tpu=None, tb_logs_dir="./example_output_folder/log_files", verbose=True)
# save the trained model with its parameters
m1.save_model(file="./example_output_folder/saved_cnn_model.HDF5")

# making probability predictions on validation data
probality_predictions = m1.predict(x_val)
print(probality_predictions)

# making class predictions on validation data
class_predictions = m1.predict_classes(x_val)
print(class_predictions)
