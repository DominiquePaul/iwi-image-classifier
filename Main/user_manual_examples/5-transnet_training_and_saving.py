"""
A script that demonstrates how to:
* load a previously saved transfer network model
* make predictions on new data that in an evaluation ready format (names + predictions)
* save the results to the local disk
"""

# these first imports are required to tell python where to look for packages
# in this case we link to the directory containing the preprocessing package
import os
import sys
from os.path import dirname

# option 1 for loading other modules
sys.path.append(dirname("/Users/dominiquepaul/xCoding/classification_tool/Main/"))

### an alternative menthod for handling file paths ###
# main_path = os.path.dirname(__file__) # this doesnt work when run in a REPL environment
# module_path = os.path.join(main_path,"../" )
# sys.path.append(dirname(module_path))

from preprocessing import join_npy_data
from transfer_learning import Transfer_net

file_paths=["./example_output_folder/apparel_image_package_train_val_split_0.npy"]
x_train, y_train, x_test, y_test, conversion = join_npy_data(file_paths, training_data_only=False)

# instantiate the empty model
transfer_model = Transfer_net()
# we create our own model architecture by passing configuration values
transfer_model.create_network(layers=2, neurons=10, dropout_rate=0.6, num_output_classes=2)
# convert the data into transfer values
x_train = transfer_model.load_transfer_data(x_train)
# alternatively we could also cache the data
# x_train = transfer_model.load_or_cache_transfer_data(x_train, file_path="./example_output_folder/transfernet_cached_files")

# traing the model
transfer_model.train(x_train, y_train, learning_rate=1e-04, epochs=5, batch_size=8, verbose=True, tb_logs_dir="./example_output_folder/log_files")
# we save the trained model to the local disk so we can use it later
transfer_model.save_model("./example_output_folder/transfer_model_example.HDF5")



#
#
# t_net2 = Transfer_net()
# t_net2.load_model("./ello_test_trans.HDF5")
#
# preds = t_net2.predict_classes(x_test)
