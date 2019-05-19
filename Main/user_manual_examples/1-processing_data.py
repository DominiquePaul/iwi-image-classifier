"""
A script that shows how to:
1. Load data from local folders
2. Save processed images to a local folder (without labels)
3. How to read in the labels for image files from a json file (dataturk formatting)
4. How to align labels with the images to create a dataset ready for training
5. How to save a labelled dataset and apply additional configurations (augmentation, balancing)
"""

# these first imports are required to tell python where to look for packages
# in this case we link to the directory containing the preprocessing package
import sys
from os.path import dirname
sys.path.append(dirname("../."))

# the module and the respectiver functions are imported
import preprocessing
from preprocessing import load_images, read_label_json, return_labelled_images, save_to_numpy_with_labels, save_to_numpy, analyse_line_dataturk_format

# this global value defines the size the image is reformated to (square)
preprocessing.resize_to_size = 48
# this global value tells the program whether to skew or pad the images
preprocessing.allow_skewing = True

# all folders to search for image files
image_folders = ["./example_data/folder_with_images_1",
                 "./example_data/folder_with_images_2"]

# the folder folder that contains all relevant json files
json_folder = "./example_data/folder_with_json_files/"

# this is folder where we want to save the
target_output_folder = "./example_output_folder/"

# loading the images and their names from the designated folders
names_raw, images = load_images(folders=image_folders)

# theoretically we could save the images directly (e.g. to use an approach that does not require training)
save_to_numpy(folder_path=target_output_folder,
            img_names=names_raw["file_name"],
            files=images,
            object="unlabelled_data")

# reading in the image labels with a specified parsing function
labels = read_label_json(folder=json_folder,
                         line_interpreter_fn=analyse_line_dataturk_format)

# merging the labels and images together
names,files = return_labelled_images(labels=labels,
                                     file_names=names_raw,
                                     files=images)

# this is for savign the files with labels merged on them
save_to_numpy_with_labels(folder_path=target_output_folder,
                        files=files,
                        labels=names["label"],
                        object="apparel",
                        balance_dataset=True,
                        augment_training_data=False,
                        train_val_test_split=True,
                        file_names=names["file_name"])
