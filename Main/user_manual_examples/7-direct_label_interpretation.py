"""
A script that demonstrates how to:
* apply the direct identification method to new images without trainin
"""

# these first imports are required to tell python where to look for packages
# in this case we link to the directory containing the preprocessing package
import sys
from os.path import dirname
sys.path.append(dirname("/Users/dominiquepaul/xCoding/classification_tool/Main/"))

from preprocessing import join_npy_data
from label_interpretation import identify_items
import numpy as np
import pandas as pd

file_path = "./example_output_folder/unlabelled_data_image_package_no_labels_0.npy"
x_test, names = np.load(file_path)

words_we_are_checking_for = ["cars","truck"]
predictions = identify_items(x_test, words_we_are_checking_for, k_labels=10, use_synonyms=True)

# again, the data is can be easily used to output the results to format that is nicer
# to read for humans or to be read into a database
df = pd.DataFrame(names[0])
# predict and print the classes for previously unseen images
df["predictions"] = predictions
# we save the results locally
df.to_csv("./example_output_folder/my_direct_interpretation_classification_results.csv")

print("Results:")
print(df)
