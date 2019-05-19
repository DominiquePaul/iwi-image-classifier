"""
A script that demonstrates how to:
* load a previously saved logistic regression (/lasso) model
* create new predictions for data
* save the data to a nice output reformated

"""

# these first imports are required to tell python where to look for packages
# in this case we link to the directory containing the preprocessing package
import sys
from os.path import dirname
sys.path.append(dirname("/Users/dominiquepaul/xCoding/classification_tool/Main/"))
sys.path.append(dirname("/Users/dominiquepaul/xCoding/classification_tool/Main/modules/"))

from regressionclass import Logistic_regression, Lasso_regression
from label_interpretation import load_industry_labels, create_feature_df
import numpy as np


OBJECT_NAME = "apparel"
industry_labels = load_industry_labels(file_path="../industry_dicts/selection_ApparelAccessoriesandLuxuryGoods.csv")

# instantiate a new empty logistic regression model
log_reg2 = Logistic_regression()
# load the previously saved model from our disk
log_reg2.load_model("./example_output_folder/Logistic_regression_model.pkl")

new_images, names  = np.load("./example_output_folder/unlabelled_data_image_package_no_labels_0.npy")

# load the new data
x_test_df = create_feature_df(imgs=new_images,
                              object_name=OBJECT_NAME,
                              ind_labels=industry_labels,
                              k_labels=10,
                              basic_feats=True,
                              wordnet_feats=True)
predictions = log_reg2.predict_classes(x_test_df)



# again, the data is can be easily used to output the results to format that is nicer
# to read for humans or to be read into a database
df = pd.DataFrame(names[0])
# predict and print the classes for previously unseen images
df["predictions"] = predictions
# we save the results locally
df.to_csv("./example_output_folder/my_indirect_lab_classification_results.csv")

print("Results:")
print(df)
