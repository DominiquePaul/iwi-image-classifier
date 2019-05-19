"""
A script that demonstrates how to:
* create a feature dataframe made out of the labels of the inception network



WARNING: This script can fail on very small datasets as a
"""

# these first imports are required to tell python where to look for packages
# in this case we link to the directory containing the preprocessing package
import sys
from os.path import dirname
sys.path.append(dirname("/Users/dominiquepaul/xCoding/classification_tool/Main/"))
sys.path.append(dirname("/Users/dominiquepaul/xCoding/classification_tool/Main/modules/"))

from regressionclass import Logistic_regression, Lasso_regression
from preprocessing import join_npy_data
from label_interpretation import load_industry_labels, create_feature_df


OBJECT_NAME = "apparel"

# load the csv containing the word associated with an object
industry_labels = load_industry_labels(file_path="../industry_dicts/selection_ApparelAccessoriesandLuxuryGoods.csv")

# load the data that was previously
file_paths=['./example_output_folder/apparel_image_package_train_val_split_0.npy']
x_train, y_train, x_val, y_val, conversion = join_npy_data(file_paths, training_data_only=False)

# we create as dataframe of features extracted from the inception network for the training data
x_train_df = create_feature_df(imgs=x_train, object_name=OBJECT_NAME, ind_labels=industry_labels, k_labels=10, basic_feats=True, wordnet_feats=True)

# train regression
log_reg = Logistic_regression()
log_reg.fit(x_train_df, y_train)
log_reg.find_best_thresh(x_train_df, y_train, optimize_for="accuracy", verbose=True)
# save the regression
log_reg.save_model("./example_output_folder/Logistic_regression_model.pkl")
print("Model saved")



# we create as dataframe of features extracted from the inception network for the validation data
x_val_df = create_feature_df(imgs=x_val, object_name=OBJECT_NAME, ind_labels=industry_labels, k_labels=10, basic_feats=True, wordnet_feats=True)

# making probability predictions on validation data
probality_predictions = log_reg.predict(x_val_df)
print(probality_predictions)

# making class predictions on validation data
class_predictions = log_reg.predict_classes(x_val_df)
print(class_predictions)
