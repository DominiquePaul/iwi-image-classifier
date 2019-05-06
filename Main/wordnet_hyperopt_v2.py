import csv
import sys
import numpy as np
from tqdm import tqdm
from os.path import dirname
from sklearn import metrics
import pandas as pd

from preprocessing import join_npy_data
from wordnet import load_industry_labels, create_feature_df

sys.path.append(dirname("/Users/dominiquepaul/xBachelorArbeit/Spring19/Bachelor-arbeit/Main/modules/"))
from regressionclass import Logistic_regression, Lasso_regression


# folder where different label evaluations are saved
FOLDER_PATH_SAVE = "/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/wnet_hyperopt_datasets"
# path with the industry dict folders
ind_labels = load_industry_labels(file_path="./industry_dicts/selection_AutomobileManufacturers.csv")

OBJECT = "car"
automotive_pckgs = ["/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/np_files/car_image_package_train_test_split0.npy"]
x_train, y_train, x_test, y_test, conversion = join_npy_data(automotive_pckgs)

n_label_list = [3, 5, 8, 10, 15, 20, 25, 50]

# transform or load the data if necessary
# for label_amount in tqdm(n_label_list):
#     x_train_df = create_feature_df(imgs=x_train, object_name=OBJECT, ind_labels=ind_labels, k_labels=label_amount)
#     x_test_df = create_feature_df(imgs=x_test, object_name=OBJECT, ind_labels=ind_labels, k_labels=label_amount)
#     x_train_df.to_csv(FOLDER_PATH_SAVE + "/train_{}".format(label_amount))
#     x_test_df.to_csv(FOLDER_PATH_SAVE + "/test_{}".format(label_amount))

basic_feats = ["max_score", "product_ref_count", "product_ref_sum", "product_reference"]
wordnet_feats = ["product_count_wordnet", "maxscorevalue_wordnet", "product_sum_wordnet"]

out_file = 'out_files/wordnet_hyperopt_v2_out.csv'
with open(out_file, 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['approach_name','label_amount','train_acc', 'train_f1', 'test_acc','test_f1', 'TP', 'FP', 'FN', 'TN'])


for label_amount in tqdm(n_label_list):
    # load the prepared datasets
    x_train_df = pd.read_csv(FOLDER_PATH_SAVE + "/train_{}".format(label_amount))
    x_test_df = pd.read_csv(FOLDER_PATH_SAVE + "/test_{}".format(label_amount))

    ### approach 1: regression with basic features
    x_train_arr = np.array(x_train_df.loc[:,basic_feats])
    x_test_arr = np.array(x_test_df.loc[:,basic_feats])
    # train regression
    lr1 = Logistic_regression()
    lr1.fit(x_train_arr, y_train)
    lr1.find_best_thresh(x_train_arr, y_train, optimize_for="f1", verbose=True)
    # evalaute
    train_acc, train_f1,_ = lr1.evaluate(x_train_arr, y_train)
    test_acc, test_f1, conf_m = lr1.evaluate(x_test_arr, y_test)
    with open(out_file, 'a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Regression_basic",label_amount, train_acc, train_f1, test_acc, test_f1, conf_m[0], conf_m[1], conf_m[2], conf_m[3]])


    ### approach 2: regression with wordnet features
    x_train_arr2 = np.array(x_train_df.loc[:,wordnet_feats])
    x_test_arr2 = np.array(x_test_df.loc[:,wordnet_feats])
    # train regression
    lr2 = Logistic_regression()
    lr2.fit(x_train_arr2, y_train)
    lr2.find_best_thresh(x_train_arr2, y_train, optimize_for="f1", verbose=True)
    # evalaute
    train_acc, train_f1,_ = lr2.evaluate(x_train_arr2, y_train)
    test_acc, test_f1, conf_m = lr2.evaluate(x_test_arr2, y_test)
    with open(out_file, 'a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Regression_wordnet",label_amount, train_acc, train_f1, test_acc, test_f1, conf_m[0], conf_m[1], conf_m[2], conf_m[3]])


    ### approach 3: using all features
    x_train_arr3 = np.array(x_train_df)
    x_test_arr3 = np.array(x_test_df)
    # train regression
    lr3 = Logistic_regression()
    lr3.fit(x_train_arr3, y_train)
    lr3.find_best_thresh(x_train_arr3, y_train, optimize_for="f1", verbose=True)
    # evalaute
    train_acc, train_f1,_ = lr3.evaluate(x_train_arr3, y_train)
    test_acc, test_f1, conf_m = lr3.evaluate(x_test_arr3, y_test)
    with open(out_file, 'a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Regression_all_feats",label_amount, train_acc, train_f1, test_acc, test_f1, conf_m[0], conf_m[1], conf_m[2], conf_m[3]])


    ### approach 4: train regression
    lasso = Lasso_regression()
    lasso.fit(x_train_arr3, y_train)
    lasso.find_best_thresh(x_train_arr3, y_train, optimize_for="f1", verbose=True)
    # evalaute
    train_acc, train_f1,_ = lasso.evaluate(x_train_arr3, y_train)
    test_acc, test_f1, conf_m = lasso.evaluate(x_test_arr3, y_test)
    with open(out_file, 'a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Lasso_regression",label_amount, train_acc, train_f1, test_acc, test_f1, conf_m[0], conf_m[1], conf_m[2], conf_m[3]])
