"""
Data inputs:
* prop normal
* prop augmented
* imagenet normal
* imagenet augmented

Models:
* own network (already optimised)
* transfer learning (already optimised)
* wordnet approaches (5x)


TODO:
    * adjust epochs for networks (10000 & 1000)
    * add timer for each method
"""
import os
import sys
import csv
import sklearn
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import dirname
from timeit import default_timer as timer

from cnn import cnn_model
from transfer_learning import Transfer_net
from label_interpretation import create_feature_df, load_industry_labels, identify_items
from preprocessing import load_images, read_label_json, return_labelled_images, save_to_numpy_with_labels, save_to_numpy, join_npy_data, augment_data, load_from_gcp

sys.path.append(dirname("./modules/"))
from regressionclass import Logistic_regression, Lasso_regression

EVAL_OUT_FILE = './out_files/master_out_food1.csv'
PREDICTIONS_MASTER_OUT_FILE = './out_files/master_predictions_food1.csv'


################################################################################
############################## Some helper function ############################
################################################################################


def write_results_to_csv(x_train, x_test, predictions, run_time, name, object_name, method_type, data_type, augmented):
    accuracy_test = sklearn.metrics.accuracy_score(y_test, predictions)
    f1_score_test = sklearn.metrics.f1_score(y_test, predictions)
    [TP, FP], [FN, TN] = sklearn.metrics.confusion_matrix(y_test, predictions)

    # writes out all summarising results to a csv
    with open(EVAL_OUT_FILE, 'a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([name, object_name, method_type, data_type, augmented, len(x_train), len(x_test), run_time, accuracy_test, f1_score_test, TP, FP, FN, TN])

def write_outputs(x_train, x_test, predictions, run_time, name, object_name, method_type, data_type, augmented):
    global ALL_PREDICTIONS_DF
    write_results_to_csv(x_train, x_test, predictions, run_time, name, object_name, method_type, data_type, augmented)
    ALL_PREDICTIONS_DF[name] = predictions

################################################################################
########################### Write functions for tests ##########################
################################################################################
# load best params here
own_network_config = {
    "conv_layers": 5,
    "conv_filters": 27,
    "dense_layers": 2,
    "dense_neurons": 17,
    "dropout_rate_dense": 0.63341368,
    "learning_rate": 0.000124563
}

def run_custom_network(object_name, data_type, augmented):
    start = timer()
    m1 = cnn_model()
    m1.new_model(x_train, y_train, own_network_config)
    print("Training custom net for {}".format(object_name))
    m1.train(epochs=1000, batch_size=256, on_tpu="dominique-c-a-paul", tb_logs_dir="./out_files/log_files/master_logs/", verbose=True)
    y_preds = m1.predict_classes(x_test)
    run_time = timer() - start
    name = "own_network_{}_{}_{}".format(data_type, augmented, object_name)

    write_outputs(x_train=x_train, x_test=x_test, predictions=y_preds, run_time=run_time, name=name, object_name=object_name,
                         method_type="own Network", data_type=data_type, augmented=augmented)

# have to change some parameters here
def run_transfer_network(object_name,data_type, augmented):
    name = "transfer_net_{}_{}_{}".format(data_type, augmented, object_name)
    start = timer()
    t_net = Transfer_net()
    t_net.create_network(layers=19, neurons=39, dropout_rate=0.486, num_output_classes=2)
    x_train_transfer = t_net.load_transfer_data(x_train)
    print("Training transfer net for {}".format(object_name))
    t_net.train(x_train_transfer, y_train, learning_rate=1.52e-05, epochs=10000, batch_size=256, verbose=True, tb_logs_dir="./out_files/log_files/master_logs/")
    x_test_transfer = t_net.load_or_cache_transfer_data(x_test, file_path= "./temp/x_test" )
    y_preds = t_net.predict_classes(x_test_transfer)
    run_time = timer() - start
    write_outputs(x_train=x_train, x_test=x_test, predictions=y_preds, run_time=run_time, name=name, object_name=object_name,
                         method_type="Transfer Network", data_type=data_type, augmented=augmented)

def run_wordnet_direct(object_name, data_type, augmented):
    start = timer()
    global x_test
    predictions=identify_items(x_test, [object_name], k_labels=100, use_synonyms=True)
    run_time = timer() - start
    name = "direct_wordnet_100_labels_{}_{}_{}".format(data_type, augmented, object_name)
    df = write_outputs(x_train=[], x_test=x_test, predictions=predictions, run_time=run_time, name=name, object_name=object_name,
                         method_type="oob_network_eval", data_type=data_type, augmented=augmented)

def run_wordnet_indirect_v3(object_name, data_type, augmented):
    start = timer()
    x_train_df = create_feature_df(imgs=x_train, object_name=object_name, ind_labels=ind_labels, k_labels=50) # 50 labels
    x_train_arr = np.array(x_train_df)
    x_test_arr = np.array(x_test_df_50)
    # train regression
    lr = Logistic_regression()
    lr.fit(x_train_arr, y_train)
    lr.find_best_thresh(x_train_arr, y_train, optimize_for="f1", verbose=True)
    y_preds = lr.predict_classes(x_test_arr)
    run_time = timer() - start
    name = "indirect_wordnet_v3_50_labels_{}_{}_{}".format(data_type, augmented, object_name)
    write_outputs(x_train=x_train, x_test=x_test, predictions=y_preds, run_time=run_time, name=name, object_name=object_name,
                         method_type="oob_network_eval", data_type=data_type, augmented=augmented)

def run_wordnet_indirect_v4(object_name, data_type, augmented):
    start = timer()
    x_train_df = create_feature_df(imgs=x_train, object_name=object_name, ind_labels=ind_labels, k_labels=20) # 20 labels
    x_train_arr = np.array(x_train_df)
    x_test_arr = np.array(x_test_df_20)
    # train regression
    lasso = Lasso_regression()
    lasso.fit(x_train_arr, y_train)
    lasso.find_best_thresh(x_train_arr, y_train, optimize_for="f1", verbose=True)
    y_preds = lasso.predict_classes(x_test_arr)
    run_time = timer() - start
    name = "indirect_wordnet_lasso_20_labels_{}_{}_{}".format(data_type, augmented, object_name)
    write_outputs(x_train=x_train, x_test=x_test, predictions=y_preds, run_time=run_time, name=name, object_name=object_name,
                         method_type="oob_network_eval", data_type=data_type, augmented=augmented)


################################################################################
########################### Run through all tests ##############################
################################################################################
OBJECT_NAME = "food"
DATA_FOLDER_PATH = "gs://data-imr-unisg/packaged_food_data/"

ind_labels = load_industry_labels(file_path="./industry_dicts/selection_PackagedFoodsandMeats.csv")


x_test, y_test, names, _  = load_from_gcp(os.path.join(DATA_FOLDER_PATH, "food_final_testing_dataset.npy"))

x_test_df_20 = create_feature_df(imgs=x_test, object_name=OBJECT_NAME, ind_labels=ind_labels, k_labels=20)
x_test_df_50 = create_feature_df(imgs=x_test, object_name=OBJECT_NAME, ind_labels=ind_labels, k_labels=50)

ALL_PREDICTIONS_DF = pd.DataFrame({"names":names})
# only method that doesnt require a training set
run_wordnet_direct("food", "custom", "Unaugmented")


# run 1/4: own images not augmented
automotive_pckgs = [os.path.join(DATA_FOLDER_PATH, "food_image_package_train_val_split_0.npy"),
                    os.path.join(DATA_FOLDER_PATH, "food_image_package_train_val_split_1.npy")]
x_train, y_train, _, _, conversion = join_npy_data(automotive_pckgs, training_data_only=False)

run_custom_network(OBJECT_NAME, "custom", "Unaugmented")
run_transfer_network(OBJECT_NAME, "custom", "Unaugmented")
run_wordnet_indirect_v3(OBJECT_NAME, "custom", "Unaugmented")
run_wordnet_indirect_v4(OBJECT_NAME, "custom", "Unaugmented")


# run 2/4: own images augmented
automotive_pckgs_augmented = [os.path.join(DATA_FOLDER_PATH, "food_image_package_train_val_split_augmented_0.npy"),
                    os.path.join(DATA_FOLDER_PATH, "food_image_package_train_val_split_augmented_1.npy"),
                    os.path.join(DATA_FOLDER_PATH, "food_image_package_train_val_split_augmented_2.npy"),
                    os.path.join(DATA_FOLDER_PATH, "food_image_package_train_val_split_augmented_3.npy"),
                    os.path.join(DATA_FOLDER_PATH, "food_image_package_train_val_split_augmented_4.npy"),
                    os.path.join(DATA_FOLDER_PATH, "food_image_package_train_val_split_augmented_5.npy"),
                    os.path.join(DATA_FOLDER_PATH, "food_image_package_train_val_split_augmented_6.npy"),
                    os.path.join(DATA_FOLDER_PATH, "food_image_package_train_val_split_augmented_7.npy"),
                    os.path.join(DATA_FOLDER_PATH, "food_image_package_train_val_split_augmented_8.npy"),
                    os.path.join(DATA_FOLDER_PATH, "food_image_package_train_val_split_augmented_9.npy"),
                    os.path.join(DATA_FOLDER_PATH, "food_image_package_train_val_split_augmented_10.npy"),
                    os.path.join(DATA_FOLDER_PATH, "food_image_package_train_val_split_augmented_11.npy"),
                    os.path.join(DATA_FOLDER_PATH, "food_image_package_train_val_split_augmented_12.npy"),
                    os.path.join(DATA_FOLDER_PATH, "food_image_package_train_val_split_augmented_13.npy"),
                    os.path.join(DATA_FOLDER_PATH, "food_image_package_train_val_split_augmented_14.npy"),
                    os.path.join(DATA_FOLDER_PATH, "food_image_package_train_val_split_augmented_15.npy")]
x_train, y_train, _, _, conversion = join_npy_data(automotive_pckgs_augmented, training_data_only=False)

run_custom_network(OBJECT_NAME, "custom", "Augmented")
run_transfer_network(OBJECT_NAME, "custom", "Augmented")
# we omit the inception/wordnet approaches, because the pure processing of the
# images takes too much time with 11x images, but is not expected to have a major impact

#
# # run 3/4: imagenet images not augmented
# x_train = load_from_gcp(os.path.join(DATA_FOLDER_PATH, "image_net_files/image_net_images_imgnet_automobile_x.npy"))
# y_train = load_from_gcp(os.path.join(DATA_FOLDER_PATH, "image_net_files/image_net_images_imgnet_automobile_y.npy"))
#
# run_custom_network(OBJECT_NAME, "ImageNet", "Unaugmented")
# run_transfer_network(OBJECT_NAME, "ImageNet", "Unaugmented")
# run_wordnet_indirect_v3(OBJECT_NAME, "ImageNet", "Unaugmented")
# run_wordnet_indirect_v4(OBJECT_NAME, "ImageNet", "Unaugmented")
#
#
# # run 4/4: imagenet images augmented
# x_train, y_train =  augment_data(x_train, y_train, shuffle=True)
# run_custom_network(OBJECT_NAME, "ImageNet", "Augmented")
# run_transfer_network(OBJECT_NAME, "ImageNet", "Augmented")
# # we again omit the inception/wordnet approaches for the augmented images


ALL_PREDICTIONS_DF.to_csv(PREDICTIONS_MASTER_OUT_FILE, index=False)

print("All tests finished")


























#
