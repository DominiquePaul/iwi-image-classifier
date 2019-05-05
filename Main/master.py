import numpy as np
from transfer_learning import Transfer_net
import sklearn
import csv
from tqdm import tqdm

from preprocessing import load_images, read_label_json, return_labelled_images, save_to_numpy_with_labels, save_to_numpy, join_npy_data


CSV_OUT_FILE = 'master_out.csv'

def write_results_to_csv(x_train, x_test, predictions, name, object_name, method_type, data_type, augmented):
    accuracy_test = sklearn.metrics.accuracy_score(y_test, preds)
    f1_score_test = sklearn.metrics.f1_score(y_test, preds)
    [TP, FP], [FN, TN] = sklearn.metrics.confusion_matrix(y_test, preds)

    with open(CSV_OUT_FILE, 'a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["harry", "car", "transfer_net", "custom", "no", len(x_train), len(x_test), accuracy_test, f1_score_test, TP, FP, FN, TN])


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


"""


own_network_config = {
    "conv_layers": 4,
    "conv_filters": 128,
    "dense_layers": 5,
    "dense_neurons": 20,
    "dropout_rate_dense": 0.2,
    "learning_rate": 1e-04,
    "activation_fn": "relu"
}




################################################################################
################################ Process Data ##################################
################################################################################

car_image_folders = ["/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/car/car",
                    "/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/car/no_car"]
target_np_folder = "/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/np_files"
car_json_folder = "/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/json_files/cars"

car_names_raw, car_images = load_images(car_image_folders)
car_labels = read_label_json(car_json_folder)
car_names, car_files = return_labelled_images(car_labels, car_names_raw, car_images)

save_to_numpy_with_labels(target_np_folder, car_files, car_names["label"], "car", augment_training_data=False, split_into_train_test=True)
save_to_numpy_with_labels(target_np_folder, car_files, car_names["label"], "car_augmented", augment_training_data=True, split_into_train_test=True)



# non_augmented
automotive_pckgs = ["/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/np_files/car_image_package_train_test_split0.npy"]
x_train, y_train, x_test, y_test, conversion = join_npy_data(automotive_pckgs)
# prop augmented
automotive_pckgs_augmented = ["/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/np_files/car_augmented_image_package_train_test_split0.npy",
                    "/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/np_files/car_augmented_image_package_train_test_split1.npy",
                    "/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/np_files/car_augmented_image_package_train_test_split2.npy",
                    "/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/np_files/car_augmented_image_package_train_test_split3.npy",
                    "/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/np_files/car_augmented_image_package_train_test_split4.npy",
                    "/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/np_files/car_augmented_image_package_train_test_split5.npy",
                    "/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/np_files/car_augmented_image_package_train_test_split6.npy",
                    "/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/np_files/car_augmented_image_package_train_test_split7.npy"]

# imagenet normal
x_train_imagenet = np.load("/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/ImageNet/image_net_images_imgnet_automobile_x.npy")
y_train_imagenet = np.load("/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/ImageNet/image_net_images_imgnet_automobile_y.npy")

# imagenet augmented
x_train_imagenet_augmented, y_train_imagenet_augmented =  augment_data(x_train_imagenet, y_train_imagenet, shuffle=True)

################################################################################
################################ Own Network ###################################
################################################################################
# load best params here

def run_custom_network(object_name,data_type, augmented):
    m1 = cnn_model()
    m1.new_model(x_train_url, y_train_url, 2, config_v1)
    print("Training custom net for {}".format(object_name))
    m1.train(epochs=2, batch_size=256, on_tpu=True, tb_logs_dir="./log_files/custom_net/")
    y_preds = m1.predict_classes(y_test)
    write_results_to_csv(x_train=x_train, x_test=x_test, predictions=y_preds, name=m1.name, object_name=object_name,
                         method_type="own Network", data_type=data_type, augmented=augmented)

################################################################################
############################## Transfer Network ################################
################################################################################
# load best params

# have to change some parameters here
def run_tranfer_network(object_name,data_type, augmented):
    t_net = Transfer_net("/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/transfernet_files", num_output_classes=2)
    t_net.create_network(layers=5, neurons=100, dropout_rate=0.5)
    x_train_transfer = t_net.cache_transfer_data(x_train, img_group_name="x_train1")
    print("Training transfer net for {}".format(object_name))
    t_net.train(x_train_transfer, y_train, epochs=2, batch_size=256, verbose=True, tb_logs_dir="/Users/dominiquepaul/xBachelorArbeit/Spring19/logs")
    x_test = t_net.cache_transfer_data(x_test, img_group_name="x_test")
    y_preds = t_net.predict_classes(x_test)

    write_results_to_csv(x_train=x_train, x_test=x_test, predictions=y_preds, name="transfer_net_{object_name}".format(), object_name=object_name,
                         method_type="Transfer Network", data_type=data_type, augmented=augmented)


################################################################################
############################## Wordnet Versions  ###############################
################################################################################

def run_wordnet_direct(object_name, data_type, augmented):
    predictions = []
    for img in tqdm(x_test):
        predictions.extend([(identify_item(img, OBJECT, k_labels=label_amount))])
    write_results_to_csv(x_train=[], x_test=x_test, predictions=y_preds, name="wordnet_direct_{object_name}".format(), object_name=object_name,
                         method_type="Transfer Network", data_type=data_type, augmented=augmented)

def run_wordnet_indirect_v1(object_name, data_type, augmented):
    # regression with basic features
    x_train_arr = np.array(x_train_df.loc[:,basic_feats])
    x_test_arr = np.array(x_test_df.loc[:,basic_feats])
    # train regression
    lr = Logistic_regression()
    lr.fit(x_train_arr, y_train)
    lr.find_best_thresh(x_train_arr, y_train, optimize_for="f1", verbose=True)
    y_preds = lr.predict_classes(x_test)
    write_results_to_csv(x_train=[], x_test=x_test, predictions=y_preds, name="wordnet_indirect_v1_{object_name}".format(), object_name=object_name,
                         method_type="Transfer Network", data_type=data_type, augmented=augmented)

def run_wordnet_indirect_v2(object_name, data_type, augmented):
    x_train_arr = np.array(x_train_df.loc[:,wordnet_feats])
    x_test_arr = np.array(x_test_df.loc[:,wordnet_feats])
    # train regression
    lr = Logistic_regression()
    lr.fit(x_train_arr, y_train)
    lr.find_best_thresh(x_train_arr, y_train, optimize_for="f1", verbose=True)
    y_preds = lr.predict_classes(x_test)
    write_results_to_csv(x_train=[], x_test=x_test, predictions=y_preds, name="wordnet_indirect_v2_{object_name}".format(), object_name=object_name,
                         method_type="Transfer Network", data_type=data_type, augmented=augmented)

def run_wordnet_indirect_v3(object_name, data_type, augmented):
    x_train_arr = np.array(x_train_df)
    x_test_arr = np.array(x_test_df)
    # train regression
    lr = Logistic_regression()
    lr.fit(x_train_arr, y_train)
    lr.find_best_thresh(x_train_arr, y_train, optimize_for="f1", verbose=True)
    y_preds = lr.predict_classes(x_test)
    write_results_to_csv(x_train=[], x_test=x_test, predictions=y_preds, name="wordnet_indirect_v3_{object_name}".format(), object_name=object_name,
                         method_type="Transfer Network", data_type=data_type, augmented=augmented)

def run_wordnet_indirect_v4(object_name, data_type, augmented):
    x_train_arr = np.array(x_train_df.loc[:,basic_feats])
    x_test_arr = np.array(x_test_df.loc[:,basic_feats])
    # train regression
    lasso = Lasso_regression()
    lasso.fit(x_train_arr3, y_train)
    lasso.find_best_thresh(x_train_arr3, y_train, optimize_for="f1", verbose=True)
    y_preds = lr.predict_classes(x_test)
    write_results_to_csv(x_train=[], x_test=x_test, predictions=y_preds, name="wordnet_indirect_lasso_{object_name}".format(), object_name=object_name,
                         method_type="Transfer Network", data_type=data_type, augmented=augmented)









































if False:
    #################################
    ### load images without label ###
    #################################
    save_to_numpy(folder_path=target_np_folder,
                img_names=car_names_raw["file_name"],
                files=car_images,
                object="testing_data")
