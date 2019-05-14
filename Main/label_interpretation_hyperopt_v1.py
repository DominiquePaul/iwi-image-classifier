import csv
import numpy as np
from tqdm import tqdm
from sklearn import metrics

from nltk.corpus import wordnet
from preprocessing import join_npy_data
from label_interpretation import identify_item


# path with the industry dict folders
INDUSTRY_DICT_FOLDER_PATH = "../Data/industry_dicts/"

# folder where different label evaluations are saved
FOLDER_PATH_SAVE = "../Data/temp_data_files"

OBJECT = "car"

# non_augmented
automotive_pckgs = ["../Data/np_files/car_image_package_train_test_split0.npy"]
_, _, x_test, y_test, conversion = join_npy_data(automotive_pckgs)

out_file = 'out_files/hyperparameter_opt/wordnet_hyperopt_v1_2_out.csv'
with open(out_file, 'w') as of_connection:
    writer = csv.writer(of_connection)
    writer.writerow(['approach_name','label_amount','train_acc', 'train_f1', 'test_acc','test_f1', 'TP', 'FP', 'FN', 'TN'])

n_label_list = [3,5,10,15,25,50,75,100,125]

for label_amount in tqdm(n_label_list):

    predictions = []
    for img in tqdm(x_test):
        predictions.extend([(identify_item(img, OBJECT, k_labels=label_amount))])

    approach = "direct approach".format(label_amount)
    train_acc = train_f1 = None
    test_acc = metrics.accuracy_score(y_test, predictions)
    test_f1 = metrics.f1_score(y_test, predictions)

    [TP, FP], [FN, TN] = metrics.confusion_matrix(y_test, predictions, labels=None, sample_weight=None)

    with open(out_file, 'a') as of_connection:
        writer = csv.writer(of_connection)
        writer.writerow([approach, label_amount, train_acc, train_f1, np.round(test_acc, 3), np.round(test_f1, 3), TP, FP, FN, TN])
        of_connection.close()
