"""
find script that already classifies images by loading the labels
copy necessary files into new folder
examine possibile measurements of similarity in wordnet
examine different methods of making sense of independent similarities in the dictionary
# cross validate approaches


similarity scores explained:
https://linguistics.stackexchange.com/questions/9084/what-do-wordnetsimilarity-scores-mean

Other similarities to try:
- Path similarity
- Leacock-Chodorow Similarity (Leacock and Chodorow 1998)
- Resnik Similarity (Resnik 1995)
- Lin Similarity (Lin 1998b)
- Jiang-Conrath distance (Jiang and Conrath 1997)

to do:
    -cross valiate cutoffs and other methods for approach 1
    -iterate over all datasets
"""

from nltk.corpus import wordnet
import statsmodels.api as sm
import numpy as np
import pandas as pd
import os
from sklearn import metrics
import csv
from tqdm import tqdm

from os.path import dirname
import sys
sys.path.append(dirname("/Users/dominiquepaul/xBachelorArbeit/Spring19/Bachelor-arbeit/4-wordnet/modules/"))

import inception_edit_dom as inception
from  regressionclass import Logistic_regression, Lasso_regression

# load dictionary
def load_industry_labels(file_name):
    file_path = os.path.join(INDUSTRY_DICT_FOLDER_PATH, file_name)
    ind_df = pd.read_csv(file_path)
    return ind_df.labelname

def max_wup_score(tested_word, tested_against):
    '''
    Tests two words against each other for their WUP similarity (WUP = )
    '''
    w1 = wordnet.synsets(tested_word)
    w2 = wordnet.synsets(tested_against)
    if len(w1) == 0 or len(w2) == 0:
        return 0
    wup = []
    for i in w1:
        for j in w2:
            score = i.wup_similarity(j)
            if score is None:
                score = 0
            wup.extend([score])
    try:
        max_score = np.max(wup)
    except:
        print(tested_word, tested_against)
    return max_score

def extract_features_basic(label_dict, ind_label_list):
    """
    Takes a dictionary of predicted labels and a list of target word and extracts different scores:
        maxscorevalue_dict: the maximum score of a label contained in the target list
        product_ref_dict: is a product referenced at least once
        product_count_dict: how often is a product referenced?
        product_sum_dict: the sum of all label scores associated with the target list

    """
    maxscorevalue_dict = 0 # highest probability score of relevant label
    product_ref_dict = 0 # is there at least one product reference
    product_count_dict = 0 # number of product references
    product_sum_dict = 0 # number of product references weighted by score

    for label in label_dict.keys():
        words = label.split(" ")
        if list(set(words) & set(ind_label_list)) != []:
            product_ref_dict = 1
            product_count_dict += 1
            product_sum_dict += label_dict[label]
            if label_dict[label] > maxscorevalue_dict:
                maxscorevalue_dict = label_dict[label]

    score_dict = {"max_score":maxscorevalue_dict,
                "product_reference": product_ref_dict,
                "product_ref_count": product_count_dict,
                "product_ref_sum": product_sum_dict}

    return score_dict

def extract_features_wordnet(label_dict, comparison_word):

    maxscorevalue_wordnet = 0 # highest probability score of relevant label
    product_count_wordnet = 0 # sum of inception likelihoods
    product_sum_wordnet = 0 # wup score * inception likelihood score

    for label, value in label_dict.items():
        words = label.split(" ")
        max_similarity = 0

        for word in words:
            score = max_wup_score(word, comparison_word)
            if score > max_similarity:
                max_similarity = score

        if max_similarity > maxscorevalue_wordnet:
            maxscorevalue_wordnet = max_similarity

        if max_similarity > 0:
            product_count_wordnet += value
            product_sum_wordnet += max_similarity * value

    score_dict = {"maxscorevalue_wordnet":maxscorevalue_wordnet,
                "product_count_wordnet": product_count_wordnet,
                "product_sum_wordnet": product_sum_wordnet}
    return score_dict

def extract_features(img ,industry_labels=None, object_name=None, k_labels=10, basic_feats=True, wordnet_feats=True):
    pred_dict = model.return_score_dict(k=k_labels, image_path = None, image=img)
    features1 = {}
    features2 = {}
    if basic_feats:
        features1 = extract_features_basic(pred_dict, industry_labels)
    if wordnet_feats:
        features2 = extract_features_wordnet(pred_dict, object_name)
    features = {**features1, **features2}
    return(features)

def create_feature_df(imgs, object_name=None, ind_labels=None, k_labels=10, basic_feats=True, wordnet_feats=True):
    df = pd.DataFrame([])
    for i in range(imgs.shape[0]):
        features = extract_features(img=imgs[i], object_name=object_name, industry_labels=ind_labels, k_labels=k_labels, basic_feats=basic_feats , wordnet_feats=wordnet_feats)
        df_temp = pd.DataFrame([features])
        df = df.append(df_temp, ignore_index=True)
    return df


# if __name__ == "__main__":

# download the inception network if necessary
inception.maybe_download()
model = inception.Inception()

# path with the industry dict folders
INDUSTRY_DICT_FOLDER_PATH = "/Users/dominiquepaul/xBachelorArbeit/Spring19/Bachelor-arbeit/4-wordnet/industry_dicts"

# folder where different label evaluations are saved
FOLDER_PATH_SAVE = "/Users/dominiquepaul/xBachelorArbeit/Spring19/Bachelor-arbeit/4-wordnet/temp_data_files"

OBJECT = "car"
ind_labels = load_industry_labels(file_name="selection_AutomobileManufacturers.csv")

x_train_path = "/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/np_array_files/x_train.npy"
y_train_path = "/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/np_array_files/class_labels_train.npy"
x_test_path = "/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/np_array_files/x_test.npy"
y_test_path = "/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/np_array_files/class_labels_test.npy"
x_train = np.load(x_train_path)
y_train = np.load(y_train_path)
x_test = np.load(x_test_path)
y_test = np.load(y_test_path)

# transform or load the data if necessary
if False:
    x_train_df = create_feature_df(imgs=x_train, object_name=OBJECT, ind_labels=ind_labels, k_labels=10)
    x_test_df = create_feature_df(imgs=x_test, object_name=OBJECT, ind_labels=ind_labels, k_labels=10)
    x_train_df.to_csv(FOLDER_PATH_SAVE + "/train")
    x_test_df.to_csv(FOLDER_PATH_SAVE + "/test")
else:
    x_train_df = pd.read_csv(FOLDER_PATH_SAVE + "/train")
    x_test_df = pd.read_csv(FOLDER_PATH_SAVE + "/test")


basic_feats = ["max_score", "product_ref_count", "product_ref_sum", "product_reference"]
wordnet_feats = ["product_count_wordnet", "maxscorevalue_wordnet", "product_sum_wordnet"]

out_file = 'wordnet_results.csv'
with open(out_file, 'w') as of_connection:
    writer = csv.writer(of_connection)
    writer.writerow(['approach_name','train_acc', 'train_f1', 'test_acc','test_f1'])



#####################
#### approach 1 #####
#####################
# direct wordnet
# probably increasing label amount or amount of words sought might improve results

def identify_item(img, item_synonyms):
    """ a model that checks whether a group of words is contained in the
    the inception model predictions
    """
    prediction_scores = model.return_score_dict(k=10, image_path = None, image=img)
    predicted_words = [i.split(" ") for i in prediction_scores.keys()]
    predicted_words = [item for sublist in predicted_words for item in sublist]
    item_found = not set(predicted_words).isdisjoint(set(item_synonyms))
    return int(item_found)

synonyms = []
for word in wordnet.synsets(OBJECT):
    synonyms.extend(word.lemma_names())

predictions = []
for img in tqdm(x_train):
    predictions.extend([(identify_item(img, synonyms))])

approach = "direct_identification"
train_acc = train_f1 = None
test_acc = metrics.accuracy_score(y_train, predictions)
test_f1 = metrics.f1_score(y_train, predictions)

with open(out_file, 'a') as of_connection:
    writer = csv.writer(of_connection)
    writer.writerow([approach, train_acc, train_f1, test_acc, test_f1])
    of_connection.close()

metrics.confusion_matrix(y_train, predictions, labels=None, sample_weight=None)


#####################
#### approach 2 #####
#####################
# regression with basic features
approach = "regression_basic"

x_train_arr = np.array(x_train_df.loc[:,basic_feats])
x_test_arr = np.array(x_test_df.loc[:,basic_feats])

# train regression
lr1 = Logistic_regression()
lr1.fit(x_train_arr, y_train)
lr1.find_best_thresh(x_train_arr, y_train, optimize_for="f1", verbose=True)

# evalaute
train_acc, train_f1 = lr1.evaluate(x_train_arr, y_train)
test_acc, test_f1 = lr1.evaluate(x_test_arr, y_test)

with open(out_file, 'a') as of_connection:
    writer = csv.writer(of_connection)
    writer.writerow([approach, train_acc, train_f1, test_acc, test_f1])
    of_connection.close()


#####################
#### approach 3 #####
#####################
# regression with wordnet features
approach =  "regression_wordnet"

x_train_arr2 = np.array(x_train_df.loc[:,wordnet_feats])
x_test_arr2 = np.array(x_test_df.loc[:,wordnet_feats])

# train regression
lr2 = Logistic_regression()
lr2.fit(x_train_arr2, y_train)
lr2.find_best_thresh(x_train_arr2, y_train, optimize_for="f1", verbose=True)

# evalaute
train_acc, train_f1 = lr2.evaluate(x_train_arr2, y_train)
test_acc, test_f1 = lr2.evaluate(x_test_arr2, y_test)

with open(out_file, 'a') as of_connection:
    writer = csv.writer(of_connection)
    writer.writerow([approach, train_acc, train_f1, test_acc, test_f1])
    of_connection.close()



#####################
#### approach 4 #####
#####################
# using all features
approach = "regression_all_feats"

x_train_arr3 = np.array(x_train_df)
x_test_arr3 = np.array(x_test_df)

# train regression
lr3 = Logistic_regression()
lr3.fit(x_train_arr3, y_train)
lr3.find_best_thresh(x_train_arr3, y_train, optimize_for="f1", verbose=True)

# evalaute
train_acc, train_f1 = lr3.evaluate(x_train_arr3, y_train)
test_acc, test_f1 = lr3.evaluate(x_test_arr3, y_test)

with open(out_file, 'a') as of_connection:
    writer = csv.writer(of_connection)
    writer.writerow([approach, train_acc, train_f1, test_acc, test_f1])
    of_connection.close()



#####################
#### approach 5 #####
#####################
# lasso regression
approach = "Lasso_regression"

# train regression
lasso = Logistic_regression()
lasso.fit(x_train_arr3, y_train)
lasso.find_best_thresh(x_train_arr3, y_train, optimize_for="f1", verbose=True)

# evalaute
train_acc, train_f1 = lasso.evaluate(x_train_arr3, y_train)
test_acc, test_f1 = lasso.evaluate(x_test_arr3, y_test)

with open(out_file, 'a') as of_connection:
    writer = csv.writer(of_connection)
    writer.writerow([approach, train_acc, train_f1, test_acc, test_f1])
    of_connection.close()

















#
