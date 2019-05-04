#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 15:56:48 2018

@author: dominiquepaul
"""
import statsmodels.api as sm
import numpy as np
import pandas as pd
from sklearn import metrics, linear_model
from sklearn.preprocessing import StandardScaler

class Logistic_regression:

    def __init__(self):
        #self.dict = dict
        self.trained_vars = None
        self.best_thresh = 0.5

    def fit(self, x_train, y_train):
        model = sm.Logit(y_train,x_train)
        self.model = model.fit()
        self.trained_vars = self.model.params

    def all_decision_thresholds(self, x_feats, y_true):
        preds = self.predict(x_feats)
        overview_df = pd.DataFrame()
        for thresh in np.arange(0,1,0.01):
            predictions = np.where(preds > thresh, 1,0)
            accuracy = metrics.accuracy_score(y_true, predictions)
            f1 = metrics.f1_score(y_true, predictions)
            metrics1 = pd.DataFrame([{"thresh":thresh, "accuracy":accuracy, "f1":f1}])
            overview_df = overview_df.append(metrics1)
        overview_df.reset_index(inplace=True)
        return overview_df

    def find_best_thresh(self, x_feats, y_true, optimize_for, verbose=False):
        overview_df = self.all_decision_thresholds(x_feats=x_feats, y_true=y_true)
        f1_max_index = overview_df["f1"].idxmax()
        f1_max_thresh = overview_df.loc[f1_max_index,"thresh"]
        f1_max = overview_df.loc[f1_max_index, "f1"]
        accuracy_max_index = overview_df["accuracy"].idxmax()
        accuracy_max = overview_df.loc[accuracy_max_index, "accuracy"]
        accuracy_max_thresh = overview_df.loc[accuracy_max_index,"thresh"]
        if verbose:
            print("Max f1: {}, at index {} and threshold {}".format(f1_max, f1_max_index, f1_max_thresh))
            print("Max accuracy: {}, at index {} and threshold {}".format(accuracy_max, accuracy_max_index, accuracy_max_thresh))
        if optimize_for=="f1":
            self.best_thresh = f1_max_thresh
            return(f1_max_thresh)
        elif optimize_for=="accuracy":
            self.best_thresh = accuracy_max_thresh
            return(accuracy_max_thresh)
        else:
            raise ValueError("'optimize_for' has to be set to either 'f1' or 'accuracy'")

    def reset(self):
        self.trained_vars = None

    def predict(self, x_test):
        preds = self.model.predict(x_test)
        return preds

    def predict_classes(self, x_test):
        preds = self.predict(x_test)
        predicted_classes = np.where(preds > self.best_thresh, 1, 0)
        return(predicted_classes)

    def evaluate(self, x_test, y_test, rounded_to=3):
        predictions = self.predict_classes(x_test)
        accuracy = metrics.accuracy_score(y_test, predictions)
        f1_metric = metrics.f1_score(y_test, predictions)
        [TP, FP], [FN, TN] = metrics.confusion_matrix(y_test, predictions)
        return(np.round(accuracy,rounded_to), np.round(f1_metric,rounded_to), (TP, FP, FN, TN))

class Lasso_regression(Logistic_regression):

    def fit(self, x_train, y_train):
        self.scaler = StandardScaler()
        self.scaler.fit(x_train)
        x_train = self.scaler.transform(x_train)
        self.model = linear_model.Lasso(alpha=0.1) # 0.1 chosen randomly
        self.model.fit(x_train, y_train)
        self.trained_vars = self.model.get_params()

    def predict(self, x_test):
        x_test = self.scaler.transform(x_test)
        preds = self.model.predict(x_test)
        return preds


if __name__ == "__main__":

    from sklearn.model_selection import train_test_split
    import pandas as pd

    df = pd.read_csv("/Users/dominiquepaul/xBachelorArbeit/Daten/July Version/classwork/example.csv")

    inputs = np.array(df.iloc[:,0:4])
    outputs = np.array(df.iloc[:,5])

    x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, train_size = 0.7)

    rc = Logistic_regression()
    rc.fit(x_train,y_train)
    rc.predict(x_train)
    rc.evaluate(x_test,y_test)
