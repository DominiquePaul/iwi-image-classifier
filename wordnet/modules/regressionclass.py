#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 15:56:48 2018

@author: dominiquepaul
"""
import statsmodels.api as sm
import numpy as np
import pandas as pd
from sklearn import metrics

class regression_class:

    def __init__(self):
        #self.dict = dict
        self.trained_vars = None

    def fit(self, x_train, y_train):
        model = sm.OLS(x_train, y_train)
        self.results = model.fit()
        self.trained_vars = self.results.params

    def reset(self):
        self.trained_vars = None

    def predict(self, x_test):
        preds = self.results.predict(x_test)
        return preds

    def predict_v2(self, x_test):
        results = np.matmul(x_test, np.transpose(self.trained_vars ))
        results = np.squeeze(results)
        results = np.round(results, 0)
        return(results)

    def evaluate(self, x_test, y_test):
        predictions = self.predict(x_test)
        accuracy = np.sum(np.equal(predictions,y_test)) / len(x_test)
        return(np.round(accuracy,2))

class logistic_regression_class:

    def __init__(self):
        #self.dict = dict
        self.trained_vars = None
        self.best_thresh = 0.5

    def fit(self, x_train, y_train):
        model = sm.Logit(y_train,x_train)
        self.results = model.fit()
        self.trained_vars = self.results.params

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
        preds = self.results.predict(x_test)
        return preds

    def predict_classes(self, x_test):
        preds = self.predict(x_test)
        predicted_classes = np.where(preds > self.best_thresh, 1, 0)
        return(predicted_classes)

    def evaluate(self, x_test, y_test, rounded_to=3):
        predictions = self.predict_classes(x_test)
        accuracy = metrics.accuracy_score(y_test, predictions)
        f1_metric = metrics.f1_score(y_test, predictions)
        return(np.round(accuracy,rounded_to), np.round(f1_metric,rounded_to))

if __name__ == "__main__":

	from sklearn.model_selection import train_test_split

	df = pd.read_csv("example.csv")

	inputs = df.iloc[:,0:5]
	outputs = df.iloc[:,5]

	x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, train_size = 0.7)

	rc = regression_class()
	rc.fit(x_train,y_train)
	rc.predict(x_test)
	rc.evaluate(x_test,y_test)










"""
df = pd.read_csv("/Users/dominiquepaul/Desktop/example.csv")
df.head()

xtrain = df.iloc[:7,0:5]
xtrain.head()

ytrain = df.iloc[:7,5]
ytrain.head()

xtest = df.iloc[7:,0:5]
xtest.head()

ytest = df.iloc[7:,5]
ytest.head()


rc = regression_class()
rc.fit(xtrain,ytrain)
rc.predict(xtest)


"""
