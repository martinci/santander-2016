#
# UTILITY FUNCTIONS TO TRAIN RANDOM FOREST CLASSIFIERS
#
# Author: Martin Citoler-Saumell
# Date: 2016-01-13
#

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from process_data import process
from itertools import product


# This function generates random subsamples of the data with some specified
# proportion of positive and negative classes.
#   data0: observations in the data with TARGET=O (i.e. happy customers).
#   data1: observations in the data with TARGET=1 (i.e unhappy customers).
#   a: fraction of the observations with TARGET=1 to sample.
#   w: desired proportion for the TARGET=0 observations in the sample.
#   return --> random weighted subsample of size (w+1)*a, split for training.
def generate_split(data0, data1, a, w=1):
    if a>1:
        flag = True
    else:
        flag = False
    data1_temp = data1.sample(frac=a, replace=flag)
    size, _ = data1_temp.shape
    temp = data0.sample(int(np.floor(w*size))).append(data1_temp, ignore_index=True).sample(frac=1).reset_index(drop=True)
    return temp.ix[:,:-1], temp.ix[:,-1]

# Given an array of binary classifiers, this function computes the mean of the positive class probabilities.
#   rfs: array of binary classifiers.
#   X_test: test data to predict.
#   return --> array of mean probabilities for the positive class.
def mean_ensemble(rfs, X_test):
    df = pd.DataFrame()
    for i, rf in enumerate(rfs):
        temp = rfs[i].predict_proba(X_test)
        Y_pred = pd.DataFrame(temp)[1]
        df = pd.concat([df,Y_pred], axis = 1)
    return df.mean(axis = 1)


# This function trains a number of random forest classifiers with specified
# weights for the positive and negative classes.
#   data: training data set.
#   a: fraction of the observations with TARGET=1 to sample.
#   w: desired proportion for the TARGET=0 observations in the sample.
#   N_forest: total number of random forest classifiers to train.
#   n_trees: the number of trees for each forest (can be an array to
#            specify different # of trees for different forests).
#   return --> array of trained random forest classifiers.
def trainForests(data, a, w, N_forest, n_trees):
    rfs = []
    data0, data1 = data[data.ix[:,-1]==0], data[data.ix[:,-1]==1]
    if isinstance(n_trees, list):
        for i in range(N_forest):
            temp =  RandomForestClassifier(n_trees[i])
            X_train, Y_train = generate_split(data0, data1, a, w)
            temp.fit(X_train, Y_train)
            rfs.append(temp)
    else:
        for i in range(N_forest):
            temp =  RandomForestClassifier(n_trees)
            X_train, Y_train = generate_split(data0, data1, a, w)
            temp.fit(X_train, Y_train)
            rfs.append(temp)        
    return rfs

# This function computes some common evaluation metrics for binary classifiers.
#   Y_test: array of true classes.
#   Y_pred: array of predicted classes.
#   print_results: boolean option to print on screen.
#   return -->  array of scores.
def eval_classification(Y_test, Y_pred, print_results = False):
    # Y_pred needs to be  1 and 0's, not just probabilities.
    n = len(Y_test)
    cm = confusion_matrix(Y_test,Y_pred)
    tp = cm[1][1]  # True positives
    fp = cm[0][1]  # False positives
    fn = cm[1][0]  # False negatives
    tn = cm[0][0]  # True negatives
    miss = (fp + fn)/n    # misclassification error
    accu = 1 - miss       # accuracy
    recall = tp/(tp + fn) # true positive rate (TPR), sensitivity, recall = True pos./(#real pos.)
    spec = tn/(tn + fp)   # true negative rate (TNR), specificity = True neg./(#real neg.)
    prec = tp/(tp + fp)   # precision = True pos./(#predicted pos.)
    f1 = 2*(prec*recall)/(prec + recall) # F1 score
    auc = roc_auc_score(Y_test, Y_pred)  # Area under the ROC curve.
    if print_results:
        print(cm,[["TP", "FP"], ["FN", "TP"]])
        print("Misclassification error:", miss)
        print("Recall (Sensitivity):", recall)
        print("Specificity:", spec)
        print("Precision:", prec)
        print("F1-score:", f1)
        print("AUC of sensitivity-specificity:", auc)
    return [miss, recall, spec, prec, f1, auc]


# This script generates cross-validation data if run independently.
if __name__ == "__main__":
    print("Loading data...")
    data = pd.read_csv("data/train.csv")
    process(data)

    train, test = train_test_split(data, test_size = 0.2, random_state = 42)
    X_test, Y_test  = test.ix[:,:-1], test.ix[:,-1]

    count=0
    
    print("Data successfully loaded!")
    print("Training Forests...")
    with open("cross-val.txt",'a') as f:
        # Tune the ranges here for parameter training.
        for a, w, N_forest, n_trees in  product([0.25], [1], range(60, 61, 10), range(100, 501, 50)):
            rfs = trainForests(train, a, w, N_forest, n_trees)
            Y_prob = mean_ensemble(rfs, X_test)
            #scores = eval_classification(test.ix[:,-1], Y_pred)
            score = roc_auc_score(test.ix[:,-1],Y_prob)
            f.write("a={}, w={}, N_forest={}, n_trees={} --> {}\n".format(a, w, N_forest, n_trees, score))
            count+= 1
            print("Total ensembles trained: {}\nLast ensemble trained: a={}, w={}, N_forests={}, n_trees={} --> {}".format(count, a, w, N_forest, n_trees, score), flush = True)
    print("Training completed!.")
