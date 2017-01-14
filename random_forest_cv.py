import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from process_data import process
from itertools import product

# Takes the data set split into 1 and 0 classes and generates
# a random split according to the parameters:
# 1/N of the 1-class and we add w times as many observations of class 0.
def generate_split(data0, data1, N, w):
    if N<1:
        flag = True
    else:
        flag = False
    data1_temp = data1.sample(frac=1/N, replace=flag)
    size, _ = data1_temp.shape
    temp = data0.sample(int(np.floor(w*size))).append(data1_temp, ignore_index=True).sample(frac=1).reset_index(drop=True)
    return temp.ix[:,:-1], temp.ix[:,-1]

# Given an array of classifiers and test data,
# returns the mean of the predicted probabilities.
def mean_ensemble(rfs, X_test):
    df = pd.DataFrame()
    for i, rf in enumerate(rfs):
        temp = rfs[i].predict_proba(X_test)
        Y_pred = pd.DataFrame(temp)[1]
        df = pd.concat([df,Y_pred], axis = 1)
    return df.mean(axis = 1)

# Given some data, trains N_forest random forest classifiers with n_trees
# the rest of the parameters indicate how to split the data (see generate_split)
def trainClassificationForests(data0, data1, N, w, N_forest, n_trees):
    rfs = []
    for i in range(N_forest):
        temp =  RandomForestClassifier(n_trees)
        rfs.append(temp)
    for i in range(N_forest):
        X_train, Y_train = generate_split(data0, data1, N,w)
        rfs[i].fit(X_train,Y_train)
    return rfs

# Given true classes and predicted classes,
# compute some std evaluation metrics
def eval_classification(Y_test, Y_pred, print_results = False):
    # Y_pred needs to be  1 and 0's, not just probabilitys.
    n = len(Y_test)
    cm = confusion_matrix(Y_test,Y_pred)
    tp = cm[1][1]  # True positives
    fp = cm[0][1]  # False positives
    fn = cm[1][0]  # False negatives
    tn = cm[0][0]  # True negatives
    
    miss = (fp + fn)/n    # missclassification error
    accu = 1 - miss       # accuracy
    recall = tp/(tp + fn) # true positive rate (TPR), sensitivity, recall = True pos./(#real pos.)
    spec = tn/(tn + fp)   # true negative rate (TNR), specificity = True neg./(#real neg.)
    prec = tp/(tp + fp)   # precision = True pos./(#predicted pos.)
    f1 = 2*(prec*recall)/(prec + recall) # F1 score
    auc = roc_auc_score(Y_test, Y_pred)  # Area under the ROC curve.
    
    if print_results:
        print("Missclasification error:", miss)
        print("Recall (Sensitivity):", recall)
        print("Specificity:", spec)
        print("Precision:", prec)
        print("F1-score:", f1)
        print("AUC of sensitivity-specificity:", auc)
    return [miss, recall, spec, prec, f1, auc]
    
if __name__ == "__main__":
    print("Loading data...")
    data = pd.read_csv("data/train.csv")
    process(data)

    train, test = train_test_split(data, test_size = 0.2, random_state = 42)
    X_test, Y_test  = test.ix[:,:-1], test.ix[:,-1]

    happy = train[train.TARGET == 0]
    unhappy = train[train.TARGET == 1]
    
    count=0
    
    print("Data loaded successfully!")
    print("Trainning Forests...")
    with open('cross-val.txt','a') as f:
        for N, w, N_forest, n_trees in  product([0.5], [1], range(60, 61, 10), range(100, 501, 50pr)):
            rfs = trainClassificationForests(happy, unhappy, N, w, N_forest, n_trees)
            Y_prob = mean_ensemble(rfs, X_test)
            #scores = eval_classification(test['TARGET'], Y_pred)
            score = roc_auc_score(test['TARGET'],Y_prob)
            f.write("N={}, w={}, N_forest={}, n_trees={} --> {}\n".format(N, w, N_forest, n_trees, score))
            count+= 1
            print("Total ensembles trained: {}\nLast ensemble trained: N={}, w={}, N_forests={}, n_trees={} --> {}".format(count, N, w, N_forest, n_trees, score), flush = True)
    print("Trainning completed!.")
