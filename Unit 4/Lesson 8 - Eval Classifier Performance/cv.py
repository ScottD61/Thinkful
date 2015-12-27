# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 20:36:20 2015

@author: scdavis6
"""
#Import modules
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import svm
import numpy as np

#Import dataset
iris = datasets.load_iris()
X = iris.data
Y = iris.target

#Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.4)

#Observations in dataset
len(X)
#150
#Observations in test set
len(X_test)
#60
#Observations in training set
len(X_train)
#90

#SVM classifier 
#Create SVM object
clf = svm.SVC()

#Fit classification model with training data
clf.fit(X_train, y_train)
#Predict data in training set
svm_pred = clf.predict(X_train)
#Evaluate accuracy of training set
accuracy_score(y_train, svm_pred)
#0.978

#Predict data in test set
svm_pred2 = clf.predict(X_test)
#Evaluate accuracy of test set
accuracy_score(y_test, svm_pred2)
#0.983

#Accuracy of validation set using k-fold cross validation
#5 folds
accu = cross_val_score(clf, X_train, y_train, scoring = 'accuracy', cv = 5)
#Take average of all cross validation folds
np.mean(accu)
#0.966
#Standard deviation of all cross validation folds
np.std(accu)
#0.043