# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 09:53:18 2015

@author: scdavis6
"""
#Import modules 
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score

#Use grid search

#Import dataset
iris = datasets.load_iris()
X = iris.data
Y = iris.target

#Create visualzations
plt.scatter(X[:, 0], X[:, 1], c = Y, cmap = plt.cm.Paired)
plt.xlabel('Sepal length') 
plt.ylabel('Sepal width')

#Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y)

#KNN algorithm training set
neigh = KNeighborsClassifier(n_neighbors = 10)
neigh.fit(X_train, y_train)

#Create validation set with k-fold cross validation
#Test for accuracy of validation set
score = cross_val_score(neigh, X_train, y_train, scoring = 'accuracy', cv = 10)
np.mean(score)
#0.96

#Predict values for test set
pred_test = neigh.predict(X_test)

#Compute accuracy of test set
accuracy_score(pred_test, y_test)
#0.947

