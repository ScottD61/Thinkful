# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 12:48:29 2015

@author: scdavis6
"""
#Import modules - in progress 
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.cross_validation import cross_val_score
from sklearn.lda import LDA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Import dataset
iris = datasets.load_iris()
X = iris.data
Y = iris.target

#Principal component analysis
#Standardize
pca = PCA(n_components = 3, whiten = True)
#Fit model
pca.fit(X)
#Transform data
x = pca.transform(X)
#####
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors = 3)
neigh.fit(x, Y) 

#Get cross validation metrics
#Create validation set with k-fold cross validation
#Test for accuracy of validation set
score = cross_val_score(neigh, x, Y, scoring = 'accuracy', cv = 10)
np.mean(score)
#0.913

#Convert array to dataframe
df = pd.DataFrame(x, columns = ['C1', 'C2', 'C3'])

#3 dimentional plot of principal components (eigenvectors)  
fig = plt.figure(1)
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(df['C1'].tolist(), df['C2'].tolist(), df['C3'].tolist(), c = 'b', marker = '*')
ax.set_xlabel('C1')
ax.set_ylabel('C2')
ax.set_zlabel('C3')
ax.legend()

#Linear discriminant analysis
#Standardize
da = LDA(n_components = 3)
#Fit model and transform data
a = da.fit_transform(X, Y)

#Perform KNN algorithm
neigh1 = KNeighborsClassifier(n_neighbors = 3)
neigh1.fit(a, Y) 

#Get cross validation metrics
#Create validation set with k-fold cross validation
#Test for accuracy of validation set
score1 = cross_val_score(neigh1, a, Y, scoring = 'accuracy', cv = 10)
np.mean(score1)
#0.973