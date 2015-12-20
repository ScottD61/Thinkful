# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 09:29:04 2015

@author: scdavis6
"""
#Import modules
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import svm
import numpy as np

#Import dataset
iris = datasets.load_iris()

#Create visualizations of dataset
#Graph 1 - petal length vs sepal width
plt.scatter(iris.data[:, 1], iris.data[:, 2], c = iris.target)
plt.xlabel(iris.feature_names[1])
plt.ylabel(iris.feature_names[2])
#Graph 2 - petal length vs sepal width with reduced observations
plt.scatter(iris.data[0:100, 1], iris.data[0:100, 2], c = iris.target[0:100])
plt.xlabel(iris.feature_names[1])
plt.ylabel(iris.feature_names[2])

#SVM model
#Create svm object
svc = svm.SVC(kernel = 'linear')

#Combination 1 - petal length vs sepal width
#Subset data
X = iris.data[0:100, 1:3]
y = iris.target[0:100]
#Fit the data
svc.fit(X, y)

#Create visualizations of svm kernel
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
#Create function for the plot
def plot_estimator(estimator, X, y):
    estimator.fit(X, y)
    #Subset petal length and sepal width
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])
    #Put result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap = cmap_light)
    #Plot training points
    plt.scatter(X[:, 0], X[:, 1], c = y, cmap = cmap_bold)
    plt.axis('tight')
    plt.axis('off')
    plt.tight_layout()
#Draw decision boundary
plot_estimator(svc, X, y)

#Combination 2 - petal length and petal width
#Subset data
X = iris.data[0:100, 2:4]
y = iris.target[0:100]
#Fit the data
svc.fit(X, y)

#Create visualizations of svm kernel
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
#Create function for the plot
def plot_estimator(estimator, X, y):
    estimator.fit(X, y)
    #Subset petal length and sepal width
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])
    #Put result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap = cmap_light)
    #Plot training points
    plt.scatter(X[:, 0], X[:, 1], c = y, cmap = cmap_bold)
    plt.axis('tight')
    plt.axis('off')
    plt.tight_layout()
#Draw decision boundary
plot_estimator(svc, X, y)
    
#Combination 3 - sepal length and sepal width
#Subset data
X = iris.data[0:100, 0:2]
y = iris.target[0:100]
#Fit the data
svc.fit(X, y)   
    
#Create visualizations of svm kernel
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
#Create function for the plot
def plot_estimator(estimator, X, y):
    estimator.fit(X, y)
    #Subset petal length and sepal width
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])
    #Put result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap = cmap_light)
    #Plot training points
    plt.scatter(X[:, 0], X[:, 1], c = y, cmap = cmap_bold)
    plt.axis('tight')
    plt.axis('off')
    plt.tight_layout()
#Draw decision boundary
plot_estimator(svc, X, y)  
    
#Combination 4 - sepal length and petal width     
#Subset data
X = iris.data[0:100, [0, 3]]
y = iris.target[0:100]
#Fit the data
svc.fit(X, y)   
    
#Create visualizations of svm kernel
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
#Create function for the plot
def plot_estimator(estimator, X, y):
    estimator.fit(X, y)
    #Subset petal length and sepal width
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])
    #Put result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap = cmap_light)
    #Plot training points
    plt.scatter(X[:, 0], X[:, 1], c = y, cmap = cmap_bold)
    plt.axis('tight')
    plt.axis('off')
    plt.tight_layout()
#Draw decision boundary
plot_estimator(svc, X, y) 

#Classification multiple independent variables
#Subset data
X = iris.data[0:100, 0:3]
y = iris.target[0:100]
#Fit the data
svc.fit(X, y) 
   
   