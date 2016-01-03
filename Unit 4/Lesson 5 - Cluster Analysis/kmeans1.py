# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 19:32:34 2015

@author: scdavis6
"""
#Import modules
from sklearn import datasets
import pandas as pd
from pandas.tools.plotting import scatter_matrix


#Import dataset
iris = datasets.load_iris()
X = iris.data
Y = iris.target
#Create pandas dataframe\
#Column names
column_iris = ['Sepal Len', 'Sepal Wid', 'Petal Len', 'Petal Wid']
dataframe = pd.DataFrame(data = X, columns = column_iris)

#Create visualzations
#Scatterplot matrix
scatter_matrix(dataframe, alpha = 0.3, diagonal = 'dke')

