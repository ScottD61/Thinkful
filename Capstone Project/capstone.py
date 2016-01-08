# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 10:06:04 2016

@author: scdavis6
"""
#Import modules
import pandas as pd
from pandas.tools.plotting import scatter_matrix

#Import data
dataframe = pd.read_csv('German.csv')

#Explore dataframe
#Dimension of dataset
dataframe.shape
#(1000, 21)
#Column datatypes
dataframe.dtypes
#Number of NaN values in each column
dataframe.notnull().sum()
#None

#Data exploration
#Summary statistics
dataframe.describe()
#Scatterplot matrix - what's the right size?
#Subset data - numeric variables
#df = dataframe[['Duration', 'Credit amount', 'Installment rate', 'Present resident since',
#                'Age', 'Number existing credits', 'Number of people liable']]
#Scatterplot matrix
#scatter_matrix(df, alpha = 0.3, diagonal = 'dke')

#Correlation between numeric
#Anova between categorical and numeric
#Chi-squared test between all unordered categorical

#Data cleaning
#Change data type of classification
dataframe['Classification'] = dataframe['Classification'].astype('category')
#Change factors of all categorical variables??

#Model building
#Subset dataframe into indepedent and dependent variables
X = dataframe.drop('Classification', axis = 1)
Y = dataframe['Classification']

#Things to vary:
    #Training set size
        #Compare different classification methods with learning curves
    #Parameter values
        #Grid search

#Grid search for multiple parameters for several classifiers
parameters = [
    {'weights': ['uniform', 'distance'], 'n_neighbors': [5, 10, 15, 20, 25, 30, 35]}, ###KNN
    {'kernel': ('linear', 'rbf'), 'C':[1, 10]}, ###SVM
    {'n_estimators': [100, 200, 250, 300, 350, 400, 450, 500, 550]}, ###random forest?? idk if will work
    #Logistic regression
    
    