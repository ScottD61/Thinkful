# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 11:08:25 2015

@author: scdavis6
"""
#Import modules
from sklearn.cross_validation import train_test_split, cross_val_score
import sklearn.linear_model as ln
import pandas as pd 
import numpy as np

#Load dataset
loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')

#Data preparation

#Check for NaN values
loansData.isnull().values.sum()
#Remove NaN values
loansData.dropna()
#Remove '%' symbols in interest rate
loansData['Interest.Rate'] = loansData['Interest.Rate'].map(lambda x: round(float(x.rstrip('%')) / 100, 4))
#Inspect results
print(loansData['Interest.Rate'][0:5])
#Remove month string in loan length
loansData['Loan.Length'] = loansData['Loan.Length'].map(lambda x: int(x.rstrip(' months')))
#Inspect results
print(loansData['Loan.Length'][0:5])
#Convert FICO range to a score
#Replace range with start and end numbers in list
loansData['FICO.Range'] = loansData['FICO.Range'].map(lambda x: x.split('-'))
#Replace start and end numbers in list with start number - problem not converting to a string
#Create variable FICO.Score as first item in list to integer
loansData['FICO.Score'] = loansData['FICO.Range'].map(lambda x: int(x[0]))

#Split into training and test sets

#Subset data into independent and dependent
X1 = loansData[['Amount.Requested', 'FICO.Score']]
Y1 = loansData['Interest.Rate']
#Training and test sets
X_train, X_test, y_train, y_test = train_test_split(X1, Y1)
#Size of training and test sets
X_train.shape
#(1875, 2)
X_test.shape
#(625, 1)
y_train.shape
#(1875, 2)
y_test.shape
#(625, 1)

#Model building 

#Other OLS method with scikitlearn
#Create linear regression object
regr = ln.LinearRegression()
#Train model with training sets
regr.fit(X_train, y_train)

#Evaluation

#Cross validation score for MSE 10 folds
MSE_Scores = cross_val_score(regr, X_train, y_train, scoring = 'mean_squared_error', cv = 10)
#Take average of all cross validation folds
np.mean(MSE_Scores)
#-0.0006163776021605567
#Cross validation score for MAE 10 fold
MAE_Scores = cross_val_score(regr, X_train, y_train, scoring = 'mean_absolute_error', cv = 10)
#Take average of all cross validation folds
np.mean(MAE_Scores)
#-0.01959622777354547
#Cross validation score for R2 10 folds
R2_Scores = cross_val_score(regr, X_train, y_train, scoring = 'r2', cv = 10)
#Take average of all cross validation folds
np.mean(R2_Scores)
#0.6505887138996289

