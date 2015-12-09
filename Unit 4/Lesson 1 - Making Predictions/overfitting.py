# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 17:37:59 2015

@author: scdavis6
"""
#Import modules
import numpy as np
import pandas as pd 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import sklearn.linear_model as ln

#Gen toy data
#Set seed
np.random.seed(414)

X = np.linspace(0, 15, 1000)
y = 3 * np.sin(X) + np.random.normal(1 + X, .2, 1000)

train_X, train_y = X[:700], y[:700]
test_X, test_y = X[700:], y[700:]

#Convert numpy array as dataframe
train_X = pd.DataFrame(train_X)
train_y = pd.DataFrame(train_y)

#Inspect data
train_X.head()
train_y.head()

#Scikit learn
#Linear model 
#Create linear regression object
regr = ln.LinearRegression()
#Train model with training sets
regr.fit(train_X, train_y)

#Quadratic model
#Add polynomial features to train_X
poly = PolynomialFeatures(degree = 2)
X_ = poly.fit_transform(train_X)
#Regression object
polyregr = ln.LinearRegression()
#Train model with polynomial transformed training set
polyregr.fit(X_, train_y)

#Predict
#Linear model
ols_predict = regr.predict(train_X)
#Polynomial model
poly_predict = polyregr.predict(X_)

#Evaluation with training mean-square error (MSE)
#Linear model
mean_squared_error(train_y, ols_predict)
# 4.0349438280411309

#Polynomial model
mean_squared_error(train_y, poly_predict)
# 3.7754092151262766

#Better model - polynomial regression because of smaller MSE