# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 17:37:59 2015

@author: scdavis6
"""
#Import modules
import numpy as np
import pandas as pd 
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error

#Generate dataset
#Set seed
np.random.seed(414)

#Gen toy data
X = np.linspace(0, 15, 1000)
y = 3 * np.sin(X) + np.random.normal(1 + X, .2, 1000)

train_X, train_y = X[:700], y[:700]
test_X, test_y = X[700:], y[700:]

train_df = pd.DataFrame({'X': train_X, 'y': train_y})
test_df = pd.DataFrame({'X': test_X, 'y': test_y})

#Model Building
#Numpy objects
#Linear regression
poly_1 = smf.ols(formula = 'y ~ 1 + X', data = train_df).fit()
#Quadratic 
poly_2 = smf.ols(formula = 'y ~ 1 + X + I(X**2)', data = train_df).fit()

#Test
#Linear model
ols_predict = poly_1.predict(train_df)
#Quadratic model
quad_predict = poly_2.predict(train_df)

#Evaluation with training mean-square error (MSE)
#Linear model
mean_squared_error(train_df['y'], ols_predict)
#4.056
#Quadratic model
mean_squared_error(train_df['y'], quad_predict)
#3.79
