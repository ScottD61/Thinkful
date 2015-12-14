# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 09:02:33 2015

@author: scdavis6
"""
#Import modules
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier


#Import dataset
dataframe = pd.read_csv('samsungdata.csv')

#Clean dataset 
#Remove '()'
dataframe.columns = dataframe.columns.str.replace('()', '')
#Remove '-' 
dataframe.columns = dataframe.columns.str.replace('-', '')
#Remove ','
dataframe.columns = dataframe.columns.str.replace(',', '')
#Remove extra ')' or '(' in column names 
dataframe.columns = dataframe.columns.str.replace('(', '')
dataframe.columns = dataframe.columns.str.replace(')', '')
#Remove duplicate column names
list_ = dataframe.columns
list_ = list(set(list_))
dataframe = dataframe[list_]
#Replace BodyBody with Body 
dataframe.columns = dataframe.columns.str.replace('BodyBody', 'Body')
#Remove columns with Body and Mag strings 
dataframe.columns = dataframe.columns.str.replace('Body', '')
dataframe.columns = dataframe.columns.str.replace('Mag', '')
#Change activity to a categorical variable 
#Get data type of activity
dataframe['activity'].dtype
dataframe['activity'] = dataframe['activity'].astype('category')
#Re-name categories

#Print order of categories: dataframe['activity']


#Binarize the data by category?

#Map mean and st to Mean and STD
dataframe.columns = dataframe.columns.str.replace('mean', 'Mean')
dataframe.columns = dataframe.columns.str.replace('std', 'STD')

print(dataframe.head())
print(dataframe.columns)

print(dataframe['activity'])

#Random forest

#Training, test, and evaluation datasets

#Subset data
Y = dataframe['activity']
dataframe.drop('activity', axis = 1, inplace = True)
X = dataframe

#Training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y)

#Random forest on training set
print(len(X_train))
print(len(X_test))

#Random forest on training set
clf = RandomForestClassifier(n_estimators = 500, n_jobs = 5, oob_score = True)

clf.fit(X_train, y_train)
clf.oob_score_
#0.98

#Look at feature importance
fi = enumerate(clf.feature_importances_) 
cols = X_train.columns 
[(value,cols[i]) for (i,value) in fi if value > 0.015]

#Top 10
#[(0.027840528316609634, 'angleXgravityMean'),
 #(0.015437936786369958, 'tGravityAccenergyY'),
 #(0.027844062323497051, 'tGravityAccmaxX'),
 #(0.027881186706028679, 'tGravityAccenergyX'),
 #(0.033828814471490609, 'tGravityAccminX'),
 #(0.021148696468698013, 'tGravityAccMeanY'),
 #(0.024855113474980737, 'tGravityAccminY'),
 #(0.025693516320570403, 'angleYgravityMean'),
 #(0.023618156514908759, 'tGravityAccmaxY'),
 #(0.026206501115150677, 'tGravityAccMeanX')]

#Feature 10 is 'tGravityAccMeanX'

#Roy's method for answering q4
import itertools
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score

skf = StratifiedKFold(y_train, n_folds = 5)
#### need to convert data frame to array for this to work
#### accuracy on training data
X_ = X_train.as_matrix()
y_ = np.array(y_train)
true_ = []
pred_ = []
for train_index, test_index in skf:
    clf = RandomForestClassifier(n_estimators = 500, n_jobs = 5)
    clf.fit(X_[train_index], y_[train_index])
    predict = clf.predict(X_[test_index])
    true_.append(y_[test_index])
    pred_.append(predict)


TrueLabel = list(itertools.chain(*true_))
PredictedLabel = list(itertools.chain(*pred_))

accuracy_score(TrueLabel, PredictedLabel)
## 0.978 accuracy which is very hight, lets apply the same model on test/validation data

X_validation = X_test.as_matrix()
y_validation =np.array(y_test)
pred_= clf.predict(X_validation)
accuracy_score(y_validation, pred_)
# 0.98 accuracy

#Calculate precision and recall for test data
#Recall
recall_score(y_test, pred_, average = 'weighted')
#0.976 
#Precision
precision_score(y_test, pred_, average = 'weighted')
#0.976 


#Blackbox assignment

#Reload dataset
dataframe1 = pd.read_csv('samsungdata.csv')

#Change column names to a series
column_names = ['x' + str(i) for i in range(562)]
column_names = column_names + ['subject', 'activity']
dataframe1.columns = column_names

#Split into training, test, and validation sets based on certain parameters

#Training set
train = dataframe1[dataframe1['subject'] >= 27]
#Subset training data
Y_train = train['activity']
train.drop('activity', axis = 1, inplace = True)
X_train = train

#Test set 
test = dataframe1[dataframe1['subject'] <= 6]
#Subset test setdata
Y_test = test['activity']
test.drop('activity', axis = 1, inplace = True)
X_test = test

#Validation set
validation = dataframe1[(dataframe1['subject'] >= 21) & (dataframe1['subject'] < 27)]
#Subset validation data
Y_validation = validation['activity']
validation.drop('activity', axis = 1, inplace = True)
X_validation = validation

#Random forest object
clf1 = RandomForestClassifier(n_estimators = 100, n_jobs = 5, oob_score = True)

clf1.fit(X_train, Y_train)
clf1.oob_score_
#0.991

#Look at feature importance
fi = enumerate(clf1.feature_importances_) 
cols = X_train.columns 
[(value,cols[i]) for (i,value) in fi if value > 0.0165]

#Top 10
#[(0.029125727610153446, 'x41'),
 #(0.022935043759407327, 'x42'),
 #(0.021496295837235051, 'x50'),
 #(0.028272601399421955, 'x53'),
 #(0.02340842097185783, 'x55'),
 #(0.025330141977476121, 'x57'),
 #(0.037118990937803964, 'x59'),
 #(0.027647492808085941, 'x559'),
 #(0.016670234658404129, 'x560'),
 #(0.018976800983291967, 'x561')]
 
#Feature 10 is x561
 
#Calculate accuracy for test and validation sets 
skf1 = StratifiedKFold(Y_train, n_folds = 5)
#Mean accuracy score 
#### need to convert data frame to array for this to work
#### accuracy on training data
X_1 = X_train.as_matrix()
y_1 = np.array(y_train)
true_1 = []
pred_1 = []
for train_index, test_index in skf1:
    clf = RandomForestClassifier(n_estimators = 500, n_jobs = 5)
    clf.fit(X_1[train_index], y_1[train_index])
    predict = clf1.predict(X_1[test_index])
    true_1.append(y_1[test_index])
    pred_1.append(predict)


TrueLabel1 = list(itertools.chain(*true_))
PredictedLabel1 = list(itertools.chain(*pred_))

accuracy_score(TrueLabel1, PredictedLabel1)
## 0.159 accuracy low

X_validation1 = X_test.as_matrix()
y_validation1 = np.array(Y_test)
pred_1 = clf1.predict(X_validation1)
accuracy_score(y_validation1, pred_1)
# 0.84 accuracy

#Plot confusion matrix for test data
confusion_matrix(y_validation1, pred_1)

#array([[220,   0,   0,   0,   0,   1],
#       [  0, 159,  38,   0,   0,   1],
#       [  0,  92, 135,   0,   0,   0],
#       [  0,   0,   0, 262,   4,   0],
#       [  0,   0,   0,   2, 186,   5],
#       [  0,   0,   0,   6,  59, 145]])

#Model's precision, recall, and F1 score on test set
#Recall
recall_score(Y_test, pred_1, average = 'weighted')
#0.84
#Precision
precision_score(Y_test, pred_1, average = 'weighted')
#0.86
#F1 score
f1_score(Y_test, pred_1, average = 'weighted')
#0.84
