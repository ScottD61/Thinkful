# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 20:29:00 2015

@author: scdavis6
"""
#Import modules
import pandas as pd
from sklearn.preprocessing import Imputer
from scipy.cluster.vq import kmeans, vq, whiten
from pylab import plot, show
import numpy as np

#Import dataset
dataframe = pd.read_csv('un.csv')

#Data preparation
#Number of rows and columns
dataframe.shape
#207 rows, 14 columns

#Number of non-null values in each column
dataframe.notnull().sum()
#country                   207
#region                    207
#tfr                       197
#contraception             144
#educationMale              76
#educationFemale            76
#lifeMale                  196
#lifeFemale                196
#infantMortality           201
#GDPperCapita              197
#economicActivityMale      165
#economicActivityFemale    165
#illiteracyMale            160
#illiteracyFemale          160

#Datatype of each column
dataframe.dtypes
#country                    object
#region                     object
#tfr                       float64
#contraception             float64
#educationMale             float64
#educationFemale           float64
#lifeMale                  float64
#lifeFemale                float64
#infantMortality           float64
#GDPperCapita              float64
#economicActivityMale      float64
#economicActivityFemale    float64
#illiteracyMale            float64
#illiteracyFemale          float64

#Number of different countries
dataframe.country.nunique()
#207

#Take out columns with string values
#Subset dataframe 
#df1 = dataframe[['lifeMale', 'lifeFemale', 'infantMortality', 'GDPperCapita']]
df1 = dataframe[['GDPperCapita', 'lifeMale']]


#Imputation of mean
#Create imputation object
imp = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#Fit the data
imp.fit(df1)
#Fill in the data
df2 = imp.transform(df1)

#Convert array to dataframe
df3 = pd.DataFrame(df2)
cluster1 = df3.values
 # must be called prior to passing an observation matrix to kmeans. Normalize a group of observations on a per feature basis.
cluster1 = whiten(cluster1)
centroids1,dist1 = kmeans(cluster1, 2)
idx1,idxdist1 = vq(cluster1, centroids1)

#Plot Male Life Expectancy vs GDP Per Capita
plot(cluster1[idx1 == 0,0], cluster1[idx1 == 0, 1], 'ob',
     cluster1[idx1 == 1,0], cluster1[idx1 == 1, 1], 'or')
plot(centroids1[:,0], centroids1[:,1], 'sg', markersize = 8)
show()

#Elbow Method
cluster1 = whiten(cluster1)
average_distance = []
for k in range(1,11):
    centroids1,dist1 = kmeans(cluster1,k) # you can calculate the eucledean distance in the next line
    idx1,idxdist1 = vq(cluster1,centroids1)
    avg_dist = np.mean(idxdist1)
    average_distance.append(avg_dist)
# Just plotting the mean distance, you can plot Euclidian distance once you update the code
plot(range(1,11), average_distance)

#Choose 3 clusters