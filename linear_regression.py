# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 13:18:26 2015

@author: scdavis6
"""
#Load packages
import pandas as pd


#Load dataset
loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')

#Observe dataset
print(loansData['Interest.Rate'][0:5])
print(loansData['Loan.Length'][0:5])
print(loansData['FICO.Range'][0:5])

#Data preparation
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
#Convert to list



loansData['FICO.Score'] = loansData['FICO.Range'].map(lambda x: int(x[0]))


#loansData['FICO.Range'].values[0]
#Check conversion
#type(loansData['FICO.Range'].values[0])
#Convert to string
#loansData['FICO.Range'].values[0][0]
#Check conversion
#type(loansData['FICO.Range'].values[0][0])
#Convert each string in list to integer
#loansData['FICO.Range'] = loansData['FICO.Range'].map(lambda x: [int(n) for n in x])


#Conversion did not work
print(loansData['FICO.Score'][0:5])


#81174    735-739
#99592    715-719
#80059    690-694
#15825    695-699
#33182    695-699
#Name: FICO.Range, dtype: object
