# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 12:37:48 2015

@author: scdavis6
"""
#Import modules
import requests
import sqlite3 as lite
import datetime
import pandas as pd


#Create dictionary of values
cities = { "Atlanta": '33.762909,-84.422675',
            "Austin": '30.303936,-97.754355',
            "Boston": '42.331960,-71.020173',
            "Chicago": '41.837551,-87.681844',
            "Cleveland": '41.478462,-81.679435'
        }
        
APIKEY = '773ab05028df16ae814586921d6d6b4e'       
        
#API call for first city
url = 'https://api.forecast.io/forecast/' + APIKEY + '/'
#Successful pasting into URL and getting JSON format data

#Create datetime objects to iterate
end_date = datetime.datetime.now()

 


#Create table in SQLite
con = lite.connect('weather.db')
cur = con.cursor()

cities.keys()
with con:
    cur.execute('CREATE TABLE daily_temp ( day_of_reading INT, Atlanta REAL, Austin REAL, Boston REAL, Chicago REAL, Cleveland REAL);') 

#Insert rows
#Inserting only 30 days
query_date = end_date - datetime.timedelta(days = 30) #the current value being processed

with con:
    while query_date < end_date:
        cur.execute("INSERT INTO daily_temp(day_of_reading) VALUES (?)", (int(query_date.strftime('%s')),))
        query_date += datetime.timedelta(days=1)

#Loop through cities and query API

for k,v in cities.items():
    query_date = end_date - datetime.timedelta(days = 30) #set value each time through the loop of cities
    while query_date < end_date:
        #query for the value
        r = requests.get(url + v + ',' +  query_date.strftime('%Y-%m-%dT12:00:00'))

        with con:
            #insert the temperature max to the database
            cur.execute('UPDATE daily_temp SET ' + k + ' = ' + str(r.json()['daily']['data'][0]['temperatureMax']) + ' WHERE day_of_reading = ' + query_date.strftime('%s'))

        #increment query_date to the next day for next operation of loop
        query_date += datetime.timedelta(days = 1) #increment query_date to the next day


#Check database
#Query database and put into pandas dataframe        
df1 = pd.read_sql("SELECT * FROM daily_temp", con)
print(df1)        
        
con.close() # a good practice to close connection to database

#Analyzing data
#Atlanta
#Range of temperatures 
max(df1['Atlanta']) - min(df1['Atlanta'])
#26.2
#Mean temperature for each city
df1['Atlanta'].mean()
#67.3683333333333
#Variance of temperature for each city
df1['Atlanta'].var()
#42.5478557471264
#Greatest temperature changes over time
df1['Atlanta'].diff().max()
#20.47000000000000
#Identify row with biggest change
df1['Atlanta'].diff().idxmax()
#8
        
#Boston
#Range of temperatures 
max(df1['Boston']) - min(df1['Boston'])
#29.74000000000000
#Mean temperature for each city
df1['Boston'].mean()
#58.6286666666666
#Variance of temperature for each city
df1['Boston'].var()  
#67.764922298850   
#Greatest temperature changes over time
df1['Boston'].diff().max()
#14.66000000000000
#Identify row with biggest change
df1['Boston'].diff().idxmax()
#16

#Chicago
#Range of temperatures 
max(df1['Chicago']) - min(df1['Chicago'])
#29.18000000000000
#Mean temperature for each city
df1['Chicago'].mean()
#60.44933333333333
#Variance of temperature for each city
df1['Chicago'].var()    
#61.2584891954023      
#Greatest temperature changes over time
df1['Chicago'].diff().max()
#10.98000000000000
#Identify row with biggest change
df1['Chicago'].diff().idxmax()  
#12
        
#Cleveland
#Range of temperatures 
max(df1['Cleveland']) - min(df1['Cleveland'])
#28.99000000000000
#Mean temperature for each city
df1['Cleveland'].mean()
#60.9126666666666
#Variance of temperature for each city
df1['Cleveland'].var()     
#67.385634022988
#Greatest temperature changes over time
df1['Cleveland'].diff().max()
#15.78000000000000
#Identify row with biggest change
df1['Cleveland'].diff().idxmax()   
#26      
