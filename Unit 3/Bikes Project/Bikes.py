# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 16:41:38 2015

@author: scdavis6
"""
#Pt.1 Clean citibike data
#Import packages
import requests
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3 as lite
import time
import datetime
import collections
from dateutil.parser import parse


#Import data
r = requests.get('http://www.citibikenyc.com/stations/json')
#Convert json to pandas dataframe
df = json_normalize(r.json()['stationBeanList'])

#Data inspection
#View head
df.head()
#Summary statistics
df.describe()
#Number of NaN values per column
df.isnull().sum()
#Number of NaN values in entire dataset
df.isnull().values.sum()


#Visualizations
#Histogram of available bikes
df['availableBikes'].hist()
#Histogram of total docks
df['totalDocks'].hist()

#Challenge
#1) 
#a)Are there any test stations?
#Check datatype of testStations column
df['testStation'].dtype
#dtype('bool')
df['testStation'].value_counts()
#False    512
#dtype: int6
#There are no test stations

#b)How many stations are In Service?
df['statusValue'].value_counts()
#In Service        496
#Not In Service     16
#dtype: int6
#496 in service
#16 not in service

#2) 
#a)Mean number of bikes in a dock?
df['totalDocks'].mean()
#32.20898437

#b)Median number of bikes in a dock?
df['totalDocks'].median()
#31.0

#c)Mean and median number of bikes in a dock with stations not in service?
a = df[df['statusValue'] == 'Not In Service']
a['totalDocks'].mean()
#9.25 for mean 
a['totalDocks'].median()
#3.0 for median


#Pt.2 Store Citibike data
#Push to database
#Create table
con = lite.connect('citi_bike.db')
cur = con.cursor()

with con:
    cur.execute('CREATE TABLE citibike_reference (id INT PRIMARY KEY, totalDocks INT, city TEXT, altitude INT, stAddress2 TEXT, longitude NUMERIC, postalCode TEXT, testStation TEXT, stAddress1 TEXT, stationName TEXT, landMark TEXT, latitude NUMERIC, location TEXT )')

#Populate table with values
sql = "INSERT INTO citibike_reference (id, totalDocks, city, altitude, stAddress2, longitude, postalCode, testStation, stAddress1, stationName, landMark, latitude, location) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)"

#for loop to populate values in the database
with con:
    for station in r.json()['stationBeanList']:
        #id, totalDocks, city, altitude, stAddress2, longitude, postalCode, testStation, stAddress1, stationName, landMark, latitude, location)
        cur.execute(sql,(station['id'],station['totalDocks'],station['city'],station['altitude'],station['stAddress2'],station['longitude'],station['postalCode'],station['testStation'],station['stAddress1'],station['stationName'],station['landMark'],station['latitude'],station['location']))

#Change column name in table 
#extract the column from the DataFrame and put them into a list
station_ids = df['id'].tolist() 

#add the '_' to the station name and also add the data type for SQLite
station_ids = ['_' + str(x) + ' INT' for x in station_ids]

#create the table
#in this case, we're concatentating the string and joining all the station ids (now with '_' and 'INT' added)
with con:
    cur.execute("CREATE TABLE available_bikes ( execution_time INT, " +  ", ".join(station_ids) + ");")

#Populate table for available bikes
#take the string and parse it into a Python datetime object
exec_time = parse(r.json()['executionTime'])

##retry for execution time
with con:
    cur.execute('INSERT INTO available_bikes (execution_time) VALUES (?)', (exec_time.strftime('%s'),))
    
#Iterate stations in stationBeanList
id_bikes = collections.defaultdict(int) #defaultdict to store available bikes by station

#loop through the stations in the station list
for station in r.json()['stationBeanList']:
    id_bikes[station['id']] = station['availableBikes']

#iterate through the defaultdict to update the values in the database 
for k, v in id_bikes.items():
    cur.execute("UPDATE available_bikes SET _" + str(k) + " = " + str(v) + " WHERE execution_time = " + exec_time.strftime('%s') + ";")

#Roy's addition 
for i in range(1):
    r = requests.get('http://www.citibikenyc.com/stations/json')
    exec_time = parse(r.json()['executionTime'])

    cur.execute('INSERT INTO available_bikes (execution_time) VALUES (?)', (exec_time.strftime('%s'),))
    con.commit()

    id_bikes = collections.defaultdict(int)
    for station in r.json()['stationBeanList']:
        id_bikes[station['id']] = station['availableBikes']

    for k, v in id_bikes.items():
        cur.execute("UPDATE available_bikes SET _" + str(k) + " = " + str(v) + " WHERE execution_time = " + exec_time.strftime('%s') + ";")
    con.commit()

    time.sleep(60)

con.close() #close the database connection when done


#OperationalError: no such column: _station

#Pt.3 Analyze an hour of citibike data
#Upload data to pandas dataframe from SQLite database

#Connect to database
con = lite.connect('citi_bike.db')
cur = con.cursor()

#Read in data
df = pd.read_sql_query("SELECT * FROM available_bikes ORDER BY execution_time", con, index_col = 'execution_time')
#Drop NA values
df = df.dropna()

#Check rows being updated
print(df)

#Calculate difference between minutes
hour_change = collections.defaultdict(int)
for col in df.columns:
    station_vals = df[col].tolist()
    station_id = col[1:] #trim the "_"
    station_change = 0
    for k,v in enumerate(station_vals):
        if k < len(station_vals) - 1:
            station_change += abs(station_vals[k] - station_vals[k+1])
    hour_change[int(station_id)] = station_change #convert the station id back to integer

#Find key in dictionary with the greatest value
def keywithmaxval(d):
    """Find the key with the greatest value"""
    return max(d, key=lambda k: d[k])

# assign the max key to max_station
max_station = keywithmaxval(hour_change)

#Summary statistics on most active station
#query sqlite for reference information
cur.execute("SELECT id, stationname, latitude, longitude FROM citibike_reference WHERE id = ?", (max_station,))
data = cur.fetchone()
print("The most active station is station id %s at %s latitude: %s longitude: %s " % data)
print("With %d bicycles coming and going in the hour between %s and %s" % (
    hour_change[max_station],
    datetime.datetime.fromtimestamp(int(df.index[0])).strftime('%Y-%m-%dT%H:%M:%S'),
    datetime.datetime.fromtimestamp(int(df.index[-1])).strftime('%Y-%m-%dT%H:%M:%S'),
))
#The most active station is station id 492 at W 33 St & 7 Ave latitude: 40.75019995 longitude: -73.99093085 
#With 3 bicycles coming and going in the hour between 2015-11-23T16:09:51 and 2015-11-23T16:10:1

print(max_station)
#492

#Plot data of hour changes
plt.bar(hour_change.keys(), hour_change.values())
plt.show()

