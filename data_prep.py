# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 21:54:30 2021

@author: wrace
"""

### Library Import ###

import pandas as pd
import datetime as dt
from scipy import stats
from sklearn.preprocessing import StandardScaler

### Data Import and Sample ###

dataRaw = pd.read_excel("Data\\Online_Retail.xlsx") # read raw data

dataRaw = dataRaw[dataRaw['CustomerID'].notna()] # remove NA customer IDs

dataRaw = dataRaw[~dataRaw.InvoiceNo.str.contains("C", na = False)] # remove invoices containing "C", denotes cancellations

dataSample = dataRaw.sample(10000, random_state = 42) # samples randomly from the raw data, random seed 42


### Variable Processing ###

dataSample["InvoiceDate"] = dataSample["InvoiceDate"].dt.date # sets InvoiceDate column to date format

dataSample["TotalSum"] = dataSample["Quantity"] * dataSample["UnitPrice"] # adds new column TotalSum, total cost of purchase

snapshotDate = max(dataSample.InvoiceDate) + dt.timedelta(days = 1) # stores most recent date for later calculations

# group by CustomerID, aggregate selected column
customerData = dataSample.groupby(['CustomerID']).agg({
    'InvoiceDate': lambda x: (snapshotDate - x.max()).days,
    'InvoiceNo': 'count',
    'TotalSum': 'sum'
    })

# rename columns in customerData, inplace returns a data frame
customerData.rename(columns = {'InvoiceDate': 'TimeSincePurchase',
                               'InvoiceNo': 'PurchaseCount',
                               'TotalSum': 'PurchaseValue'}, inplace = True)


### Statistical Processing ###

stats.skew(customerData) # assess skewness

# apply boxcox transformation to remedy skewness
customerBoxcox = customerData.apply(lambda col: stats.boxcox(col)[0])
stats.skew(customerBoxcox)

# scale variables
scaler = StandardScaler() # initialise scaler
scaler.fit(customerBoxcox) # fit scaler
customerScaled = customerBoxcox # create new dataframe for scaled data
customerScaled[customerScaled.columns] = scaler.transform(customerScaled) # apply scaling transformation
