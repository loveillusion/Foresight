import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import csv
import joblib

province = input("Enter Alpha Code of Province: ")
provinces = ['Header', 'AB', 'BC', 'MB', 'NB', 'NL', 'NS', 'ON', 'PE', 'QC', 'SK']
columns = ['prov', 'min_NoH', 'max_NoH', 'min_Pop', 'max_Pop', 'min_NoB', 'max_NoB', 'min_Disc', 'max_Disc']

# Get Required Row:
maxmin = open('Foresight/datasets/maxmin.csv')
maxmin = csv.reader(maxmin)
rows = []
for row in maxmin:
        rows.append(row)
maxmin = rows[provinces.index(province)]

# Loading Models:
m1 = joblib.load('Foresight/models/' + province + '/m1.sav')
m2 = joblib.load('Foresight/models/' + province + '/m2.sav')

# Getting Inputs:
NoH = int(input("Enter of Homes: "))
Pop = int(input("Enter Population: "))
print()

# Output R2 & CV Scores:
print("R Squared Score: ", R2scores[provinces.index(province)])
print("Cross Validation Mean: ", CVscores[provinces.index(province)])
print()

# Normalization:
NoH = (NoH - int(maxmin[columns.index('min_NoH')])) / (int(maxmin[columns.index('max_NoH')]) - int(maxmin[columns.index('min_NoH')]))
Pop = (Pop - int(maxmin[columns.index('min_Pop')])) / (int(maxmin[columns.index('max_Pop')]) - int(maxmin[columns.index('min_Pop')]))

# Run Prediction:
dataframe1 = pd.DataFrame([[NoH, Pop]], columns = ['Number of Homes', 'Population'])
NoB = (m1.predict(dataframe1))[0][0]
dataframe2 = pd.DataFrame([[NoH, Pop, NoB]], columns = ['Number of Homes', 'Population', 'Number of Beds'])
Disc = (m2.predict(dataframe2))[0][0]

# Find Actual Prediction:
NoB = NoB * (int(maxmin[columns.index('max_NoB')]) - int(maxmin[columns.index('min_NoB')])) + int(maxmin[columns.index('min_NoB')])
Disc = Disc * (int(maxmin[columns.index('max_Disc')]) - int(maxmin[columns.index('min_Disc')])) + int(maxmin[columns.index('min_Disc')])
prediction = int(NoB + Disc)

# Output Values of M1, M2 & Prediction:
print("Predicted Number of Beds: ", NoB)
print("Predicted Adjustment: ", Disc)
print("Total Number of Beds Needed (with Adjustment): ", prediction) 
