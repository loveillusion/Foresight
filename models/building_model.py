# CREATING DATAFRAMES, NORMALIZING & SPLITTING SETS

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

province = input("Alpha Code of Province: ")
df = pd.read_csv('datasets/' + province + '.csv')
scaler = preprocessing.MinMaxScaler()
m1_X = df[['Number of Homes', 'Population']]
m1_y = df['Number of Beds']
m2_X = df[['Number of Homes', 'Population', 'Number of Beds']]
m2_y = df['Discrepancy']

#print(m1_X)
#print(m1_y)
#print(m2_X)
#print(m2_y)

m1_scaled_X = scaler.fit_transform(m1_X)
m1_scaled_X = pd.DataFrame(m1_scaled_X, columns = ['Number of Homes', 'Population'])
#print(m1_scaled_X)
m1_scaled_y = preprocessing.normalize([m1_y])
m1_scaled_y = pd.DataFrame(m1_scaled_y[0], columns = ['Number of Beds'])
#print(m1_scaled_y)

m2_scaled_X = scaler.fit_transform(m2_X)
m2_scaled_X = pd.DataFrame(m2_scaled_X, columns = ['Number of Homes', 'Population', 'Number of Beds'])
#print(m2_scaled_X)
m2_scaled_y = preprocessing.normalize([m2_y])
m2_scaled_y = pd.DataFrame(m2_scaled_y[0], columns = ['Discrepancy'])
#print(m2_scaled_y)

m1_X_train, m1_X_test, m1_y_train, m1_y_test = train_test_split(m1_scaled_X, m1_scaled_y, test_size = 1/3)
m2_X_train, m2_X_test, m2_y_train, m2_y_test = train_test_split(m2_scaled_X, m2_scaled_y, test_size = 1/3)



# BUILDING ML MODEL & MEASURING ACCURACY

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import statistics
from sklearn.metrics import r2_score

m1_lr = LinearRegression()
m1_fit = m1_lr.fit(m1_X_train, m1_y_train)
m1_y_pred = m1_lr.predict(m1_X_test)
#print(m1_y_pred)
scores = cross_val_score(m1_fit, m1_X, m1_y, cv=5)
print(scores)
print(statistics.mean(scores)*100)

m1_r2 = r2_score(m1_y_test, m1_y_pred)
print(m1_r2)
sum = m1_r2

m1_y_pred = m1_lr.predict(m1_X_train)
m1_r2 = r2_score(m1_y_train, m1_y_pred)
print(m1_r2)
sum += m1_r2
print('r2:', (sum/2)*100)

m2_lr = LinearRegression()
m2_fit = m2_lr.fit(m2_X_train, m2_y_train)
m2_y_pred = m2_lr.predict(m2_X_test)
#print(m2_y_pred)
scores = cross_val_score(m2_fit, m2_X, m2_y, cv=5)
print(scores)

m2_r2 = r2_score(m2_y_test, m2_y_pred)
print(m2_r2)

m2_y_pred = m2_lr.predict(m2_X_train)
m2_r2 = r2_score(m2_y_train, m2_y_pred)
print(m2_r2)



# SAVING THE MODELS

import joblib
m1_file = 'm1.sav'
m2_file = 'm2.sav'
joblib.dump(m1_lr, m1_file)
joblib.dump(m2_lr, m2_file)
