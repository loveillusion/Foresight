import numpy as np
import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn import preprocessing


df = pd.read_csv('/content/drive/MyDrive/Borealis/Ontario.csv')
scaler = preprocessing.MinMaxScaler()
X = df[['Number of Homes', 'Population']]
y = df['Number of Beds']
print(X)
print(y)

scaled_X = scaler.fit_transform(X)
scaled_X = pd.DataFrame(scaled_X, columns = ['Number of Homes', 'Population'])
print(scaled_X)
scaled_y = preprocessing.normalize([y])
scaled_y = pd.DataFrame(scaled_y[0], columns = ['Number of Beds'])
print(scaled_y)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, scaled_y, test_size = 1/3)


lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
#print(y_pred)

r2 = r2_score(y_test, y_pred)
print(r2)

y_pred = lr.predict(X_train)
r2 = r2_score(y_train, y_pred)
print(r2)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plt.rcParams["figure.figsize"] = [12.00, 8.00]
plt.rcParams["figure.autolayout"] = True

ax.set_xlabel("Population")
ax.set_ylabel("NoH")
ax.set_zlabel("NoB")

x1 = X_train['Population'].values
z1 = y_train.values
y1 = X_train['Number of Homes'].values

x2 = X_test['Population'].values
z2 = y_test.values
y2 = X_test['Number of Homes'].values

'''
coefs = lr.coef_
intercept = lr.intercept_
print(coefs, intercept)
xs = np.tile(np.arange(61), (61,1))
ys = np.tile(np.arange(61), (61,1)).T
zs = xs*coefs[0]+zs*coefs[1]+intercept
print("Equation: y = {:.2f} + {:.2f}x1 + {:.2f}x2".format(intercept, coefs[0], coefs[1]))
'''

ax.scatter(x1, y1, z1, c='r', marker='x')
ax.scatter(x2, y2, z2, c = 'b', marker = 'x')
#ax.plot_surface(xs,ys,zs, alpha=0.5)
plt.show()
