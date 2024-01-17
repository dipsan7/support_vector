import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#  IMPORTING DATASET 
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)
y = y.reshape(len(y),1)
print(y)
# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_y=StandardScaler()
X=sc_X.fit_transform(X)
y=sc_y.fit_transform(y)
print(X)
print(y)
# TRAIINING SVR MODEL
from sklearn.svm import SVR
regressor =SVR(kernel="rbf")
regressor.fit(X,y)
# PREDICTING THE RESULT
sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1,1))
# VISUALISING SVR RESULT
plt.scatter(sc_X.inverse_transform(X),sc_y.inverse_transform(y),color="red")
plt.plot(sc_X.inverse_transform(X),sc_y.inverse_transform(regressor.predict(X).reshape(-1,1)),color="blue")
plt.title("SVR model")
plt.xlabel("position lavel")
plt.ylabel("salary")
plt.show()