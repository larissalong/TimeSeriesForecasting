# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 10:09:35 2019
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import matplotlib.pyplot as plt
np.random.seed(1337) # for reproducibility


# Fetch data
# As of Aug 2019, the CSV can be downloaded as follows:
#!curl -o FremontBridge.csv https://data.seattle.gov/api/views/65db-xm6k/rows.csv?accessType=DOWNLOAD

data = pd.read_csv('FremontBridge.csv', index_col='Date', parse_dates=True)
data.head()

# For convenience, I'll further process this dataset by shortening the column names and adding a "Total" column
data.columns = ['West', 'East']
data['Total'] = data.eval('West + East')
data.dropna().describe()
data.plot()

# aggregate to weekly level
weekly = pd.DataFrame(data.resample('W').sum())
weekly.plot(style=[':', '--', '-'])
plt.ylabel('Weekly bicycle count')

weekly = weekly.loc[:, 'Total':]

##############################
# Part 1 - Data prepocessing #
##############################

# split into train & test
size = int(len(weekly) * .8)
train, test = weekly[0:size], weekly[size:len(weekly)]

# convert training set to array
training_set = train.iloc[:, 0:1].values

# Feature Scaling: X_scaled = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping to 3 dimensions: batch_size (no. of observations), timesteps, input_dim (no. of predictors)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


##############################
# Part 2 - Building the RNN #
#############################

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation: 50 neurons
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2)) # 20% of neurons to be dropped out to avoid overfit

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2)) 

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


#############################################################
# Part 3 - Making the predictions and visualising the results
#############################################################

# Getting the real units sold in test set
real = test.iloc[:, 0:1].values

# Getting the predicted value
inputs = weekly[len(weekly) - len(test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

# create 3D structure for test set
X_test = []
for i in range(60, len(inputs)):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
pred = regressor.predict(X_test)

# inverse transformation back to the actual units
pred = sc.inverse_transform(pred)

# Visualising the results
plt.plot(real, color = 'red', label = 'Real Units Sold')
plt.plot(pred, color = 'blue', label = 'Predicted Units Sold')
plt.title('SKU Units Sold Prediction')
plt.xlabel('Time')
plt.ylabel('Units Sold')
plt.legend()
plt.show()

# MAPE 
abs(real - pred).sum() / real.sum() # 15.7%
