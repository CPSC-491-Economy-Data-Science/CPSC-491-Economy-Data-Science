# Time Series Training Python Example for CPSC-491 Final Project
# This high-level script has been implemented for use of predicting stock closing prices
# of the S&P 500 (Yahoo finance data) from 12/21-12/22
# References to TensorFlow's example + our own implementation can be found below (referenced in our documentation)

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pand
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

# Importing GPSC data set from 12/21-12/22
data = pand.read_csv('./data/GSPC_data.csv')

# Verify our data is imported correctly
print(data.shape)
print(data.sample(5))

# Verify data set columns and values
data.info()

# TENSORFLOW - Convert Date from Yahoo Finance records to 'dateTime' for Pandas
data['Date'] = pand.to_datetime(data['Date'])

# Verify date column has changed
data.info()

# Initial plot of our data set
# Plot is Date/Close as Close shows the value of the stock on a particular day at close time
plt.plot(data['Date'], data['Close'])
plt.xlabel("Date")
plt.ylabel("Close")
plt.title("S&P 500 Prices 2022")
plt.show()

# Prepare training set
close_data = data.filter(['Close'])
dataset = close_data.values
training = int(np.ceil(len(dataset) * .95))

# Output the length of our training set
print(training)


scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:int(training), :]
# prepare feature and labels
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Prepare our LSTM Model (sequential model base)
model = keras.models.Sequential()
model.add(keras.layers.LSTM(units=64,
                            return_sequences=True,
                            input_shape=(x_train.shape[1], 1)))
model.add(keras.layers.LSTM(units=64))
model.add(keras.layers.Dense(32))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))

# Output for the model - verifies all layers added are true
model.summary

model.compile(optimizer='adam',
              loss='mean_squared_error')

# TRAINING - Fitting our model based on our x and y arrays.
# With each epoch, loss will be output. Loss is our greatest determinant
# to see how well our model fits our testing set
history = model.fit(x_train,
                    y_train,
                    epochs=500)

# Creating our Testing subset
test_data = scaled_data[training - 60:, :]
x_test = []
y_test = dataset[training:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Predicting testing data
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# MSE/RMSE - Our evaluation metrics (aside from loss) on how well our model fits
mse = np.mean(((predictions - y_test) ** 2))
print("MSE", mse)
print("RMSE", np.sqrt(mse))

# Prepare for plot
train = data[:training]
test = data[training:]
test['Predictions'] = predictions

# Plotting our final output between train, test, and predictions
plt.figure(figsize=(15, 8))
plt.plot(train['Close'])
plt.plot(test[['Close', 'Predictions']])
plt.title('S&P Stock Close Price')
plt.xlabel('Date')
plt.ylabel("Close")
plt.legend(['Train', 'Test', 'Predictions'])
plt.show()
