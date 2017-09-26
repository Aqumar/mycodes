# Part 1 - Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)


# Creating a data structure with 20 timesteps and t+1 output

X_train = []
y_train = []

for i in range(20,len(training_set_scaled)):
    X_train.append(training_set_scaled[i-20:i,0])
    y_train.append(training_set_scaled[i,0])

X_train,y_train = np.array(X_train),np.array(y_train)

# Reshaping
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Initialising the RNN
regressor = Sequential()

# Adding the input layer and the LSTM layer
regressor.add(LSTM(units = 3, return_sequences = True, input_shape = (None, 1)))

# Adding the second LSTM layer 
regressor.add(LSTM(units = 3, return_sequences = True))

# Adding the third LSTM layer 
regressor.add(LSTM(units = 3, return_sequences = True))

# Adding the fourth LSTM layer 
regressor.add(LSTM(units = 3))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


# Part 3 - Making the predictions and visualising the results

# Getting the real stock price for February 1st 2012 - January 31st 2017

dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')

test_set = dataset_test.iloc[:,1:2].values

real_stock_price = np.concatenate((training_set[0:len(training_set)],test_set),axis=0)

# Getting the predicted stock price of 2017

scaled_real_stock_price = sc.fit_transform(real_stock_price)

inputs = []

for i in range(len(training_set),len(real_stock_price)):
    inputs.append(scaled_real_stock_price[i-20:i,0])
    
inputs = np.array(inputs)

inputs = np.reshape(inputs,(inputs.shape[0],inputs.shape[1],1))

predicted_stocck_price = regressor.predict(inputs)

predicted_stocck_price = sc.inverse_transform(predicted_stocck_price)

plt.plot(real_stock_price[len(training_set):],color='red',label = 'Real Google Stock Price')

plt.plot(predicted_stocck_price,color='blue',label='Predicted Google Stock Price')

plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()