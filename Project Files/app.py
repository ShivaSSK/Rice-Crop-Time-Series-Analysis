import joblib

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
data = pd.read_csv('rice_crop_data.csv')
data.head()
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Value'].values.reshape(-1, 1))
time_steps = 5
X, y = [], []
for i in range(time_steps, len(scaled_data)):
    X.append(scaled_data[i-time_steps:i, 0])
    y.append(scaled_data[i, 0])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=100, return_sequences=True))
model.add(LSTM(units=100))
model.add(Dense(units=1))
import tensorflow as tf
optimizer = tf.keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')
model.fit(X, y, epochs=50, batch_size=32)
country_name = input('Enter the country name: ')
year = int(input('Enter the year: '))

country_data = data[(data['Area'] == country_name) & (data['Year'] <= year)]
inputs = country_data['Value'].values[-time_steps:]
inputs = scaler.transform(inputs.reshape(-1, 1))
inputs = np.array(inputs).reshape(1, time_steps, 1)
predicted_value = model.predict(inputs)
predicted_value = scaler.inverse_transform(predicted_value)
print('Predicted rice crop production for', country_name, 'in', year, ':', predicted_value[0][0], 'tonnes')
import matplotlib.pyplot as plt
country_data = data[data['Area'] == country_name]
plt.plot(country_data['Year'], country_data['Value'], label='Actual')
plt.scatter(year, predicted_value[0][0], color='red', label='Predicted')
plt.xlabel('Year')
plt.ylabel('Value (tonnes)')
plt.title('Rice Crop Production: ' + country_name)
plt.legend()
plt.show()
from sklearn.metrics import mean_squared_error
from math import sqrt

actual_values = country_data['Value'].values[-len(inputs):]

actual_values = np.reshape(actual_values, (len(actual_values), 1))
actual_values = scaler.inverse_transform(actual_values)

rmse = sqrt(mean_squared_error(actual_values, predicted_value))
print('Root Mean Squared Error (RMSE):', rmse)

# Train your model
model.fit(X, y, epochs=50, batch_size=32)

# Save the model as a .pkl file
model_filename = 'rice_crop_model.pkl'
joblib.dump(model, model_filename)
