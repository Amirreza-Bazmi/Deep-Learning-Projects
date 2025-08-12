import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Load CSV file
df = pd.read_csv("your-dataset.csv")

# Data preprocessing
price = df[["Date", "Close"]]
price = price[price["Date"] >= "2021-01-01"]
close_price = price["Close"].values

# Normalizing the data
scaler = MinMaxScaler(feature_range=(0, 1))
close_price = scaler.fit_transform(close_price.reshape(-1, 1))

# Function to split data into sequences
def spliter(data, size):
    x, y = [], []
    for i in range(len(data) - size):
        x.append(data[i: i + size, 0])
        y.append(data[i + size, 0])
    x = np.array(x)
    y = np.array(y)
    x = x[..., tf.newaxis]  # Add new axis for LSTM input compatibility
    return x, y

# Splitting the data into training and testing sets
train_size = int(len(close_price) * 0.8)
X_train, X_test = close_price[:train_size], close_price[train_size:]

# Splitting the data into sequences
x_train, y_train = spliter(X_train, 20)  # Using a window size of 20
x_test, y_test = spliter(X_test, 20)

# Define the model
model = Sequential()
model.add(LSTM(100, input_shape=(x_train.shape[1], 1), return_sequences=True, dropout=0.2))
model.add(LSTM(50, dropout=0.2))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
print(model.summary())

# Training the model with early stopping
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(x_train, y_train, epochs=200, validation_data=(x_test, y_test))

# Prediction
train_predicted = model.predict(x_train)
test_predicted = model.predict(x_test)

# Converting predictions back to original scale
train_predicted = scaler.inverse_transform(train_predicted)
test_predicted = scaler.inverse_transform(test_predicted)
original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate RMSE and MAE
train_rmse = np.sqrt(mean_squared_error(original_ytrain, train_predicted))
test_rmse = np.sqrt(mean_squared_error(original_ytest, test_predicted))
train_mae = mean_absolute_error(original_ytrain, train_predicted)
test_mae = mean_absolute_error(original_ytest, test_predicted)

print(f"Train RMSE: {train_rmse}, Test RMSE: {test_rmse}")
print(f"Train MAE: {train_mae}, Test MAE: {test_mae}")

# Plotting results
plt.figure(figsize=(12, 6))
plt.plot(original_ytrain, label='Actual Train')
plt.plot(train_predicted, label='Predicted Train')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(original_ytest, label='Actual Test')
plt.plot(test_predicted, label='Predicted Test')
plt.legend()
plt.show()
