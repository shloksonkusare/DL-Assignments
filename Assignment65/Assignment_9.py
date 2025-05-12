import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# 1. Load Stock Price Data
ticker = 'AAPL'  # You can change this to any stock symbol
data = yf.download(ticker, start='2018-01-01', end='2023-12-31')
closing_prices = data['Close'].values.reshape(-1, 1)

# 2. Normalize the Data
scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(closing_prices)

# 3. Create Sequences
def create_sequences(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i + time_step])
        y.append(data[i + time_step])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_sequences(scaled_prices, time_step)

# 4. Split into Train and Test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 5. Define the RNN Model
model = Sequential([
    SimpleRNN(50, return_sequences=False, input_shape=(time_step, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# 6. Train the Model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# 7. Predict and Plot
y_pred = model.predict(X_test)
y_pred_scaled = scaler.inverse_transform(y_pred)
y_test_scaled = scaler.inverse_transform(y_test)

# 8. Plot the Results
plt.figure(figsize=(12, 6))
plt.plot(y_test_scaled, label='Actual Price')
plt.plot(y_pred_scaled, label='Predicted Price')
plt.title(f"{ticker} Stock Price Prediction using RNN")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.show()
