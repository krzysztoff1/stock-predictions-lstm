import yfinance as yf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt


stock_data = yf.download("DAC", start="2014-02-21", end="2024-02-21")
stock_data


scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data["Close"].values.reshape(-1, 1))


def create_dataset(data, time_step):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i : (i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)


time_step = 100
X, y = create_dataset(scaled_data, time_step)

train_size = 0.8
X_train, X_test = X[: int(X.shape[0] * train_size)], X[int(X.shape[0] * train_size) :]
y_train, y_test = y[: int(y.shape[0] * train_size)], y[int(y.shape[0] * train_size) :]


model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(units=64))
model.add(Dense(units=64))
model.add(Dense(units=1))

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X_train, y_train, epochs=10, batch_size=64)


model.summary()


test_loss = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)


actual_values = scaler.inverse_transform(y_test.reshape(-1, 1))


test_data_size = len(predictions)
time_indices = stock_data.index[-test_data_size:]


plt.figure(figsize=(12, 6))
plt.plot(time_indices, actual_values, label="Actual Stock Price", color="blue")
plt.plot(
    time_indices,
    predictions,
    label="Predicted Stock Price",
    color="red",
    linestyle="--",
)
plt.title("DAC Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Stock Price (USD)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
