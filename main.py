import yfinance as yf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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


X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

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
full_actual_values = stock_data["Close"].values


test_data_size = len(predictions)
training_data_size = len(full_actual_values) - test_data_size - time_step - 1


future_days = 30
last_sequence = scaled_data[-time_step:].reshape(1, time_step, 1)


future_predictions = []
current_sequence = last_sequence.copy()

for _ in range(future_days):
    next_pred = model.predict(current_sequence, verbose=0)
    future_predictions.append(next_pred[0, 0])

    current_sequence = np.roll(current_sequence, -1, axis=1)
    current_sequence[0, -1, 0] = next_pred[0, 0]


future_predictions = np.array(future_predictions).reshape(-1, 1)
future_predictions = scaler.inverse_transform(future_predictions)


last_date = stock_data.index[-1]
future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1), periods=future_days, freq="B"
)


plt.figure(figsize=(15, 7))


plt.plot(
    stock_data.index[time_step + 1 : time_step + 1 + training_data_size],
    full_actual_values[time_step + 1 : time_step + 1 + training_data_size],
    label="Training Data",
    color="green",
)
plt.plot(
    stock_data.index[-test_data_size:],
    actual_values,
    label="Test Data (Actual)",
    color="green",
)
plt.plot(
    stock_data.index[-test_data_size:],
    predictions,
    label="Test Data (Predicted)",
    color="red",
    linestyle="--",
)
plt.plot(
    future_dates,
    future_predictions,
    label="Future Predictions",
    color="purple",
    linestyle="--",
)

plt.title("DAC Stock Price - Historical Data and Future Predictions")
plt.xlabel("Date")
plt.ylabel("Stock Price (USD)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
