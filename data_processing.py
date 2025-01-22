import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def download_stock_data(symbol, start, end):
    return yf.download(symbol, start=start, end=end)


def create_dataset(data, time_step):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i : (i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)


def prepare_data(stock_data, time_step, train_size=0.8):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data["Close"].values.reshape(-1, 1))

    X, y = create_dataset(scaled_data, time_step)

    X_train, X_test = (
        X[: int(X.shape[0] * train_size)],
        X[int(X.shape[0] * train_size) :],
    )
    y_train, y_test = (
        y[: int(y.shape[0] * train_size)],
        y[int(y.shape[0] * train_size) :],
    )

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    return X_train, X_test, y_train, y_test, scaler, scaled_data
