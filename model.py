from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np


def create_lstm_model(time_step):
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(units=64))
    model.add(Dense(units=64))
    model.add(Dense(units=1))

    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def predict_future(model, scaled_data, time_step, future_days, scaler):
    last_sequence = scaled_data[-time_step:].reshape(1, time_step, 1)
    future_predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(future_days):
        next_pred = model.predict(current_sequence, verbose=0)
        future_predictions.append(next_pred[0, 0])

        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, 0] = next_pred[0, 0]

    future_predictions = np.array(future_predictions).reshape(-1, 1)
    return scaler.inverse_transform(future_predictions)
