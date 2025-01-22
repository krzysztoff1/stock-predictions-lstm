import pandas as pd
from data_processing import download_stock_data, prepare_data
from model import create_lstm_model, predict_future
from visualization import plot_stock_predictions


def predict_stock_price(symbol, start, end):
    stock_data = download_stock_data(symbol, start, end)
    time_step = 100
    X_train, X_test, y_train, y_test, scaler, scaled_data = prepare_data(
        stock_data, time_step
    )

    model = create_lstm_model(time_step)
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
    future_predictions = predict_future(
        model, scaled_data, time_step, future_days, scaler
    )

    last_date = stock_data.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=future_days, freq="B"
    )

    plot_stock_predictions(
        stock_data,
        predictions,
        actual_values,
        full_actual_values,
        future_predictions,
        future_dates,
        time_step,
        training_data_size,
        symbol,
    )


if __name__ == "__main__":
    predict_stock_price("DAC", start="2014-02-21", end="2024-02-21")
