import pandas as pd
from data_processing import download_stock_data, prepare_data
from model import create_lstm_model, predict_future
from visualization import plot_stock_predictions_multi


def predict_stock_price(symbol, start, end):
    stock_data = download_stock_data(symbol, start, end)
    
    data_length = len(stock_data)
    time_step = max(min(int(data_length * 0.2), 100), 5)
    
    X_train, X_test, y_train, y_test, scaler, scaled_data = prepare_data(
        stock_data, time_step
    )

    model = create_lstm_model(time_step)
    model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=0)

    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"{symbol} Test Loss:", test_loss)

    predictions = model.predict(X_test, verbose=0)
    predictions = scaler.inverse_transform(predictions)
    actual_values = scaler.inverse_transform(y_test.reshape(-1, 1))
    full_actual_values = stock_data["Close"].values

    test_data_size = len(predictions)
    training_data_size = len(full_actual_values) - test_data_size - time_step - 1

    future_days = 7
    future_predictions = predict_future(
        model, scaled_data, time_step, future_days, scaler
    )

    last_date = stock_data.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=future_days, freq="B"
    )

    print(f"predictions: {predictions}")
    print(f"actual_values: {actual_values}")

    return {
        "stock_data": stock_data,
        "predictions": predictions,
        "actual_values": actual_values,
        "full_actual_values": full_actual_values,
        "future_predictions": future_predictions,
        "future_dates": future_dates,
        "time_step": time_step,
        "training_data_size": training_data_size,
        "symbol": symbol,
    }


def compare_stocks(symbols, start, end):
    predictions_data = []
    
    for symbol in symbols:
        print(f"\nProcessing {symbol}...")
        stock_prediction = predict_stock_price(symbol, start, end)
        predictions_data.append(stock_prediction)

    plot_stock_predictions_multi(predictions_data)


if __name__ == "__main__":
    symbols = [
        "TSLA",
        "AAPL",
        "NVDA",
        # "SPY",  # S&P 500 ETF
        # "QQQ",  # Nasdaq 100 ETF
        # "VTI",  # Total Stock Market ETF
        # "VUSA.L",
    ]
    start = "2024-01-01"  # Using 5 years of data
    end = "2024-12-31"
    compare_stocks(symbols, start, end)
