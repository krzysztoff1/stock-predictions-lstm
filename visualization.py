import matplotlib.pyplot as plt


def plot_stock_predictions(
    stock_data,
    predictions,
    actual_values,
    full_actual_values,
    future_predictions,
    future_dates,
    time_step,
    training_data_size,
    symbol,
):
    plt.figure(figsize=(15, 7))

    plt.plot(
        stock_data.index[time_step + 1 : time_step + 1 + training_data_size],
        full_actual_values[time_step + 1 : time_step + 1 + training_data_size],
        label="Training Data",
        color="green",
    )
    plt.plot(
        stock_data.index[-len(predictions) :],
        actual_values,
        label="Test Data (Actual)",
        color="green",
    )
    plt.plot(
        stock_data.index[-len(predictions) :],
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

    plt.title(f"{symbol} Stock Price - Historical Data and Future Predictions")
    plt.xlabel("Date")
    plt.ylabel("Stock Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
