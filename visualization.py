import matplotlib.pyplot as plt
import numpy as np


def plot_prediction_difference(dates, actual_values, predictions, symbol):
    # Calculate percentage difference: (predicted - actual) / actual * 100
    percentage_differences = (
        (predictions - actual_values.flatten()) / actual_values.flatten()
    ) * 100

    plt.figure(figsize=(15, 5))
    plt.plot(dates, percentage_differences, label="Prediction Error (%)", color="blue")
    plt.axhline(y=0, color="r", linestyle="--", alpha=0.3)

    plt.title(f"{symbol} Stock Price - Prediction Error Percentage")
    plt.xlabel("Date")
    plt.ylabel("Prediction Error (%)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


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
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    # First subplot - Stock prices and predictions
    ax1.plot(
        stock_data.index[time_step + 1 : time_step + 1 + training_data_size],
        full_actual_values[time_step + 1 : time_step + 1 + training_data_size],
        label="Training Data",
        color="green",
    )
    ax1.plot(
        stock_data.index[-len(predictions) :],
        actual_values,
        label="Test Data (Actual)",
        color="green",
    )
    ax1.plot(
        stock_data.index[-len(predictions) :],
        predictions,
        label="Test Data (Predicted)",
        color="red",
        linestyle="--",
    )
    ax1.plot(
        future_dates,
        future_predictions,
        label="Future Predictions",
        color="purple",
        linestyle="--",
    )

    ax1.set_title(f"{symbol} Stock Price - Historical Data and Future Predictions")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Stock Price (USD)")
    ax1.legend()
    ax1.grid(True)
    ax1.tick_params(axis="x", rotation=45)

    # Second subplot - Prediction error percentage
    # Ensure arrays are 1D and same shape
    actual_values_flat = actual_values.flatten()
    predictions_flat = predictions.flatten()

    # Calculate percentage error
    percentage_differences = np.where(
        actual_values_flat != 0,
        ((predictions_flat - actual_values_flat) / actual_values_flat) * 100,
        0,
    )

    ax2.plot(
        stock_data.index[-len(predictions) :],
        percentage_differences,
        label="Prediction Error (%)",
        color="blue",
    )
    ax2.axhline(y=0, color="r", linestyle="--", alpha=0.3)

    ax2.set_title(f"{symbol} Stock Price - Prediction Error Percentage")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Prediction Error (%)")
    ax2.legend()
    ax2.grid(True)
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()
