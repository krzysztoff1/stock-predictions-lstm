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


def plot_stock_predictions_multi(predictions_data):
    """
    Plot multiple stocks predictions and their error percentages.
    predictions_data: list of dictionaries containing:
        {
            'stock_data': pd.DataFrame,
            'predictions': np.array,
            'actual_values': np.array,
            'full_actual_values': np.array,
            'future_predictions': np.array,
            'future_dates': pd.DatetimeIndex,
            'time_step': int,
            'training_data_size': int,
            'symbol': str
        }
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    colors = ["green", "blue", "red", "purple", "orange", "brown", "pink", "gray"]

    # First subplot - Stock prices and predictions (normalized)
    for idx, stock in enumerate(predictions_data):
        # Normalize the stock prices to percentage change from first point
        base_price = stock["full_actual_values"][stock["time_step"] + 1]

        training_normalized = (
            stock["full_actual_values"][
                stock["time_step"] + 1 : stock["time_step"]
                + 1
                + stock["training_data_size"]
            ]
            / base_price
            - 1
        ) * 100
        actual_normalized = (stock["actual_values"] / base_price - 1) * 100
        predictions_normalized = (stock["predictions"] / base_price - 1) * 100
        future_normalized = (stock["future_predictions"] / base_price - 1) * 100

        color = colors[idx % len(colors)]

        ax1.plot(
            stock["stock_data"].index[
                stock["time_step"] + 1 : stock["time_step"]
                + 1
                + stock["training_data_size"]
            ],
            training_normalized,
            label=f"{stock['symbol']} Training",
            color=color,
            alpha=0.3,
        )
        ax1.plot(
            stock["stock_data"].index[-len(stock["predictions"]) :],
            actual_normalized.flatten(),
            label=f"{stock['symbol']} Actual",
            color=color,
        )
        ax1.plot(
            stock["stock_data"].index[-len(stock["predictions"]) :],
            predictions_normalized.flatten(),
            label=f"{stock['symbol']} Predicted",
            color=color,
            linestyle="--",
        )
        ax1.plot(
            stock["future_dates"],
            future_normalized.flatten(),
            label=f"{stock['symbol']} Future",
            color=color,
            linestyle=":",
        )

    ax1.set_title("Multiple Stocks - Price Changes (%)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price Change (%)")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True)
    ax1.tick_params(axis="x", rotation=45)

    # Second subplot - Prediction error percentage comparison
    for idx, stock in enumerate(predictions_data):
        actual_values_flat = stock["actual_values"].flatten()
        predictions_flat = stock["predictions"].flatten()

        # Calculate percentage error
        percentage_differences = np.where(
            actual_values_flat != 0,
            ((predictions_flat - actual_values_flat) / actual_values_flat) * 100,
            0,
        )

        ax2.plot(
            stock["stock_data"].index[-len(stock["predictions"]) :],
            percentage_differences,
            label=f"{stock['symbol']} Error",
            color=colors[idx % len(colors)],
        )

    ax2.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    ax2.set_title("Prediction Error Percentage Comparison")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Prediction Error (%)")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.grid(True)
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()
