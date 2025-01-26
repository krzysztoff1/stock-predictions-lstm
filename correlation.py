import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_correlated_pair(data, stock1, stock2, correlation):
    """Plot the most correlated pair of stocks."""
    normalized_data = pd.DataFrame()
    normalized_data[stock1] = data[stock1] / data[stock1].iloc[0] * 100
    normalized_data[stock2] = data[stock2] / data[stock2].iloc[0] * 100

    plt.figure(figsize=(15, 8))
    plt.plot(normalized_data.index, normalized_data[stock1], label=stock1, linewidth=2)
    plt.plot(normalized_data.index, normalized_data[stock2], label=stock2, linewidth=2)

    plt.title(
        f"Most Correlated Pair: {stock1} vs {stock2} (correlation: {correlation:.3f})"
    )
    plt.xlabel("Date")
    plt.ylabel("Normalized Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("correlated_pair.png", bbox_inches="tight")


def calculate_stock_correlations(stocks, start_date, end_date):
    data = pd.DataFrame()
    for stock in stocks:
        try:
            ticker = yf.Ticker(stock)
            hist = ticker.history(start=start_date, end=end_date)["Close"]
            data[stock] = hist
        except Exception as e:
            print(f"Error fetching data for {stock}: {e}")
            continue

    data = data.dropna()

    if data.empty:
        print("No valid data found for any stocks")
        return None, None

    correlation_matrix = data.corr()

    plt.figure(figsize=(15, 10))
    mask = np.triu(np.ones_like(correlation_matrix), k=1)
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        mask=mask,
        fmt=".2f",
    )
    plt.title("Stock Price Correlations")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")

    return correlation_matrix, data


if __name__ == "__main__":
    etfs = [
        "SPY",  # S&P 500
        "QQQ",  # NASDAQ 100
        "VTI",  # Total Market
        "GLD",  # Gold
    ]

    stocks = [
        "TSLA",
        "AAPL",
        "NVDA",
        "MSFT",
        "GOOG",
        "AMZN",
        "META",
        "BRK-B",
        "JPM",
        "JNJ",
        "V",
        "PG",
        "MA",
        "UNH",
        "HD",
        "BAC",
        "XOM",
        "DIS",
        "NFLX",
    ]

    # stocks_to_compare = etfs + stocks
    # stocks_to_compare = etfs
    stocks_to_compare = stocks

    start = "2023-01-01"  # Using 5 years of data
    end = "2024-01-01"

    correlation_matrix, price_data = calculate_stock_correlations(
        stocks_to_compare, start, end
    )

    if correlation_matrix is not None:
        correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                stock1 = correlation_matrix.columns[i]
                stock2 = correlation_matrix.columns[j]
                corr = correlation_matrix.iloc[i, j]
                if not np.isnan(corr):
                    correlations.append((stock1, stock2, corr))

        correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        print("\nTop 5 most correlated pairs:")
        for stock1, stock2, corr in correlations[:5]:
            print(f"{stock1} - {stock2}: {corr:.3f}")

        best_pair = correlations[0]
        plot_correlated_pair(price_data, best_pair[0], best_pair[1], best_pair[2])

        plt.show()
