# Stock Price Correlation Analysis

A Python tool for analyzing correlations between stock prices and visualizing the relationships between different stocks and ETFs.

## Features

- Calculate correlations between multiple stocks and ETFs
- Generate correlation heatmaps
- Plot normalized price comparisons for the most correlated pairs
- Support for major stocks and ETFs
- Customizable date ranges

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/stock-predictions.git
cd stock-predictions
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

The script can analyze three different sets of assets:

- Individual stocks
- ETFs
- Both stocks and ETFs combined

To run the analysis:

```bash
python correlation.py
```

### Available Assets

#### ETFs

- `SPY` - S&P 500 Index
- `QQQ` - NASDAQ 100 Index
- `VTI` - Total Market Index
- `GLD` - Gold Trust

#### Stocks

- Tech: `AAPL`, `MSFT`, `GOOG`, `AMZN`, `META`, `NVDA`, `TSLA`, `NFLX`
- Finance: `JPM`, `V`, `MA`, `BAC`
- Healthcare: `JNJ`, `UNH`
- Others: `BRK-B`, `PG`, `HD`, `XOM`, `DIS`

### Example Output

1. **Correlation Matrix**

   - A heatmap showing correlations between all pairs of assets
   - Saved as `correlation_heatmap.png`

2. **Most Correlated Pair**

   - A line chart showing normalized prices of the most correlated pair
   - Saved as `correlated_pair.png`

3. **Console Output**

```
Top 5 most correlated pairs:
MSFT - AAPL: 0.939
V - MA: 0.934
QQQ - SPY: 0.928
GOOG - MSFT: 0.925
AMZN - NVDA: 0.922
```

## Customization

You can modify the analysis by:

1. Changing the date range:

```python
start = "2020-01-01"
end = "2024-01-01"
```

2. Selecting which assets to analyze (uncomment one of these lines):

```python
stocks_to_compare = etfs + stocks  # Both ETFs and stocks
stocks_to_compare = etfs           # Only ETFs
stocks_to_compare = stocks         # Only stocks
```

## Output Files

- `correlation_heatmap.png`: Visualization of all correlations
- `correlated_pair.png`: Chart of the most correlated pair

## Dependencies

- yfinance
- pandas
- seaborn
- matplotlib
- numpy

## Notes

- Data is fetched from Yahoo Finance using the `yfinance` library
- Prices are normalized to start at 100 for easier comparison
- Missing data points are automatically handled
- The correlation coefficient ranges from -1 (perfect negative correlation) to 1 (perfect positive correlation)

# Stock Predictions

A Python-based machine learning project for predicting stock market movements using historical data and advanced analytics.
![Preview](https://github.com/user-attachments/assets/ac3d25ca-fefa-4c8b-b7f4-5e5f58a9d65f)

## Setup

### Install uv

First, install `uv`, a fast Python package installer and environment manager ([github.com/astral-sh/uv](https://github.com/astral-sh/uv)) if you don't have it already.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create Environment and Install Dependencies

1. Create a virtual environment using uv:

```bash
uv venv
```

2. Activate the virtual environment:

```bash
source .venv/bin/activate
```

3. Install dependencies using uv:

```bash
uv pip install -r requirements.txt
```

## Usage

Run the main script to start the stock prediction analysis:

```bash
uv run main.py
```

## Findings

### LSTM Model Performance

LSTMs are designed to remember long-term patterns, which makes them good at capturing underlying trends in stock price movements.

#### Stable Stocks (e.g. AAPL)

Result in better predictions because:

- The relationship between past and future values is more consistent
- There are fewer extreme events to disrupt the learned patterns

#### Volatile Stocks (e.g. TSLA)

Show larger prediction errors because:

- Sudden price changes can't be predicted from historical patterns alone
- External factors (news, tweets, market sentiment) cause rapid shifts that historical data doesn't capture

#### ETFs (e.g. VTI)

Show smaller prediction errors because:

- They represent a basket of stocks, which reduces volatility
- The relationship between past and future values is more consistent
- There are fewer extreme events to disrupt the learned patterns

### Correlation Analysis

#### ETF Correlations

ETFs like SPY (S&P 500), QQQ (NASDAQ 100), and VTI (Total Market) show very high correlation with each other since they track overlapping market segments. This is because:

- Many of the same large-cap stocks are included in multiple ETFs
- When the overall market moves up or down, these ETFs tend to move together

![ETFs correlation heat map](https://github.com/user-attachments/assets/aeff97a0-f225-4824-a559-492fd349ee8c)
![ETFs correlation chart](https://github.com/user-attachments/assets/b63c6c38-1b6a-437c-b1e3-70fb20872eb4)

#### Stock Correlations

Some stocks display high correlation with each other. For example, V (Visa) and MA (Mastercard):
