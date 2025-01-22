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

LSTMs are designed to remember long-term patterns, which makes them good at capturing underlying trends.

More stable patterns (like AAPL) result in better predictions because:

- The relationship between past and future values is more consistent
- There are fewer extreme events to disrupt the learned patterns

High volatility stocks (like TSLA) show larger errors because:

- Sudden price changes can't be predicted from historical patterns alone
- External factors (like news, tweets, market sentiment) cause rapid shifts that historical data doesn't capture

ETFs (like VTI) show smaller errors because:

- They represent a basket of stocks, which reduces volatility
- The relationship between past and future values is more consistent
- There are fewer extreme events to disrupt the learned patterns

Correlation between ETFs:

ETFs like SPY (S&P 500), QQQ (NASDAQ 100), and VTI (Total Market) show very high correlation with each other since they track overlapping market segments. This is because:

- Many of the same large-cap stocks are included in multiple ETFs
- When the overall market moves up or down, these ETFs tend to move together

![ETFs correlation heat map](https://github.com/user-attachments/assets/aeff97a0-f225-4824-a559-492fd349ee8c)
![ETFs correlation chart](https://github.com/user-attachments/assets/b63c6c38-1b6a-437c-b1e3-70fb20872eb4)

Some stocks display high correlation with other stocks:
