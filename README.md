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

Run the correlation script to analyze stock correlations:

```bash
uv run correlation.py
```

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

![v-ma](https://github.com/user-attachments/assets/9547dc48-f056-474b-a93b-7279c8c43a43)
