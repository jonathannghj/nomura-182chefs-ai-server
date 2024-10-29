import yfinance as yf
import pandas as pd

def fetch_historical_data(symbols, period='1y'):
    """
    Fetch historical data for given stock symbols

    Parameters
    ==========
    * symbols (list): list of stock symbols
    * period (str): period for fetching historical data (default: '1y')

    Returns
    =======
    * dict: Dictionary with stock symbols as keys and historical close prices as values
    """
    data = {}
    for symbol in symbols:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        data[symbol] = hist['Close']
    return data

def calculate_metrics(historical_data):
    """
    Calculate expected returns and covariance matrix from historical data

    Parameters
    ==========
    * historical_data (dict): Dictionary with stock symbols as keys and historical close prices as values

    Returns
    =======
    * tuple: Expected returns and covariance matrix
    """
    prices = pd.DataFrame(historical_data)
    returns = prices.pct_change().dropna()
    expected_returns = returns.mean().values
    cov_matrix = returns.cov().values
    return expected_returns, cov_matrix

def fetch_current_prices(symbols):
    """
    Fetch current prices for given stock symbols

    Parameters
    ==========
    * symbols (list): List of stock symbols

    Returns
    =======
    * dict: Dictionary with stock symbols as keys and current prices as values
    """
    prices = {}
    for symbol in symbols:
        stock = yf.Ticker(symbol)
        prices[symbol] = stock.history(period="1d")['Close'].iloc[0]
    return prices
