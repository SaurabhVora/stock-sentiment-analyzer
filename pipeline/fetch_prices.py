"""Stock price ingestion service using yfinance.

This module downloads stock OHLCV data, computes daily closing price percentage changes,
and handles caching and edge cases like missing or unavailable trading days.
"""

import logging
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

# Setup logging
logger = logging.getLogger(__name__)


def fetch_stock_prices(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches daily stock closing prices and calculates percentage changes from yfinance.

    To ensure that the percentage change for the first requested date is computed
    correctly, this method fetches 5 extra buffer days before the start_date.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL').
        start_date (str): Start date string (YYYY-MM-DD).
        end_date (str): End date string (YYYY-MM-DD).

    Returns:
        pd.DataFrame: A pandas DataFrame containing:
            'Date': Date string (YYYY-MM-DD)
            'Close': Daily close price (float)
            'PriceChangePct': Day-over-day percentage change (float)
    """
    logger.info("Fetching yfinance prices for ticker %s between %s and %s...",
                ticker, start_date, end_date)

    try:
        # Request a slightly wider window to allow percentage change on the first day
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        fetch_start_dt = start_dt - timedelta(days=7)
        fetch_start_str = fetch_start_dt.strftime("%Y-%m-%d")

        # Adjust end_date to make it inclusive (yfinance end date is exclusive by default)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        fetch_end_dt = end_dt + timedelta(days=1)
        fetch_end_str = fetch_end_dt.strftime("%Y-%m-%d")

        # Download stock price data
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(start=fetch_start_str, end=fetch_end_str)

        if df.empty:
            logger.warning("No price data returned by yfinance for %s.", ticker)
            return pd.DataFrame(columns=["Date", "Close", "PriceChangePct"])

        # Handle yfinance multi-index column outputs if yfinance returns multi-index
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Confirm Close column exists
        if "Close" not in df.columns:
            logger.error("'Close' price column missing from downloaded yfinance DataFrame.")
            return pd.DataFrame(columns=["Date", "Close", "PriceChangePct"])

        # Calculate daily percent change
        df["PriceChangePct"] = df["Close"].pct_change() * 100.0

        # Ensure index name is Date before resetting
        df.index.name = "Date"
        df = df.reset_index()

        # Format Date to YYYY-MM-DD
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

        # Filter the DataFrame to match the requested date range
        df_filtered = df[df["Date"] >= start_date].copy()

        # Keep only the required columns and fill NaNs
        df_filtered = df_filtered[["Date", "Close", "PriceChangePct"]]
        df_filtered = df_filtered.fillna(0.0)

        # Typecast values to ensure clean Python types
        df_filtered["Close"] = df_filtered["Close"].astype(float)
        df_filtered["PriceChangePct"] = df_filtered["PriceChangePct"].astype(float)

        logger.info("Successfully fetched %d price entries for %s.", len(df_filtered), ticker)
        return df_filtered

    except Exception as e:
        logger.error("Error fetching stock prices for %s from yfinance: %s", ticker, e)
        # Return empty DataFrame rather than crashing to fulfill error tolerance requirements
        return pd.DataFrame(columns=["Date", "Close", "PriceChangePct"])
