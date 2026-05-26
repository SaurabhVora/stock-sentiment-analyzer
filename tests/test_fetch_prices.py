import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from pipeline.fetch_prices import fetch_stock_prices


class TestFetchPrices(unittest.TestCase):
    """Unit tests for the price fetching module using yfinance."""

    @patch("pipeline.fetch_prices.yf.Ticker")
    def test_fetch_stock_prices_success(self, mock_ticker: MagicMock) -> None:
        """Tests fetch_stock_prices successfully retrieves and formats historical data."""
        # Create a mock ticker instance and mock history method
        mock_instance = MagicMock()
        mock_ticker.return_value = mock_instance

        # Build mock DataFrame index and columns
        # Note: history method index will be Date
        date_range = pd.date_range(start="2026-05-15", end="2026-05-24", freq="D")
        mock_df = pd.DataFrame(
            {
                "Close": [100.0, 102.0, 101.0, 105.0, 104.0, 108.0, 107.0, 110.0, 109.0, 112.0],
                "Open": [99.0, 100.0, 102.0, 101.0, 105.0, 104.0, 108.0, 107.0, 110.0, 109.0],
            },
            index=date_range,
        )
        mock_instance.history.return_value = mock_df

        # Fetch prices for 2026-05-22 to 2026-05-24
        result_df = fetch_stock_prices("AAPL", "2026-05-22", "2026-05-24")

        self.assertFalse(result_df.empty)
        self.assertEqual(list(result_df.columns), ["Date", "Close", "PriceChangePct"])

        # Check date boundaries
        self.assertEqual(result_df["Date"].min(), "2026-05-22")
        self.assertEqual(result_df["Date"].max(), "2026-05-24")

        # Let's verify percentage change calculation for 2026-05-24
        row_24 = result_df[result_df["Date"] == "2026-05-24"].iloc[0]
        # In mock_df:
        # 2026-05-23: 109.0 (index 8)
        # 2026-05-24: 112.0 (index 9)
        # Change pct = (112.0 - 109.0) / 109.0 * 100 = 2.75229%
        self.assertAlmostEqual(row_24["PriceChangePct"], 2.75229, places=4)
        self.assertEqual(row_24["Close"], 112.0)

    @patch("pipeline.fetch_prices.yf.Ticker")
    def test_fetch_stock_prices_empty(self, mock_ticker: MagicMock) -> None:
        """Tests fetch_stock_prices when yfinance returns an empty DataFrame."""
        mock_instance = MagicMock()
        mock_ticker.return_value = mock_instance
        mock_instance.history.return_value = pd.DataFrame()

        result_df = fetch_stock_prices("AAPL", "2026-05-22", "2026-05-24")
        self.assertTrue(result_df.empty)
        self.assertEqual(list(result_df.columns), ["Date", "Close", "PriceChangePct"])

    @patch("pipeline.fetch_prices.yf.Ticker")
    def test_fetch_stock_prices_exception(self, mock_ticker: MagicMock) -> None:
        """Tests fetch_stock_prices when an exception is raised by yfinance."""
        mock_instance = MagicMock()
        mock_ticker.return_value = mock_instance
        mock_instance.history.side_effect = Exception("yfinance API error")

        result_df = fetch_stock_prices("AAPL", "2026-05-22", "2026-05-24")
        self.assertTrue(result_df.empty)
        self.assertEqual(list(result_df.columns), ["Date", "Close", "PriceChangePct"])
