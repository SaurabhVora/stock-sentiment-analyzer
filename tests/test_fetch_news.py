from datetime import datetime
import unittest
from unittest.mock import MagicMock, patch
from pipeline.fetch_news import (
    clean_date,
    fetch_headlines,
    fetch_news_from_newsapi,
    fetch_news_from_newsdata,
)


class TestFetchNews(unittest.TestCase):
    """Unit tests for the news fetching pipeline, verifying success, failure, and failover cases."""

    def test_clean_date_valid(self) -> None:
        """Tests clean_date with various standard date strings."""
        self.assertEqual(clean_date("2026-05-24T12:00:00Z"), "2026-05-24")
        self.assertEqual(clean_date("2026-05-24 15:30:22"), "2026-05-24")
        self.assertEqual(clean_date("2026-05-24T12:00:00+00:00"), "2026-05-24")
        self.assertEqual(clean_date("invalid-date"), datetime.utcnow().strftime("%Y-%m-%d"))

    @patch("pipeline.fetch_news.NewsApiClient")
    @patch("pipeline.fetch_news.NEWS_API_KEY", "fake_news_key")
    def test_fetch_news_from_newsapi_success(self, mock_newsapi_client: MagicMock) -> None:
        """Tests fetch_news_from_newsapi under successful response."""
        mock_instance = MagicMock()
        mock_newsapi_client.return_value = mock_instance
        mock_instance.get_everything.return_value = {
            "status": "ok",
            "articles": [
                {
                    "title": "Apple launches new iPhone",
                    "source": {"name": "TechCrunch"},
                    "publishedAt": "2026-05-24T08:00:00Z",
                }
            ],
        }

        results = fetch_news_from_newsapi("AAPL", "2026-05-24", "2026-05-24")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["ticker"], "AAPL")
        self.assertEqual(results[0]["headline"], "Apple launches new iPhone")
        self.assertEqual(results[0]["source"], "TechCrunch")
        self.assertEqual(results[0]["date"], "2026-05-24")

    @patch("pipeline.fetch_news.NewsApiClient")
    @patch("pipeline.fetch_news.NEWS_API_KEY", "fake_news_key")
    def test_fetch_news_from_newsapi_failure(self, mock_newsapi_client: MagicMock) -> None:
        """Tests fetch_news_from_newsapi when NewsAPI returns error status."""
        mock_instance = MagicMock()
        mock_newsapi_client.return_value = mock_instance
        mock_instance.get_everything.return_value = {
            "status": "error",
            "message": "Rate limit exceeded",
        }

        with self.assertRaises(Exception) as context:
            fetch_news_from_newsapi("AAPL", "2026-05-24", "2026-05-24")
        self.assertIn("Rate limit exceeded", str(context.exception))

    @patch("pipeline.fetch_news.requests.get")
    @patch("pipeline.fetch_news.NEWSDATA_API_KEY", "fake_newsdata_key")
    def test_fetch_news_from_newsdata_success(self, mock_get: MagicMock) -> None:
        """Tests fetch_news_from_newsdata under successful response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "success",
            "results": [
                {
                    "title": "Apple Stock Rises",
                    "source_id": "yahoo_finance",
                    "pubDate": "2026-05-24 10:00:00",
                }
            ],
        }
        mock_get.return_value = mock_response

        results = fetch_news_from_newsdata("AAPL")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["ticker"], "AAPL")
        self.assertEqual(results[0]["headline"], "Apple Stock Rises")
        self.assertEqual(results[0]["source"], "yahoo_finance")
        self.assertEqual(results[0]["date"], "2026-05-24")

    @patch("pipeline.fetch_news.requests.get")
    @patch("pipeline.fetch_news.NEWSDATA_API_KEY", "fake_newsdata_key")
    def test_fetch_news_from_newsdata_failure(self, mock_get: MagicMock) -> None:
        """Tests fetch_news_from_newsdata when response is an API failure."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "error",
            "results": {"message": "Invalid API key"},
        }
        mock_get.return_value = mock_response

        with self.assertRaises(Exception) as context:
            fetch_news_from_newsdata("AAPL")
        self.assertIn("Invalid API key", str(context.exception))

    @patch("pipeline.fetch_news.fetch_news_from_newsapi")
    @patch("pipeline.fetch_news.fetch_news_from_newsdata")
    @patch("pipeline.fetch_news.NEWS_API_KEY", "fake_news_key")
    @patch("pipeline.fetch_news.NEWSDATA_API_KEY", "fake_newsdata_key")
    def test_fetch_news_orchestration_primary_success(
        self, mock_newsdata: MagicMock, mock_newsapi: MagicMock
    ) -> None:
        """Tests fetch_headlines when NewsAPI works (should not call Newsdata.io)."""
        mock_newsapi.return_value = [
            {"ticker": "AAPL", "date": "2026-05-24", "headline": "Primary Headline"}
        ]

        results = fetch_headlines("AAPL", "2026-05-24", "2026-05-24")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["headline"], "Primary Headline")
        mock_newsapi.assert_called_once_with("AAPL", "2026-05-24", "2026-05-24")
        mock_newsdata.assert_not_called()

    @patch("pipeline.fetch_news.fetch_news_from_newsapi")
    @patch("pipeline.fetch_news.fetch_news_from_newsdata")
    @patch("pipeline.fetch_news.NEWS_API_KEY", "fake_news_key")
    @patch("pipeline.fetch_news.NEWSDATA_API_KEY", "fake_newsdata_key")
    def test_fetch_news_orchestration_failover(
        self, mock_newsdata: MagicMock, mock_newsapi: MagicMock
    ) -> None:
        """Tests fetch_headlines when NewsAPI fails, verify it falls back to Newsdata.io."""
        mock_newsapi.side_effect = Exception("NewsAPI failed")
        mock_newsdata.return_value = [
            {"ticker": "AAPL", "date": "2026-05-24", "headline": "Fallback Headline"}
        ]

        results = fetch_headlines("AAPL", "2026-05-24", "2026-05-24")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["headline"], "Fallback Headline")
        mock_newsapi.assert_called_once_with("AAPL", "2026-05-24", "2026-05-24")
        mock_newsdata.assert_called_once_with("AAPL")
