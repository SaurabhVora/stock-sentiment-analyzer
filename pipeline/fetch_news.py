import logging
import os
from datetime import datetime
from typing import Any, Dict, List
from dotenv import load_dotenv
from newsapi import NewsApiClient
import requests

logger = logging.getLogger(__name__)

# Load local environment variables
load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")


def clean_date(date_str: str) -> str:
    """Cleans and parses a date string into ISO 8601 format YYYY-MM-DD.

    Args:
        date_str (str): The raw date string from the API.

    Returns:
        str: The cleaned date in YYYY-MM-DD format.
    """
    if not date_str:
        return datetime.utcnow().strftime("%Y-%m-%d")
    try:
        # Extract first 10 characters (YYYY-MM-DD)
        date_part = date_str[:10]
        datetime.strptime(date_part, "%Y-%m-%d")
        return date_part
    except ValueError:
        # Try custom common formats
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z"):
            try:
                return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        return datetime.utcnow().strftime("%Y-%m-%d")


def fetch_news_from_newsapi(ticker: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """Fetches news headlines from NewsAPI for a given stock ticker and date range.

    Args:
        ticker (str): The stock ticker to query.
        start_date (str): Start date string (YYYY-MM-DD).
        end_date (str): End date string (YYYY-MM-DD).

    Returns:
        List[Dict[str, Any]]: A list of uniform articles:
            [{'ticker': str, 'date': str, 'headline': str, 'source': str}]

    Raises:
        ValueError: If NEWS_API_KEY is not set.
        Exception: If the API response status is not 'ok' or request fails.
    """
    if not NEWS_API_KEY or NEWS_API_KEY == "your_news_api_key_here":
        raise ValueError("NEWS_API_KEY is missing from the environment variables.")

    # Map tickers to specific brand queries to improve search relevancy
    brand_keywords = {
        "AAPL": "Apple OR AAPL",
        "XOM": "ExxonMobil OR Exxon OR XOM",
        "CVX": "Chevron OR CVX",
        "SHEL": "Royal Dutch Shell OR SHEL"
    }
    query = brand_keywords.get(ticker.upper(), ticker)

    client = NewsApiClient(api_key=NEWS_API_KEY)
    response = client.get_everything(
        q=query,
        from_param=start_date,
        to=end_date,
        language="en",
        sort_by="publishedAt",
        page_size=100
    )

    if response.get("status") != "ok":
        error_msg = response.get("message", "Unknown NewsAPI error")
        raise RuntimeError(f"NewsAPI error: {error_msg}")

    articles = response.get("articles", [])
    results: List[Dict[str, Any]] = []

    for art in articles:
        headline = art.get("title")
        if not headline or headline == "[Removed]":
            continue

        source_name = "Unknown"
        source_dict = art.get("source")
        if isinstance(source_dict, dict):
            source_name = source_dict.get("name", "Unknown")

        pub_date = clean_date(art.get("publishedAt", ""))

        results.append({
            "ticker": ticker.upper(),
            "date": pub_date,
            "headline": headline,
            "source": source_name
        })

    return results


def fetch_news_from_newsdata(ticker: str) -> List[Dict[str, Any]]:
    """Fetches news headlines from Newsdata.io fallback API for a given stock ticker.

    Args:
        ticker (str): The stock ticker to query.

    Returns:
        List[Dict[str, Any]]: A list of uniform articles:
            [{'ticker': str, 'date': str, 'headline': str, 'source': str}]

    Raises:
        ValueError: If NEWSDATA_API_KEY is not set.
        Exception: If the requests call fails or the API returns an error status.
    """
    if not NEWSDATA_API_KEY or NEWSDATA_API_KEY == "your_newsdata_api_key_here":
        raise ValueError("NEWSDATA_API_KEY is missing from the environment variables.")

    url = "https://newsdata.io/api/1/news"
    params = {
        "apikey": NEWSDATA_API_KEY,
        "q": ticker,
        "language": "en"
    }

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()

    if data.get("status") != "success":
        error_info = data.get("results", {})
        error_msg = (
            error_info.get("message")
            if isinstance(error_info, dict)
            else "Unknown Newsdata.io error"
        )
        raise RuntimeError(f"Newsdata.io error: {error_msg}")

    articles = data.get("results", [])
    results: List[Dict[str, Any]] = []

    for art in articles:
        headline = art.get("title")
        if not headline:
            continue

        source_name = art.get("source_id", "Unknown")
        pub_date = clean_date(art.get("pubDate", ""))

        results.append({
            "ticker": ticker.upper(),
            "date": pub_date,
            "headline": headline,
            "source": source_name
        })

    return results


def fetch_headlines(ticker: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """Fetches stock sentiment news headlines for a ticker using NewsAPI with Newsdata.io fallback.

    Args:
        ticker (str): The stock ticker to fetch headlines for.
        start_date (str): Start date string (YYYY-MM-DD).
        end_date (str): End date string (YYYY-MM-DD).

    Returns:
        List[Dict[str, Any]]: A cleaned, uniform list of article dictionaries:
            [{'ticker': str, 'date': str, 'headline': str, 'source': str}]
    """
    if not NEWS_API_KEY and not NEWSDATA_API_KEY:
        raise ValueError("Both NEWS_API_KEY and NEWSDATA_API_KEY are missing. "
                         "Please define at least one in the environment.")

    logger.info("Initializing news fetch for ticker: %s from %s to %s",
                ticker, start_date, end_date)

    # 1. Attempt NewsAPI
    if NEWS_API_KEY and NEWS_API_KEY != "your_news_api_key_here":
        try:
            logger.info("Attempting to fetch news from NewsAPI...")
            articles = fetch_news_from_newsapi(ticker, start_date, end_date)
            logger.info("Successfully fetched %d articles from NewsAPI", len(articles))
            return articles
        except Exception as e:
            logger.warning("NewsAPI ingestion failed: %s. Initiating failover...", e)
    else:
        logger.info("NEWS_API_KEY is not set. Skipping NewsAPI step.")

    # 2. Attempt Newsdata.io Fallback
    if NEWSDATA_API_KEY and NEWSDATA_API_KEY != "your_newsdata_api_key_here":
        try:
            logger.info("Attempting to fetch news from Newsdata.io...")
            articles = fetch_news_from_newsdata(ticker)
            logger.info("Successfully fetched %d articles from Newsdata.io fallback", len(articles))
            # Filter fallback headlines by start_date and end_date if they fall outside the range
            filtered_articles = [
                art for art in articles
                if start_date <= art["date"] <= end_date
            ]
            logger.info("Filtered to %d articles within date range", len(filtered_articles))
            return filtered_articles
        except Exception as e:
            logger.error("Newsdata.io fallback ingestion failed: %s", e)
    else:
        logger.warning("NEWSDATA_API_KEY is not set. No fallback available.")

    logger.error("All news ingestion sources failed to retrieve data for ticker %s", ticker)
    return []
