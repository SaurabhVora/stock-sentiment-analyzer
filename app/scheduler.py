"""APScheduler-based background job and data sync orchestrator.

This module sets up a non-blocking BackgroundScheduler that runs every 15 minutes
to fetch news and price updates. It also handles the initial database bootloader
backfill in a separate background thread to keep the user interface responsive.
"""

import logging
import threading
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler

from pipeline.fetch_news import fetch_headlines
from pipeline.sentiment import score_headlines_finbert
from pipeline.fetch_prices import fetch_stock_prices
from pipeline.storage import save_headlines, save_daily_summaries, get_headlines

# Setup logging
logger = logging.getLogger(__name__)

# Tracked tickers list
TRACKED_TICKERS = ["AAPL", "XOM", "CVX", "SHEL"]

# Global scheduler reference
_scheduler = None


def run_pipeline_for_ticker(ticker: str, start_date: str, end_date: str) -> None:
    """Executes the complete data ingest, NLP scoring, and daily aggregation pipeline.

    Args:
        ticker: The stock ticker symbol.
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).
    """
    logger.info("Running pipeline for %s from %s to %s...", ticker, start_date, end_date)

    # 1. Ingest stock price data from yfinance
    price_data = fetch_stock_prices(ticker, start_date, end_date)
    if price_data.empty:
        logger.warning(
            "No price data retrieved for %s (possible rate limit). "
            "Continuing with daily aggregation using headlines only.",
            ticker
        )

    # 2. Ingest news headlines from NewsAPI
    headlines = []
    try:
        headlines = fetch_headlines(ticker, start_date, end_date)
    except Exception as e:
        logger.warning(
            "NewsAPI fetch failed for %s: %s. Continuing with price updates only.", ticker, e
        )

    # 3. Score headlines via FinBERT and insert into database
    if headlines:
        try:
            texts = [hl["headline"] for hl in headlines]
            scores = score_headlines_finbert(texts)

            # Map classifications and scores
            for hl, score_dict in zip(headlines, scores):
                hl["sentiment"] = score_dict["label"]
                hl["score"] = score_dict["score"]
                hl["weighted"] = score_dict["weighted"]

            save_headlines(headlines)
        except Exception as e:
            logger.error("Failed to run sentiment scoring or write headlines for %s: %s", ticker, e)

    # 4. Generate daily aggregates based on stored headlines and fetched price points
    all_dates = set()
    if not price_data.empty:
        all_dates.update(price_data["Date"].astype(str).tolist())
    if headlines:
        all_dates.update([hl["date"] for hl in headlines])

    # Fallback: if both are empty, generate chronological date range
    if not all_dates:
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            delta = end_dt - start_dt
            for i in range(delta.days + 1):
                all_dates.add((start_dt + timedelta(days=i)).strftime("%Y-%m-%d"))
        except Exception:
            pass

    # Map price_data by date for fast lookup
    price_map = {}
    if not price_data.empty:
        for _, row in price_data.iterrows():
            d = str(row["Date"])
            price_map[d] = {
                "Close": float(row["Close"]),
                "PriceChangePct": float(row["PriceChangePct"])
            }

    summaries = []
    for date_str in sorted(all_dates):
        # Retrieve price details if available, otherwise default to 0.0
        price_info = price_map.get(date_str)
        if price_info:
            close_price = price_info["Close"]
            pct_change = price_info["PriceChangePct"]
        else:
            close_price = 0.0
            pct_change = 0.0

        # Fetch all headlines for this ticker/date to compute exact weighted average sentiment
        stored_hl = get_headlines(ticker, date_str, date_str)
        if stored_hl:
            # Weighted average sentiment = sum(weighted_scores) / total_headlines
            total_weighted = sum(float(h["weighted"]) for h in stored_hl)
            avg_sentiment = total_weighted / len(stored_hl)
        else:
            avg_sentiment = 0.0  # Defaults to neutral if no headlines exist

        summaries.append({
            "ticker": ticker.upper(),
            "date": date_str,
            "avg_sentiment": avg_sentiment,
            "close_price": close_price,
            "price_change_pct": pct_change
        })

    try:
        save_daily_summaries(summaries)
        logger.info("Successfully completed daily pipeline upserts for %s.", ticker)
    except Exception as e:
        logger.error("Failed to save daily summary aggregates for %s: %s", ticker, e)


def scheduler_job() -> None:
    """The recurring scheduler callback function.

    Runs every 15 minutes, processing data for the last 2 days to capture
    new articles and price updates.
    """
    logger.info("Starting scheduled 15-minute pipeline updates...")
    today_str = datetime.now().strftime("%Y-%m-%d")
    yesterday_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    for ticker in TRACKED_TICKERS:
        try:
            run_pipeline_for_ticker(ticker, yesterday_str, today_str)
        except Exception as e:
            logger.error("Error during scheduled sync for ticker %s: %s", ticker, e)
    logger.info("Scheduled pipeline updates completed.")


def run_backfill_job() -> None:
    """Executes a 7-day initial backfill for all tracked tickers.

    Runs in a dedicated thread to avoid blocking Streamlit startup processes.
    """
    logger.info("Initializing background 7-day database backfill job...")
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

    for ticker in TRACKED_TICKERS:
        try:
            run_pipeline_for_ticker(ticker, start_date, end_date)
        except Exception as e:
            logger.error("Backfill failed for ticker %s: %s", ticker, e)
    logger.info("Background database backfill job completed successfully.")


def trigger_backfill() -> None:
    """Launches the backfill job asynchronously in a daemonized background thread."""
    thread = threading.Thread(target=run_backfill_job, name="BackfillThread", daemon=True)
    thread.start()
    logger.info("Database backfill thread started.")


def start_scheduler() -> BackgroundScheduler:
    """Instantiates and activates the BackgroundScheduler.

    Returns:
        BackgroundScheduler: The active scheduler instance.
    """
    global _scheduler
    if _scheduler is None:
        _scheduler = BackgroundScheduler(daemon=True)
        # Schedule the job to execute every 15 minutes
        _scheduler.add_job(
            scheduler_job,
            trigger="interval",
            minutes=15,
            id="sentiment_analyzer_sync",
            replace_existing=True
        )
        _scheduler.start()
        logger.info("BackgroundScheduler initialized and running every 15 minutes.")
    return _scheduler
