"""SQLite database read/write helpers for the Stock Sentiment Analyzer.

This module provides interface functions to initialize the database, check
if data is present, and save or load headlines and daily aggregates.
"""

import os
import sqlite3
import logging
from typing import List, Dict, Any

# Setup logging
logger = logging.getLogger(__name__)

DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "sentiment.db")


def init_db() -> None:
    """Initializes the SQLite database and creates the tables if they do not exist.

    Sets WAL journal mode for improved multi-threaded concurrency.
    """
    try:
        if not os.path.exists(DB_DIR):
            os.makedirs(DB_DIR, exist_ok=True)

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Enable Write-Ahead Logging (WAL) for better concurrent performance
        cursor.execute("PRAGMA journal_mode=WAL;")

        # Create headlines table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS headlines (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT    NOT NULL,
            date        TEXT    NOT NULL,           -- ISO 8601: YYYY-MM-DD
            headline    TEXT    NOT NULL,
            source      TEXT,
            sentiment   TEXT,                      -- 'positive' | 'negative' | 'neutral'
            score       REAL,                      -- raw FinBERT confidence score
            weighted    REAL                       -- signed weighted score for aggregation
        );
        """)

        # Create daily_summary table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_summary (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker          TEXT    NOT NULL,
            date            TEXT    NOT NULL,
            avg_sentiment   REAL,                  -- daily weighted average [-1, +1]
            close_price     REAL,
            price_change_pct REAL,
            UNIQUE(ticker, date)
        );
        """)

        conn.commit()
        conn.close()
        logger.info("Database initialized successfully at %s.", DB_PATH)
    except sqlite3.Error as e:
        logger.error("Failed to initialize database: %s", e)
        raise


def is_db_empty() -> bool:
    """Checks if the daily_summary table has zero entries.

    Returns:
        bool: True if the table is empty or does not exist, False otherwise.
    """
    if not os.path.exists(DB_PATH):
        return True
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM daily_summary;")
        count = cursor.fetchone()[0]
        conn.close()
        return count == 0
    except sqlite3.Error as e:
        logger.error("Error checking if database is empty: %s", e)
        return True


def save_headlines(headlines: List[Dict[str, Any]]) -> None:
    """Saves a batch of headlines to the database.

    Args:
        headlines: A list of dictionaries representing headlines. Each dict
            must contain 'ticker', 'date', 'headline', 'source', 'sentiment',
            'score', and 'weighted'.
    """
    if not headlines:
        logger.info("No headlines provided to save.")
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        for hl in headlines:
            cursor.execute("""
            INSERT INTO headlines (ticker, date, headline, source, sentiment, score, weighted)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """, (
                hl["ticker"],
                hl["date"],
                hl["headline"],
                hl.get("source"),
                hl.get("sentiment"),
                hl.get("score"),
                hl.get("weighted")
            ))
        conn.commit()
        conn.close()
        logger.info("Successfully saved %d headlines.", len(headlines))
    except sqlite3.Error as e:
        logger.error("Error saving headlines to SQLite: %s", e)
        raise


def save_daily_summaries(summaries: List[Dict[str, Any]]) -> None:
    """Upserts daily summary records into the daily_summary table.

    Args:
        summaries: A list of dictionaries representing daily summaries. Each dict
            must contain 'ticker', 'date', 'avg_sentiment', 'close_price', and
            'price_change_pct'.
    """
    if not summaries:
        logger.info("No daily summaries provided to save.")
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        for summary in summaries:
            cursor.execute("""
            INSERT INTO daily_summary (ticker, date, avg_sentiment, close_price, price_change_pct)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(ticker, date) DO UPDATE SET
                avg_sentiment = excluded.avg_sentiment,
                close_price = excluded.close_price,
                price_change_pct = excluded.price_change_pct;
            """, (
                summary["ticker"],
                summary["date"],
                summary["avg_sentiment"],
                summary["close_price"],
                summary["price_change_pct"]
            ))
        conn.commit()
        conn.close()
        logger.info("Successfully upserted %d daily summaries.", len(summaries))
    except sqlite3.Error as e:
        logger.error("Error upserting daily summaries to SQLite: %s", e)
        raise


def get_headlines(ticker: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """Retrieves list of recent headlines for a specific ticker and date range.

    Args:
        ticker: The stock ticker symbol (e.g., 'AAPL').
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).

    Returns:
        A list of dicts representing raw headline records.
    """
    if not os.path.exists(DB_PATH):
        return []

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
        SELECT date, headline, source, sentiment, score, weighted
        FROM headlines
        WHERE ticker = ? AND date >= ? AND date <= ?
        ORDER BY date DESC, id DESC;
        """, (ticker, start_date, end_date))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        logger.error("Error loading headlines: %s", e)
        return []


def get_daily_summaries(ticker: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """Retrieves aggregate daily records for a specific ticker and date range.

    Args:
        ticker: The stock ticker symbol (e.g., 'AAPL').
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).

    Returns:
        A list of dicts representing daily summaries ordered chronologically.
    """
    if not os.path.exists(DB_PATH):
        return []

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
        SELECT date, avg_sentiment, close_price, price_change_pct
        FROM daily_summary
        WHERE ticker = ? AND date >= ? AND date <= ?
        ORDER BY date ASC;
        """, (ticker, start_date, end_date))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        logger.error("Error loading daily summaries: %s", e)
        return []
