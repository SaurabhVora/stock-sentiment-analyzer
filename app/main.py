"""Streamlit entry point for the Stock Sentiment Analyzer.

This dashboard integrates the frontend UI elements, loads database configurations,
triggers scheduled background threads, and renders metrics, Plotly charts,
recent headline grids, and day-over-day spike alerts.
"""

import logging
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Any

import pandas as pd
import streamlit as st

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from pipeline.storage import init_db, is_db_empty, get_headlines, get_daily_summaries  # noqa: E402
from app.scheduler import start_scheduler, trigger_backfill, run_pipeline_for_ticker  # noqa: E402
from app.dashboard import (  # noqa: E402
    render_metrics,
    render_plotly_chart,
    render_keyword_bar_chart,
)

# Configure logging standard PEP 8 compliance
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Streamlit Page Configuration
st.set_page_config(
    page_title="Stock Sentiment Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- CACHED DATA FETCHERS -----------------


@st.cache_data(ttl=900)
def _fetch_db_data_cached(
    ticker: str, start_date_str: str, end_date_str: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Caches direct SQLite queries to prevent database lockups under high traffic."""
    raw_headlines = get_headlines(ticker, start_date_str, end_date_str)
    raw_summaries = get_daily_summaries(ticker, start_date_str, end_date_str)
    return raw_headlines, raw_summaries


def get_cached_dashboard_data(
    ticker: str, start_date_str: str, end_date_str: str, model: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads and computes the dashboard summaries and headlines based on the selected model.

    Utilizes Streamlit st.cache_data synchronized with the 15-minute scheduler refresh rate.
    Handles cold-start caching by bypassing the cache if the database was recently populated.
    """
    logger.info(
        "Loading cached dashboard data for %s from %s to %s via %s",
        ticker,
        start_date_str,
        end_date_str,
        model,
    )

    # 1. Fetch data from SQLite (cached)
    raw_headlines, raw_summaries = _fetch_db_data_cached(ticker, start_date_str, end_date_str)

    # Cold-start handling: If the cache contains empty results (from when the DB
    # was empty during boot) but the database actually has data now, invalidate
    # the cache and fetch the fresh records.
    if not raw_headlines or not raw_summaries:
        raw_headlines_uncached = get_headlines(ticker, start_date_str, end_date_str)
        raw_summaries_uncached = get_daily_summaries(ticker, start_date_str, end_date_str)
        if raw_headlines_uncached and raw_summaries_uncached:
            st.cache_data.clear()  # Clear cache to discard the empty values
            raw_headlines = raw_headlines_uncached
            raw_summaries = raw_summaries_uncached
        else:
            return pd.DataFrame(), pd.DataFrame()

    headlines_df = pd.DataFrame(raw_headlines)
    summaries_df = pd.DataFrame(raw_summaries)

    # 2. Re-calculate on-the-fly if using VADER comparative mode
    if model == "VADER":
        from pipeline.sentiment import score_headlines_vader
        texts = headlines_df["headline"].tolist()
        vader_scores = score_headlines_vader(texts)

        # Override headlines metrics
        headlines_df["sentiment"] = [v["label"] for v in vader_scores]
        headlines_df["score"] = [v["score"] for v in vader_scores]
        headlines_df["weighted"] = [v["weighted"] for v in vader_scores]

        # Re-group by date to calculate average VADER sentiment
        vader_agg = headlines_df.groupby("date")["weighted"].mean().reset_index()
        vader_agg.rename(columns={"weighted": "vader_avg"}, inplace=True)

        # Merge new VADER sentiment into summaries_df
        summaries_df = summaries_df.merge(vader_agg, on="date", how="left")
        summaries_df["avg_sentiment"] = summaries_df["vader_avg"].fillna(0.0)
        summaries_df.drop(columns=["vader_avg"], inplace=True)

    if not summaries_df.empty and "date" in summaries_df.columns:
        summaries_df.rename(columns={"date": "Date"}, inplace=True)

    return summaries_df, headlines_df


# ----------------- BOOTLOADER / SCHEDULER INITIALIZATION -----------------

try:
    init_db()
    if is_db_empty():
        # Empty DB, trigger initial backfill thread
        st.warning(
            "📊 Initialize Database: Database is empty. Spawning background 7-day ingest..."
        )
        trigger_backfill()
        st.info(
            "🔄 Backfilling AAPL, XOM, CVX, SHEL data in background. "
            "Refreshing in a few seconds..."
        )
    start_scheduler()
except Exception as e:
    logger.critical("Database or Scheduler failed during boot phase: %s", e)
    st.error(
        "⚠️ Dashboard Critical Error: Scheduler bootloader failed. "
        "Check environment configuration."
    )

# ----------------- SIDEBAR LAYOUT -----------------

st.sidebar.title("📊 Control Panel")
st.sidebar.markdown("Customize your parameters and data views.")

# Ticker Input Search
tracked_tickers = ["AAPL", "XOM", "CVX", "SHEL"]
selected_ticker = st.sidebar.selectbox(
    "Stock Ticker",
    options=tracked_tickers,
    index=0,
    help="Select one of the pre-tracked stocks to view sentiment."
)

ticker = selected_ticker

# Date Range Picker
thirty_days_ago = datetime.now() - timedelta(days=30)
selected_dates = st.sidebar.date_input(
    "Date Range Picker",
    value=(thirty_days_ago, datetime.now()),
    max_value=datetime.now(),
    help="Choose the range of headlines and stock prices to render."
)

if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
else:
    start_date = thirty_days_ago.date()
    end_date = datetime.now().date()

start_date_str = start_date.strftime("%Y-%m-%d")
end_date_str = end_date.strftime("%Y-%m-%d")

# Model Selector (FinBERT vs VADER comparative view)
selected_model = st.sidebar.radio(
    "Sentiment Classifier Model",
    options=["FinBERT", "VADER"],
    index=0,
    help=(
        "FinBERT is custom-tuned for financial contexts. "
        "VADER is a rule-based dictionary fallback."
    )
)

# Manual Ingest Button for instant synchronization
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Sync Live Data Now"):
    with st.sidebar.status("Fetching live news & prices...", expanded=True) as status:
        try:
            run_pipeline_for_ticker(ticker, start_date_str, end_date_str)
            status.update(label="Sync Completed!", state="complete", expanded=False)
            st.cache_data.clear()  # Clear cache to display newly loaded data
            st.rerun()
        except Exception as ex:
            status.update(label=f"Sync Failed: {ex}", state="error", expanded=True)

# ----------------- MAIN PANEL LAYOUT -----------------

st.title("📈 Stock Sentiment & Price Analyzer")
st.markdown(
    "This dashboard correlates financial news headlines sentiment "
    "(extracted using ProsusAI/FinBERT) "
    "with live stock price closing figures downloaded from Yahoo Finance."
)

# Fetch Dashboard data
try:
    daily_df, headlines_df = get_cached_dashboard_data(
        ticker, start_date_str, end_date_str, selected_model
    )
except Exception as err:
    logger.error("Failed to load cached dashboard data: %s", err)
    daily_df, headlines_df = pd.DataFrame(), pd.DataFrame()
    st.error(
        "⚠️ An error occurred while retrieving stock or sentiment summaries. "
        "Check your .env file."
    )

# 1. Metric KPI Cards Row
if not daily_df.empty:
    render_metrics(daily_df)
else:
    st.info(
        "ℹ️ No aggregate price/sentiment logs found. "
        "Use the 'Sync Live Data Now' button to seed details."
    )

# 2. Dual-Axis Interactive Plotly Chart
if not daily_df.empty:
    render_plotly_chart(daily_df)

# 2.5 Word Frequency Bar Chart
if not headlines_df.empty:
    render_keyword_bar_chart(headlines_df)

# 3. Spike Alert Banner
if not daily_df.empty:
    daily_sorted = daily_df.sort_values("Date").reset_index(drop=True)
    spikes = []
    for i in range(1, len(daily_sorted)):
        prev_sent = float(daily_sorted.iloc[i - 1]["avg_sentiment"])
        curr_sent = float(daily_sorted.iloc[i]["avg_sentiment"])
        date_str = str(daily_sorted.iloc[i]["Date"])
        prev_date_str = str(daily_sorted.iloc[i - 1]["Date"])
        diff = abs(curr_sent - prev_sent)

        if diff > 0.3:
            spikes.append((prev_date_str, date_str, diff, prev_sent, curr_sent))

    if spikes:
        alert_text = "🚨 **Sentiment Spike Warning (Day-over-Day Shift > 0.3):**\n\n"
        for p_date, c_date, dev, p_val, c_val in spikes:
            alert_text += (
                f"- **{c_date}** (vs {p_date}): Sentiment shifted by **{dev:+.2f}** "
                f"(from {p_val:+.2f} to {c_val:+.2f})\n"
            )
        st.warning(alert_text)

# 4. Recent Headlines Table
st.subheader("📰 Recent Headlines Ledger")
if not headlines_df.empty:
    recent_headlines = headlines_df.head(20)

    # Styling labels into CSS badges
    badge_colors = {
        "positive": (
            "background-color: #D4EDDA; color: #155724; border-radius: 12px; "
            "padding: 3px 10px; font-size: 11px; font-weight: bold; border: 1px solid #C3E6CB;"
        ),
        "negative": (
            "background-color: #F8D7DA; color: #721C24; border-radius: 12px; "
            "padding: 3px 10px; font-size: 11px; font-weight: bold; border: 1px solid #F5C6CB;"
        ),
        "neutral": (
            "background-color: #E2E3E5; color: #383D41; border-radius: 12px; "
            "padding: 3px 10px; font-size: 11px; font-weight: bold; border: 1px solid #D6D8DB;"
        )
    }

    # Construct HTML table for modern layout
    html_markup = """<div style='overflow-x:auto;'>
<table style='width:100%; border-collapse: collapse; text-align: left;
font-family: sans-serif; font-size: 14px;'>
    <thead>
        <tr style='border-bottom: 2px solid #DEE2E6; background-color: #F8F9FA;'>
            <th style='padding: 12px; font-weight: bold;'>Date</th>
            <th style='padding: 12px; font-weight: bold;'>Headline</th>
            <th style='padding: 12px; font-weight: bold;'>Source</th>
            <th style='padding: 12px; font-weight: bold; text-align: center;'>Sentiment</th>
            <th style='padding: 12px; font-weight: bold; text-align: right;'>Score</th>
        </tr>
    </thead>
    <tbody>"""

    for _, row in recent_headlines.iterrows():
        lbl = str(row["sentiment"]).lower()
        badge_style = badge_colors.get(lbl, badge_colors["neutral"])
        score_val = float(row["score"])

        html_markup += f"""<tr style='border-bottom: 1px solid #E9ECEF;'>
    <td style='padding: 10px; white-space: nowrap;'>{row["date"]}</td>
    <td style='padding: 10px; font-weight: 500;'>{row["headline"]}</td>
    <td style='padding: 10px; color: #6C757D;'>{row["source"]}</td>
    <td style='padding: 10px; text-align: center;'>
        <span style='{badge_style}'>{lbl.upper()}</span>
    </td>
    <td style='padding: 10px; text-align: right; font-weight: bold;
    color: #495057;'>{score_val:.4f}</td>
</tr>"""

    html_markup += """</tbody>
</table>
</div>"""
    st.markdown(html_markup, unsafe_allow_html=True)
else:
    st.info("📰 No individual headlines logged yet in this date range.")
