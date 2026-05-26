"""Plotly chart and metric layout builder for the Stock Sentiment Analyzer.

This module formats and structures the visual assets shown on the main panel of
the Streamlit dashboard, including the Plotly charts and key performance indicator (KPI) cards.
"""

import logging
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from collections import Counter

# Setup logging
logger = logging.getLogger(__name__)


def render_metrics(df: pd.DataFrame) -> None:
    """Renders the top row of 3 metric cards (KPIs) in the dashboard.

    Args:
        df: Pandas DataFrame containing daily aggregate columns
            ('Date', 'avg_sentiment', 'close_price', 'price_change_pct').
    """
    if df.empty:
        st.warning(
            "⚠️ No data available to calculate metric summaries. "
            "Run pipeline ingest to populate data."
        )
        return

    # Sort chronologically to ensure metrics calculate correctly
    df_sorted = df.sort_values("Date").reset_index(drop=True)

    try:
        # 1. Current Daily Sentiment (most recent day)
        current_row = df_sorted.iloc[-1]
        current_sentiment = float(current_row["avg_sentiment"])

        # Delta: Day-over-day sentiment shift
        if len(df_sorted) > 1:
            prev_sentiment = float(df_sorted.iloc[-2]["avg_sentiment"])
            sentiment_delta = current_sentiment - prev_sentiment
        else:
            sentiment_delta = 0.0

        # 2. 7-Day Average Sentiment (average of last 7 records or fewer if not available)
        last_7_records = df_sorted.tail(7)
        ma_7day = float(last_7_records["avg_sentiment"].mean())

        # Delta: Compare 7-day average with prior 7-day average if we have enough history
        if len(df_sorted) >= 14:
            prior_7_records = df_sorted.iloc[-14:-7]
            prior_ma = float(prior_7_records["avg_sentiment"].mean())
            ma_delta = ma_7day - prior_ma
        else:
            ma_delta = 0.0

        # 3. Stock Price Change % over the selected range
        first_price = float(df_sorted.iloc[0]["close_price"])
        last_price = float(df_sorted.iloc[-1]["close_price"])

        if first_price > 0:
            range_price_change_pct = ((last_price - first_price) / first_price) * 100.0
        else:
            range_price_change_pct = 0.0
        price_diff = last_price - first_price

        # Render metric cards in 3 responsive columns
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Current Daily Sentiment",
                value=f"{current_sentiment:+.2f}",
                delta=f"{sentiment_delta:+.2f} vs yesterday",
                help=(
                    "Weighted sentiment score for the most recent day in date range "
                    "[-1.0 = highly negative, +1.0 = highly positive]"
                )
            )

        with col2:
            st.metric(
                label="7-Day Avg Sentiment",
                value=f"{ma_7day:+.2f}",
                delta=f"{ma_delta:+.2f} vs prior week" if len(df_sorted) >= 14 else None,
                help=(
                    "Simple moving average of the weighted average sentiment score "
                    "over the last 7 calendar days"
                )
            )

        with col3:
            st.metric(
                label="Stock Closing Price",
                value=f"${last_price:,.2f}",
                delta=f"{range_price_change_pct:+.2f}% (${price_diff:+.2f})",
                help=(
                    "Closing price of the stock on the final date. "
                    "Delta represents total price shift across selected range."
                )
            )
    except Exception as e:
        logger.error("Failed to compute and render metric cards: %s", e)
        st.error("⚠️ An error occurred while computing summary metrics. Please check the logs.")


def render_plotly_chart(df: pd.DataFrame) -> None:
    """Renders the interactive dual-axis chart (Sentiment vs Price).

    Left Y-axis: Daily weighted sentiment score (color-coded scatter points).
    Right Y-axis: Stock closing price (charcoal gray line).
    Shared X-axis: Date.

    Args:
        df: Pandas DataFrame containing daily aggregate columns.
    """
    if df.empty:
        st.warning("⚠️ No data available to plot chart. Add credentials or adjust date settings.")
        return

    # Sort chronologically to draw the time-series correctly
    df_sorted = df.sort_values("Date").reset_index(drop=True)

    try:
        # Create a dual-axis subplots figure
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Trace 1: Sentiment Scatter + Lines (Left Y-Axis)
        # Using a marker color scale to color points by sentiment value:
        # Red for negative, Green for positive
        fig.add_trace(
            go.Scatter(
                x=df_sorted["Date"],
                y=df_sorted["avg_sentiment"],
                name="Daily Sentiment",
                mode="lines+markers",
                line=dict(color="#5C7A99", width=2),  # Soft steel blue connector line
                marker=dict(
                    color=df_sorted["avg_sentiment"],
                    colorscale="RdYlGn",
                    cmin=-1.0,
                    cmax=1.0,
                    size=10,
                    showscale=True,
                    colorbar=dict(
                        title=dict(
                            text="Sentiment Score",
                            side="top",
                            font=dict(color="#E2E8F0")
                        ),
                        thickness=15,
                        len=0.7,
                        x=1.15,
                        y=0.5,
                        tickfont=dict(color="#A0AEC0")
                    )
                ),
                hovertemplate="Date: %{x}<br>Sentiment Score: %{y:+.3f}<extra></extra>"
            ),
            secondary_y=False,
        )

        # Trace 2: Stock Closing Price Line (Right Y-Axis) - Silver for dark background visibility
        fig.add_trace(
            go.Scatter(
                x=df_sorted["Date"],
                y=df_sorted["close_price"],
                name="Stock Close Price",
                mode="lines",
                line=dict(color="#D1D5DB", width=3),  # Bright visible gray
                hovertemplate="Date: %{x}<br>Close Price: $%{y:,.2f}<extra></extra>"
            ),
            secondary_y=True,
        )

        # Update labels and styling for Dark Theme Compatibility
        fig.update_xaxes(
            title_text="Date",
            showgrid=True,
            gridcolor="#2D3748",
            title_font=dict(color="#E2E8F0"),
            tickfont=dict(color="#A0AEC0")
        )
        fig.update_yaxes(
            title_text="Sentiment Score ([-1, +1])",
            secondary_y=False,
            range=[-1.05, 1.05],
            showgrid=True,
            gridcolor="#2D3748",
            title_font=dict(color="#E2E8F0"),
            tickfont=dict(color="#A0AEC0")
        )
        fig.update_yaxes(
            title_text="Stock Close Price ($)",
            secondary_y=True,
            showgrid=False,
            title_font=dict(color="#E2E8F0"),
            tickfont=dict(color="#A0AEC0")
        )

        fig.update_layout(
            title=dict(
                text="Daily Weighted Sentiment vs. Stock Closing Price",
                font=dict(size=18, color="#FFFFFF"),
                x=0.01,
                y=0.98
            ),
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(0, 0, 0, 0)",
                font=dict(color="#E2E8F0")
            ),
            margin=dict(l=40, r=80, t=80, b=40),
            plot_bgcolor="rgba(0,0,0,0)",   # Transparent plot background
            paper_bgcolor="rgba(0,0,0,0)",  # Transparent container background
            height=450
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        logger.error("Failed to build and render dual-axis Plotly chart: %s", e)
        st.error(
            "⚠️ Failed to generate interactive Plotly charts. "
            "Please review the application logs."
        )


def render_keyword_bar_chart(headlines_df: pd.DataFrame) -> None:
    """Renders a horizontal bar chart of the top 10 most frequent keywords.

    Args:
        headlines_df: Pandas DataFrame containing 'headline' text.
    """
    if headlines_df.empty:
        return

    # Hardcoded stop words to avoid external dependencies
    stop_words = {
        "a", "about", "above", "after", "again", "against", "all", "am",
        "an", "and", "any", "are", "as", "at", "be", "because", "been",
        "before", "being", "below", "between", "both", "but", "by", "could",
        "did", "do", "does", "doing", "down", "during", "each", "few",
        "for", "from", "further", "had", "has", "have", "having", "he",
        "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
        "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
        "i've", "if", "in", "into", "is", "it", "it's", "its", "itself",
        "let's", "me", "more", "most", "my", "myself", "nor", "of", "on",
        "once", "only", "or", "other", "ought", "our", "ours", "ourselves",
        "out", "over", "own", "same", "she", "she'd", "she'll", "she's",
        "should", "so", "some", "such", "than", "that", "that's", "the",
        "their", "theirs", "them", "themselves", "then", "there", "there's",
        "these", "they", "they'd", "they'll", "they're", "they've", "this",
        "those", "through", "to", "too", "under", "until", "up", "very",
        "was", "we", "we'd", "we'll", "we're", "we've", "were", "what",
        "what's", "when", "when's", "where", "where's", "which", "while",
        "who", "who's", "whom", "why", "why's", "with", "would", "you",
        "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
        "yourselves", "will", "can", "just", "now", "new", "says", "said",
        "one", "two", "also", "may", "apple", "exxon", "chevron", "shell",
        "stock", "stocks", "market", "shares", "investors", "earnings",
        "aapl", "xom", "cvx", "shel", "inc", "corp", "company"
    }

    all_text = " ".join(headlines_df["headline"].tolist()).lower()
    # Extract words
    words = re.findall(r'\b[a-z]{3,}\b', all_text)

    filtered_words = [w for w in words if w not in stop_words]

    if not filtered_words:
        return

    word_counts = Counter(filtered_words).most_common(10)
    # Reverse so the highest count is at the top of the horizontal bar chart
    word_counts.reverse()

    words, counts = zip(*word_counts)

    fig = go.Figure(go.Bar(
        x=counts,
        y=words,
        orientation='h',
        marker=dict(
            color=counts,
            colorscale='Teal',
            showscale=False
        ),
        hovertemplate="Keyword: %{y}<br>Frequency: %{x}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(
            text="Top Trending Keywords in Recent Headlines",
            font=dict(size=18, color="#FFFFFF"),
            x=0.01,
            y=0.95
        ),
        xaxis_title="Frequency",
        yaxis_title="",
        margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=350,
        font=dict(color="#E2E8F0")
    )

    fig.update_xaxes(showgrid=True, gridcolor="#2D3748")
    fig.update_yaxes(showgrid=False)

    st.plotly_chart(fig, use_container_width=True)
