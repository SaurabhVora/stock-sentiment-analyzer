# Stock Sentiment Analyzer — Agent Instructions

> Place this file at your project root. Every Antigravity agent reads it before starting any task.

---

## Project overview

This is a real-time stock sentiment analyzer built by Saurabh Vora. It fetches financial news
headlines from NewsAPI, scores them with FinBERT (a finance-domain NLP model from Hugging Face),
correlates sentiment with live stock price data from yfinance, and displays everything in an
interactive Streamlit dashboard with Plotly charts.

**Target audience:** Recruiters and hiring managers evaluating data science / AI-ML fresher candidates.
This project must be clean, well-documented, and demoable from a public Streamlit Cloud URL.

---

## Tech stack (strict — do not deviate without asking)

| Layer        | Tool / Library                                  |
|--------------|-------------------------------------------------|
| Data: News   | NewsAPI (`newsapi-python`)                      |
| Data: Prices | `yfinance`                                      |
| NLP model    | `ProsusAI/finbert` via Hugging Face `pipeline`  |
| Fallback NLP | `vaderSentiment` (for comparison only)          |
| Data wrangling | `pandas`, `numpy`                             |
| Storage      | SQLite via `sqlite3` (built-in, no ORM)         |
| Scheduler    | `APScheduler`                                   |
| Frontend     | `streamlit`, `plotly`                           |
| Deployment   | Docker + Streamlit Cloud + GitHub Actions CI    |

---

## Project folder structure

Always maintain this exact structure. Do not create files outside of it without asking.

```
stock-sentiment-analyzer/
├── AGENTS.md                  ← this file
├── README.md
├── requirements.txt
├── Dockerfile
├── .github/
│   └── workflows/
│       └── ci.yml             ← GitHub Actions: lint + test on push
├── app/
│   ├── main.py                ← Streamlit entry point
│   ├── dashboard.py           ← all Plotly chart builders
│   └── scheduler.py           ← APScheduler refresh job
├── pipeline/
│   ├── fetch_news.py          ← NewsAPI fetch + clean
│   ├── fetch_prices.py        ← yfinance OHLCV fetch
│   ├── sentiment.py           ← FinBERT + VADER scoring
│   └── storage.py             ← SQLite read/write helpers
├── tests/
│   ├── test_fetch_news.py
│   ├── test_fetch_prices.py
│   └── test_sentiment.py
└── .env.example               ← API key placeholders (never commit real keys)
```

---

## Coding standards

- **Language:** Python 3.11+
- **Style:** PEP 8. Max line length 100 characters.
- **Type hints:** Required on every function signature. No bare `def foo(x):` — always `def foo(x: str) -> pd.DataFrame:`.
- **Docstrings:** Google-style docstrings on every function and class. One-liner is fine for trivial helpers.
- **Error handling:** All external API calls (NewsAPI, yfinance) must be wrapped in `try/except` with informative logging. Never let an API failure crash the Streamlit app — catch and surface a user-friendly `st.warning()` instead.
- **No hardcoded credentials:** API keys go in `.env` only, loaded with `python-dotenv`. Raise a `ValueError` with a clear message if a required env var is missing.
- **No global mutable state:** Pass data explicitly between functions. No module-level variables that accumulate state.

---

## Sentiment scoring rules

- Primary model: `ProsusAI/finbert` loaded via `transformers.pipeline("text-classification")`.
- Score each headline individually. Return a dict: `{"label": "positive"|"negative"|"neutral", "score": float}`.
- Daily aggregate: compute the **weighted average** of confidence scores — positive scores are positive, negative scores are negated. Result is a float in `[-1.0, +1.0]`.
- VADER is a secondary model used only for comparison in the README. Do not replace FinBERT with VADER in the main pipeline.
- Batch headlines in groups of 16 when calling FinBERT to avoid OOM on CPU.

---

## Database schema

SQLite file: `data/sentiment.db` (create the `data/` directory if it doesn't exist).

```sql
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

CREATE TABLE IF NOT EXISTS daily_summary (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT    NOT NULL,
    date            TEXT    NOT NULL,
    avg_sentiment   REAL,                  -- daily weighted average [-1, +1]
    close_price     REAL,
    price_change_pct REAL,
    UNIQUE(ticker, date)
);
```

Always use parameterised queries (`?` placeholders). Never use f-string SQL.

---

## Streamlit dashboard requirements

The dashboard must have these components in order:

1. **Sidebar:** Ticker search input (default: `AAPL`), date range picker (default: last 30 days), model selector (FinBERT / VADER toggle for comparison view).
2. **Metric row (3 cards):** Current daily sentiment score, 7-day average sentiment, stock price change % over selected range.
3. **Dual-axis Plotly chart:** Left y-axis = sentiment score (line, colored by positive/negative), right y-axis = closing price (line, gray). Shared x-axis = date.
4. **Headline table:** Columns: Date, Headline, Source, Sentiment (color-coded badge), Score. Show the 20 most recent rows.
5. **Spike alert banner:** If any day has a sentiment change > 0.3 from the prior day, show an `st.warning()` banner listing those dates.

Use `st.cache_data(ttl=900)` on all data-fetch functions (15-minute cache matching the refresh interval).

---

## Auto-refresh schedule

`APScheduler` runs a background job every 15 minutes:
1. Fetch latest headlines for all tracked tickers.
2. Score headlines with FinBERT.
3. Fetch latest price data.
4. Upsert results into SQLite.

The scheduler starts in `scheduler.py` and is imported into `main.py`. Use `BackgroundScheduler` (not blocking).

---

## GitHub Actions CI (`ci.yml`)

On every push to `main` or any pull request:

```yaml
steps:
  - Checkout code
  - Set up Python 3.11
  - Install dependencies from requirements.txt
  - Run: flake8 pipeline/ app/ tests/ --max-line-length=100
  - Run: pytest tests/ -v --tb=short
```

Do not add deployment steps to CI. Deployment is handled automatically by Streamlit Cloud watching the `main` branch.

---

## Docker

The `Dockerfile` must:
- Use `python:3.11-slim` as the base image.
- Copy only necessary files (use `.dockerignore` to exclude `data/`, `.env`, `__pycache__`, `.git`).
- Install dependencies via `pip install --no-cache-dir -r requirements.txt`.
- Expose port `8501`.
- Set `CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]`.

---

## README requirements

The README must include:
1. One-line project description.
2. A live demo link (Streamlit Cloud URL — add after deployment).
3. Architecture diagram (simple ASCII or linked image).
4. Setup instructions: clone → create `.env` → `pip install -r requirements.txt` → `streamlit run app/main.py`.
5. A "How it works" section explaining FinBERT vs VADER comparison.
6. A demo GIF or screenshot of the dashboard (add after first working build).

---

## What agents should NOT do

- Do not install any library not listed in the tech stack without asking first.
- Do not use an ORM (SQLAlchemy, etc.) — raw `sqlite3` only.
- Do not use `st.experimental_*` APIs — use stable Streamlit APIs only.
- Do not store API keys anywhere in code, even in comments.
- Do not create Jupyter notebooks — this is a production-style project, not an exploratory one.
- Do not use `print()` for logging — use Python's `logging` module with `INFO` level by default.

---

## Deployment checklist (run before marking any task done)

- [ ] `flake8` passes with zero errors
- [ ] All `pytest` tests pass
- [ ] `Dockerfile` builds successfully with `docker build -t stock-sentiment .`
- [ ] `.env.example` is up to date with all required keys
- [ ] No real API keys committed to git (`git log --all --full-history -- .env`)
- [ ] README has the live Streamlit Cloud URL filled in

---

*Last updated: May 2026 · Saurabh Vora · Stock Sentiment Analyzer*
