# 📊 Stock Sentiment & Price Analyzer

A real-time financial market sentiment analytics engine. This application aggregates live financial news headlines (via **NewsAPI**) and scores their sentiment using **ProsusAI/FinBERT** (a financial-domain NLP model from Hugging Face). It then correlates these sentiment scores with real-time stock price metrics (via **yfinance**) inside an interactive dual-axis **Streamlit** dashboard.

## 🚀 Live Demo
🌐 **Streamlit Cloud URL:** *[Live Application Demo Placeholder - Add deployed link here]*

---

## 🏛️ System Architecture

```
                               +-----------------------------+
                               |        NewsAPI (News)       |
                               +--------------+--------------+
                                              |
                                              v (headlines)
+------------------------+     +--------------v--------------+     +-----------------------+
|   yfinance (Prices)    |     |   ProsusAI/FinBERT (NLP)    |     |  VADER (NLP Fallback) |
+-----------+------------+     +--------------+--------------+     +-----------+-----------+
            |                                 |                                |
            | (OHLCV Close)                   v (sentiment scores)             v (on-the-fly)
            |                  +--------------v--------------+                 |
            +----------------->|    SQLite: data/sentiment   |                 |
                               +--------------+--------------+                 |
                                              |                                |
                                              v (daily summaries & headlines)  v (VADER re-score)
                               +--------------v--------------------------------v---+
                               |              Streamlit Dashboard UI               |
                               |  - KPI Metric Rows          - Plotly Charts       |
                               |  - Headlines Table          - Spike Warning Alert |
                               +---------------------------------------------------+
```

---

## 📂 Key Features

1. **Dual-Axis Interactive Plotly Chart:** Correlates daily average sentiment score (colored from green/positive to red/negative) on the left Y-axis with stock closing prices (charcoal gray line) on the right Y-axis.
2. **Three-Card Metric Row:** Displays the current daily weighted average sentiment, the 7-day moving average sentiment, and the stock price change percentage over the selected date range.
3. **Sentiment Spike Alerts:** Automatically scans for day-over-day sentiment fluctuations exceeding **0.3** and triggers a clear warning banner.
4. **Recent Headlines Table:** Displays the 20 most recent headlines with metadata and visual status-coded sentiment badges (Positive, Negative, Neutral).
5. **Non-Blocking Orchestrator Scheduler:** Integrates a daemonized `BackgroundScheduler` running every 15 minutes to run data synchronization, coupled with an smart multi-threaded bootloader to backfill records on startup.

---

## 🔬 How it Works: FinBERT vs. VADER

The dashboard supports two distinct sentiment analysis models:
- **FinBERT (Primary):** A language model based on BERT, fine-tuned specifically on financial corpora (Financial PhraseBank). It excels at detecting complex financial contexts (e.g., distinguishing between standard "growth" vs. "interest rate hikes" connotations).
- **VADER (Fallback & Comparative View):** A fast, rule-based lexicon and sentiment analyzer. In VADER comparative mode, the dashboard dynamically re-scores headlines and aggregates metrics on-the-fly to show how the deep-learning model stacks up against traditional lexicon techniques.

---

## 🛠️ Local Setup & Execution Guide

### Prerequisites
- Python 3.11 or 3.12 (standard 64-bit environment)
- A free API key from [NewsAPI](https://newsapi.org/)

### Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/stock-sentiment-analyzer.git
   cd stock-sentiment-analyzer
   ```

2. **Configure Environment Variables:**
   Copy the example file to `.env`:
   ```bash
   cp .env.example .env
   ```
   Open the `.env` file and insert your API key:
   ```env
   NEWS_API_KEY=your_real_newsapi_key_here
   ```

3. **Install Dependencies:**
   We recommend utilizing a Python virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Launch the Application:**
   Run the Streamlit application command:
   ```bash
   streamlit run app/main.py
   ```

---

## 🐳 Containerization & Deployment

### Build the Docker Image
To containerize the application, execute:
```bash
docker build -t stock-sentiment .
```

### Run Container with Database Persistence (Volume Mount)
To ensure the SQLite data directory is preserved across container lifecycles, launch the container utilizing a local directory host volume mount:
```bash
# Create local data directory
mkdir -p data

# Run docker mounting the local data folder
docker run -d \
  -p 8501:8501 \
  --env-file .env \
  -v "$(pwd)/data:/usr/src/app/data" \
  --name stock-sentiment-container \
  stock-sentiment
```

---

## 🧪 Testing and CI Compliance

The project is equipped with automated unit testing suites running inside **GitHub Actions** (`ci.yml`) on every push or pull request:
- **Linter Check:** Strict PEP 8 enforcement via `flake8` (max-line-length=100).
- **Test Runner:** Executed using `pytest`.

### Run Checks Locally
```bash
# Style guide compliance
flake8 pipeline/ app/ tests/ --max-line-length=100

# Executing test cases
pytest tests/ -v --tb=short
```

---
*Created and maintained by Saurabh Vora · May 2026 · Stock Sentiment Analyzer*
