import sqlite3
import os

def check_database():
    db_path = os.path.join("data", "sentiment.db")
    if not os.path.exists(db_path):
        print(f"Database file not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("==================================================")
    print("Database SQLite Verification Summary")
    print("==================================================")

    # Check daily_summary table
    try:
        cursor.execute("SELECT COUNT(*) FROM daily_summary")
        summary_count = cursor.fetchone()[0]
        print(f"Total Rows in daily_summary: {summary_count}")

        cursor.execute("SELECT ticker, date, avg_sentiment, close_price, price_change_pct FROM daily_summary ORDER BY date DESC LIMIT 10")
        summaries = cursor.fetchall()
        print("\nLast 10 Daily Summaries:")
        print(f"{'Ticker':<8} | {'Date':<10} | {'Avg Sent':<10} | {'Close':<10} | {'Change %':<10}")
        print("-" * 55)
        for row in summaries:
            val2 = row[2] if row[2] is not None else 0.0
            val3 = row[3] if row[3] is not None else 0.0
            val4 = row[4] if row[4] is not None else 0.0
            print(f"{row[0]:<8} | {row[1]:<10} | {val2:<10.4f} | {val3:<10.2f} | {val4:<10.2f}%")
    except sqlite3.Error as e:
        print(f"Error checking daily_summary: {e}")

    # Check headlines table
    try:
        cursor.execute("SELECT COUNT(*) FROM headlines")
        headline_count = cursor.fetchone()[0]
        print(f"\nTotal Rows in headlines: {headline_count}")

        cursor.execute("SELECT ticker, date, headline, source, sentiment, score FROM headlines ORDER BY date DESC, id DESC LIMIT 5")
        headlines = cursor.fetchall()
        print("\nLast 5 News Headlines Ingested & Scored:")
        for idx, row in enumerate(headlines, 1):
            print(f"{idx}. [{row[0]} - {row[1]}] (Sent: {row[4]} | Score: {row[5]:.4f}) from {row[3]}")
            print(f"   Headline: \"{row[2][:80]}...\"")
    except sqlite3.Error as e:
        print(f"Error checking headlines: {e}")

    conn.close()

if __name__ == "__main__":
    check_database()
