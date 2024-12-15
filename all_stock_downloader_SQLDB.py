import os
import sqlite3
import yfinance as yf
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from datetime import timedelta

# Target Database KRX_DB  SSE_DB  US_DB  IND_DB  JPX_DB
## target_DB can be obtained from my repository : tickers DB folder
## This DB is intended for obtaining all ticker list. If you already have your own list, you can used your own list instead.

target_DB = "KRX_DB"

# SQLite database file path
sqlite_db_path = f"./FINDB/STOCK_DB.db"  ## Modify this code to your own path where STOCK_DB.db is stored.

# Table name
table_name = target_DB

# SQLite connection function
def connect_db():
    if not os.path.exists(sqlite_db_path):
        print("Database file does not exist. Creating new database.")
    return sqlite3.connect(sqlite_db_path)

# SQLite connection
conn = connect_db()
cursor = conn.cursor()

# Function to create the table
def create_table():
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            ticker TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            adj_close REAL,
            volume INTEGER,
            PRIMARY KEY (ticker, date)
        )
    """)
    conn.commit()

# Check if DB file exists and create table
create_table()

# Download data and save to DB
file = pd.read_csv(f"./FINDB/DB/{target_DB}.csv") ## Modify this code to your own path where target_DB is saved
tickers = file["tickers"].tolist()

chunk_size = 100

# Check for existing data in the DB
last_date_query = f"SELECT MAX(date) FROM {table_name}"
cursor.execute(last_date_query)
last_date = cursor.fetchone()[0]

if last_date:
    last_date = datetime.strptime(last_date.split(" ")[0], '%Y-%m-%d').date()
    print(f"Last data date in DB: {last_date}")
else:
    print("No data found in DB. Initializing new download.")
    last_date = (datetime.today() - pd.Timedelta(days=180)).date()

today_date = datetime.today().date()

if today_date.weekday() in [5, 6]:  
    days_to_subtract = today_date.weekday() - 4  
    today_date = today_date - timedelta(days=days_to_subtract)

if last_date >= today_date:
    print("Data is already up to date. Skipping update.")
else:
    print(f"Updating data from {last_date} to {today_date}")

    start_date = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    for i in tqdm(range(0, len(tickers), chunk_size), desc="Downloading new data"):
        chunk = tickers[i:i + chunk_size]
        print(f"Downloading new data for chunk: {chunk}")
        try:
            new_data = yf.download(
                chunk, start=start_date, end=today_date.strftime('%Y-%m-%d'), 
                group_by="ticker", progress=False
            )

            for ticker in chunk:
                if ticker in new_data.columns.get_level_values(0):
                    ticker_data = new_data[ticker].reset_index()
                    ticker_data["ticker"] = ticker
                    ticker_data.drop(columns=["Close"], inplace=True)  # Drop 'Close' column
                    ticker_data.rename(columns={
                        "Date": "date", "Open": "open", "High": "high", 
                        "Low": "low", "Adj Close": "close", "Volume": "volume"
                    }, inplace=True)

                    # Save to database
                    ticker_data.to_sql(table_name, conn, if_exists="append", index=False)
        except Exception as e:
            print(f"Error downloading {chunk}: {e}")

print(f"Data updated in SQLite database: {sqlite_db_path}")


# Close SQLite connection
conn.close()
