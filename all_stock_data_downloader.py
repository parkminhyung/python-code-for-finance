import pandas as pd
import pickle
from tqdm import tqdm
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import yfinance as yf

# Define fetch_data function
def fetch_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        df = df[["Open", "High", "Low", "Adj Close"]]
        df = df.rename(
            columns={
                "Adj Close": "close",
                "Open": "open",
                "High": "high",
                "Low": "low"
            })
        df["ticker"] = ticker
        return ticker, df

    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return ticker, None  

# target_DB : KRX_DB  SSE_DB  US_DB
# Download DB file from my github repository (ticker DB folder)
target_DB = "KRX_DB"

if __name__ == "__main__":
    # Path to the existing pickle file
    pickle_path = "./FINDB/" + target_DB + ".pkl"

    # Load existing data
    try:
        with open(pickle_path, "rb") as f:
            mdf = pickle.load(f)
            print("Existing data loaded successfully!")
    except FileNotFoundError:
        print("No existing data found. Creating a new database.")
        mdf = {}

    # If there is no existing data, load tickers from CSV
    if not mdf:
        file = pd.read_csv("./FINDB/" + target_DB + ".csv")
        tickers = np.unique(file["tickers"].tolist())  
        print("Tickers loaded from CSV!")
    else:
        # If existing data is found, use mdf.keys()
        tickers = list(mdf.keys())

    # Calculate the latest date
    latest_dates = []
    for ticker, df in mdf.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            latest_dates.append(df.index.max())

    latest_end = datetime.now().strftime("%Y-%m-%d") 

    if latest_dates:
        latest_start = max(latest_dates).strftime("%Y-%m-%d")  # Most recent date
    else:
        latest_start = (datetime.now() - relativedelta(months=6)).strftime("%Y-%m-%d")  # Default is 6 months ago

    print(f"Date range for downloading latest data: {latest_start} ~ {latest_end}")

    # Check if the data is already up-to-date
    if latest_start == latest_end:
        print("All data is up-to-date. Skipping download.")
    else:
        # Download the latest data
        results = []
        for ticker in tqdm(tickers, desc="Downloading new data"):
            results.append(fetch_data(ticker, start=latest_start, end=latest_end))

        # Merge results
        for ticker, new_data in tqdm(results, desc="Merging data"):
            if new_data is not None:
                if ticker in mdf and isinstance(mdf[ticker], pd.DataFrame):
                    # Merge with existing data
                    mdf[ticker] = pd.concat([mdf[ticker], new_data]).drop_duplicates().sort_index()
                else:
                    # Add new data if ticker does not exist
                    mdf[ticker] = new_data

    # Check the size of the merged data
    print(f"Total data size: {len(mdf)}")

    # Save the merged data
    with open(pickle_path, "wb") as f:
        pickle.dump(mdf, f)
        print(f"Updated database saved to {pickle_path}.")
