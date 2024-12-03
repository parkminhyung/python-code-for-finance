
import pandas as pd
import pickle
from pathos.multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import yfinance as yf

# fetch_data 함수 정의
def fetch_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end,progress=False)
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
    # Existing pickle file path
    pickle_path = "./FINDB/" + target_DB + ".pkl"

    # Load existing data
    try:
        with open(pickle_path, "rb") as f:
            mdf = pickle.load(f)
            print("Existing data loaded successfully!")
    except FileNotFoundError:
        print("No existing data found. Creating a new database.")
        mdf = {}

    # If no existing data, load tickers from CSV file
    if not mdf:
        file = pd.read_csv("./FINDB/" + target_DB + ".csv")
        tickers = np.unique(file["tickers"])  
        print("Tickers loaded from CSV successfully!")
    else:
        # If existing data is available, use mdf.keys()
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
        latest_start = (datetime.now() - relativedelta(months=6)).strftime("%Y-%m-%d")  # Default 6 months ago

    print(f"Latest data download period: {latest_start} ~ {latest_end}")

    # Check if data is already up to date
    if latest_start == latest_end:
        print("All data is up-to-date. Skipping download.")
    else:
        # Download latest data
        with Pool(processes=4) as pool:
            results = list(
                tqdm(
                    pool.imap(partial(fetch_data, start=latest_start, end=latest_end), tickers),
                    total=len(tickers),
                    desc="Downloading new data",
                )
            )

        # Merge results
        for ticker, new_data in tqdm(results, desc="Merging data"):
            if new_data is not None:
                if ticker in mdf and isinstance(mdf[ticker], pd.DataFrame):
                    # Merge with existing data
                    mdf[ticker] = pd.concat([mdf[ticker], new_data]).drop_duplicates().sort_index()
                else:
                    # Add new data if ticker doesn't exist
                    mdf[ticker] = new_data

    # Check the merged data
    print(f"Total data size: {len(mdf)}")

    # Save the merged data
    with open(pickle_path, "wb") as f:
        pickle.dump(mdf, f)
        print(f"Updated database saved to {pickle_path}.")
