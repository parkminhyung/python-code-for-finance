import os
import yfinance as yf
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# taget_DB  KRX_DB  SSE_DB  US_DB  JPX_DB 
## target_DB is for getting tickers, and target_DB can be obtained from my repository "python code for finance / tickers" folder
## target_DB contains all stock ticker list, so if you have all stock ticker list, you can use your own list. 

target_DB = "SSE_DB"

# Define Parquet file path
parquet_file_path = f"./FINDB/{target_DB}.parquet"

# Check if Parquet file exists
if not os.path.exists(parquet_file_path):
    print(f"Parquet file does not exist. Downloading new data: {parquet_file_path}")
    
    # If file does not exist: Initial data download
    file = pd.read_csv(f"./FINDB/{target_DB}.csv")
    tickers = file["tickers"].tolist()

    chunk_size = 100
    data_frames = []

    for i in tqdm(range(0, len(tickers), chunk_size), desc="Downloading chunks"):
        chunk = tickers[i:i + chunk_size]
        print(f"Downloading chunk: {chunk}")
        try:
            data = yf.download(chunk.tolist(), period="6mo", group_by="ticker", progress=False)
            data_frames.append(data)
        except Exception as e:
            print(f"Error downloading {chunk}: {e}")
    
    # Combine and save data
    final_df = pd.concat(data_frames, axis=1)
    final_df.to_parquet(parquet_file_path)
    print(f"Saved as Parquet file: {parquet_file_path}")

else:
    print(f"Parquet file already exists: {parquet_file_path}")
    
    # If file exists: Update data
    parquet_df = pd.read_parquet(parquet_file_path)
    last_date = parquet_df.index[-1].date()
    today_date = datetime.today().date()

    if last_date == today_date:
        print("Data is already up to date. Skipping update.")
    else:
        print(f"Update required. Last data date: {last_date}, Today's date: {today_date}")
        
        # Determine start date and ticker list for update
        start_date = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        tickers = parquet_df.columns.levels[0]  # Extract ticker list from existing Parquet data

        # Download new data
        new_data_frames = []
        chunk_size = 100

        for i in tqdm(range(0, len(tickers), chunk_size), desc="Downloading new data"):
            chunk = tickers[i:i + chunk_size]
            print(f"Downloading new data for chunk: {chunk}")
            try:
                data = yf.download(
                    chunk.tolist(), start=start_date, end=today_date.strftime('%Y-%m-%d'), 
                    group_by="ticker", progress=False
                )
                new_data_frames.append(data)
            except Exception as e:
                print(f"Error downloading {chunk}: {e}")

        # Combine and update with new data
        if new_data_frames:
            new_data = pd.concat(new_data_frames, axis=1)
            print("New data download completed.")
            
            # Merge old and new data
            updated_df = pd.concat([parquet_df, new_data]).sort_index()
        else:
            print("No new data available.")
            updated_df = parquet_df

        # Save updated data
        updated_df.to_parquet(parquet_file_path)
        print(f"Updated data saved as Parquet file: {parquet_file_path}")

