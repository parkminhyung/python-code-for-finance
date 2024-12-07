
import yfinance as yf
import pandas as pd
from tqdm import tqdm

# target_DB CSV file can be downloaded from the "tickers DB" folder on my GitHub repository
# this step drives all tickers lists from the CSV file. if you have all tickers list file, you can skip this step and use your own ticker list

target_DB = "KRX_DB" 
parquet_file_path = f"./{target_DB}.parquet"

################################################################################################
################################################################################################


# initial steps

file = pd.read_csv("./" + target_DB + ".csv") # laod target_DB csv file from your own directory path
tickers = file["tickers"].tolist() 

# Downloading stock data using yfinance is too slow, so I used the " chunk " method. 

chunk_size = 100 #chunk_size can be adjusted. 
data_frames = []

for i in tqdm(range(0, len(tickers), chunk_size), desc="Downloading chunks"):
    chunk = tickers[i:i + chunk_size]
    print(f"Downloading chunk: {chunk}")
    try:
        # download data
        data = yf.download(chunk, period="6mo", group_by="ticker", progress=False)
        data_frames.append(data)
    except Exception as e:
        print(f"Error downloading {chunk}: {e}")
    
    if i == len(tickers):
        print("download complete")

# Integrate data frame 
final_df = pd.concat(data_frames, axis=1)


# save parquet file on yor own path.
final_df.to_parquet(parquet_file_path)


# load parquet data
parquet_df = pd.read_parquet(parquet_file_path)
