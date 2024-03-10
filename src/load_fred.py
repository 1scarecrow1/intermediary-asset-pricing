import pandas as pd
import pandas_datareader
import config
from pathlib import Path

DATA_DIR = Path(config.DATA_DIR)


def load_fred(
    data_dir=DATA_DIR,
    from_cache=True,
    save_cache=False,
    start="1913-01-01",
    end="2023-10-01",
):
    """
    Must first run pull_and_save_fred_data. If loading from cache, then start
    and dates are ignored
    """
    if from_cache:
        file_path = Path(data_dir) / "pulled" / "fred.parquet"
        # df = pd.read_csv(file_path, parse_dates=["DATE"])
        df = pd.read_parquet(file_path)
        # df = df.set_index("DATE")
    else:
        # Load CPI, nominal GDP, and real GDP data from FRED, seasonally adjusted
        df = pandas_datareader.get_data_fred(
            ["CPIAUCNS", "GDP", "GDPC1"], start=start, end=end
        )
        if save_cache:
            file_dir = Path(data_dir) / "pulled"
            file_dir.mkdir(parents=True, exist_ok=True)
            # df.to_csv(file_dir / "fred_cpi.csv")
            df.to_parquet(file_dir / 'fred.parquet')

    # df.info()
    # df = pd.read_parquet(file_path)
    return df


def resample_quarterly(df):
    """
    Resample the data to quarterly frequency
    """
    df = df.resample('Q').mean()
    return df
    


macro_series_descriptions = {
    'UNRATE': 'Unemployment Rate (Seasonally Adjusted)',
    'NFCI': 'Chicago Fed National Financial Conditions Index',
    'GDPC1':'Real Gross Domestic Product',
    'A191RL1Q225SBEA':' Real Gross Domestic Product Growth',
}

fred_bd_series_descriptions = {
    'BOGZ1FL664090005Q': 'Security Brokers and Dealers; Total Financial Assets, Level',
    'BOGZ1FL664190005Q': 'Security Brokers and Dealers; Total Liabilities, Level',
}

def pull_fred_macro_data(data_dir=DATA_DIR, 
                         start="1969-01-01", end="2024-02-29"):
    try:
        series_keys = list(macro_series_descriptions.keys())
        df = pandas_datareader.data.get_data_fred(series_keys, start=start, end=end)
        file_dir = Path(data_dir) / "pulled"
        file_dir.mkdir(parents=True, exist_ok=True)
        file_path = file_dir / 'fred_macro.parquet'
        df.to_parquet(file_path)
        print(f"Data pulled and saved to {file_path}")
    except Exception as e:
        print(f"Failed to pull or save FRED macro data: {e}")

def load_fred_macro_data(data_dir=DATA_DIR, from_cache=True, 
                         start="1969-01-01", end="2024-02-29"):
    file_path = Path(data_dir) / "pulled" / "fred_macro.parquet"
    try:
        if from_cache and file_path.exists():
            df = pd.read_parquet(file_path)
            print("Loaded data from cache.")
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        print("Cache not found, pulling data...")
        pull_fred_macro_data(data_dir=data_dir, start=start, end=end)
        df = pd.read_parquet(file_path)
    return df


def pull_fred_bd_data(data_dir=DATA_DIR, 
                         start="1969-01-01", end="2024-02-29"):
    try:
        series_keys = list(fred_bd_series_descriptions.keys())
        df = pandas_datareader.data.get_data_fred(series_keys, start=start, end=end)
        file_dir = Path(data_dir) / "pulled"
        file_dir.mkdir(parents=True, exist_ok=True)
        file_path = file_dir / 'fred_bd.parquet'
        df.to_parquet(file_path)
        print(f"Data pulled and saved to {file_path}")
    except Exception as e:
        print(f"Failed to pull or save FRED macro data: {e}")

def load_fred_bd_data(data_dir=DATA_DIR, from_cache=True, 
                         start="1969-01-01", end="2024-02-29"):
    file_path = Path(data_dir) / "pulled" / "fred_bd.parquet"
    try:
        if from_cache and file_path.exists():
            df = pd.read_parquet(file_path)
            print("Loaded data from cache.")
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        print("Cache not found, pulling data...")
        pull_fred_bd_data(data_dir=data_dir, start=start, end=end)
        df = pd.read_parquet(file_path)
    return df


def demo():
    df = load_fred()


if __name__ == "__main__":
    # Pull and save cache of fred data
    _ = load_fred(start="1913-01-01", end="2023-10-01", 
        data_dir=DATA_DIR, from_cache=False, save_cache=True)
    
    # pull and save cache of macroeconomic data
    _ = load_fred_macro_data(start="1969-01-01", end="2024-01-01", 
        data_dir=DATA_DIR, from_cache=False, save_cache=True)
