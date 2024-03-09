import pandas as pd
import requests
from io import BytesIO
from pathlib import Path

import config

DATA_DIR = Path(config.DATA_DIR)
url = "https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/downloads/ie_data.xls"


def pull_shiller_pe(url, save_cache=True, data_dir=DATA_DIR):
    """
    Download Shiller's S&P 500 P/E list from the website and optionally save it to a cache.
    """
    print(f"Downloading from {url}")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            if save_cache:
                file_path = data_dir / "pulled" / "shiller_pe.xls"
                file_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
                with open(file_path, 'wb') as file:
                    file.write(response.content)
                print(f"File saved to cache at {file_path}.")
            return BytesIO(response.content)
        else:
            response.raise_for_status()
    except Exception as e:
        print(f"Error downloading Excel file: {e}")
        raise


def load_shiller_pe(url=url, data_dir=DATA_DIR, from_cache=True, save_cache=True, sheet_name=None):
    """
    Load NY Fed primary dealers list from cache or directly if cache is not available.
    """
    file_path = data_dir / "pulled" / "shiller_pe.xls"
    print(f"Attempting to load from: {file_path}")
    
    if from_cache and file_path.exists():
        print("Loading from cache.")
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    else:
        print(f"File not found in cache or from_cache set to False. Downloading from {url}")
        try:
            df = pd.read_excel(url, sheet_name=sheet_name)
            if save_cache:
                file_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
                df.to_excel(file_path, index=False)
                print(f"File saved to cache at {file_path}.")
        except Exception as e:
            print(f"Error loading Excel file: {e}")
            raise
    return df


if __name__ == "__main__":
    # Example usage
    try:
        # Try to load from cache first
        df = load_shiller_pe(from_cache=True)
    except FileNotFoundError:
        # If not found, download and then load
        pull_shiller_pe(url=url, save_cache=True)
        df = load_shiller_pe(from_cache=True)
    print(df.head())
