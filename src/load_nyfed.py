import pandas as pd
import requests
from io import BytesIO
from pathlib import Path

import config

DATA_DIR = Path(config.DATA_DIR)
url = "https://www.newyorkfed.org/medialibrary/media/markets/Dealer_Lists_1960_to_2014.xls"


def pull_nyfed_primary_dealers_list(url, save_cache=True, data_dir=DATA_DIR):
    """
    Download NY Fed primary dealers list from the NYFed website and optionally save it to a cache.
    """
    print(f"Downloading from {url}")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            if save_cache:
                file_path = data_dir / "pulled" / "nyfed_primary_dealers_list.xls"
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


def load_nyfed_primary_dealers_list(url=url, data_dir=DATA_DIR, from_cache=True, save_cache=True, sheet_name=None):
    """
    Load NY Fed primary dealers list from cache or directly if cache is not available.
    """
    file_path = data_dir / "pulled" / "nyfed_primary_dealers_list.xls"
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
        df = load_nyfed_primary_dealers_list(from_cache=True)
    except FileNotFoundError:
        # If not found, download and then load
        pull_nyfed_primary_dealers_list(url=url, save_cache=True)
        df = load_nyfed_primary_dealers_list(from_cache=True)
    print(df.head())
