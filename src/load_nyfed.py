import pandas as pd
import config
from pathlib import Path

DATA_DIR = Path(config.DATA_DIR)
url = "https://www.newyorkfed.org/medialibrary/media/markets/Dealer_Lists_1960_to_2014.xls"


def load_nyfed(
        url=url,
        data_dir=DATA_DIR,
        from_cache=True,
        save_cache=False,
        sheet_name=None
        ):
    """
    Load NY Fed primary dealers list from NYFed website or from cache.
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
    # Pull and save cache of NYFED primary dealers list data
    df = load_nyfed(url=url, data_dir=DATA_DIR, from_cache=False, save_cache=True)
    print(df.head())  # Display the first few rows of the dataframe





# def load_nyfed(url=url, data_dir=DATA_DIR, save_cache=False):
#     """
#     Directly download NY Fed primary dealers list from NYFed website and optionally save to cache.
#     """
#     file_path = data_dir / "pulled" / "nyfed_primary_dealers_list.xls"
    
#     print(f"Downloading from {url}")
#     try:
#         df = pd.read_excel(url)
#         if save_cache:
#             file_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
#             df.to_excel(file_path, index=False)
#             print(f"File saved to cache at {file_path}.")
#     except Exception as e:
#         print(f"Error loading Excel file: {e}")
#         raise
    
#     return df


# if __name__ == "__main__":
#     # Always download and optionally save the cache of NYFED primary dealers list data
#     df = load_nyfed.load_nyfed(url=url, data_dir=DATA_DIR, save_cache=True)
#     print(df.head())  # Display the first few rows of the dataframe