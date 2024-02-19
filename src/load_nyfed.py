import pandas as pd
import config
from pathlib import Path

DATA_DIR = Path(config.DATA_DIR)
url = "https://www.newyorkfed.org/medialibrary/media/markets/Dealer_Lists_1960_to_2014.xls"


def load_nyfed(
        url=url,
        data_dir=DATA_DIR,
        from_cache=True,
        save_cache=False
        ):
    """
    Load NY Fed primary dealers list from NYFed website or from cache.
    """
    
    if from_cache:
        file_path = Path(data_dir) / "pulled" / "nyfed_primary_dealers_list.xls"
        df = pd.read_excel(file_path)
    else:
        try:
            df = pd.read_excel(url)
            if save_cache:
                file_dir = Path(data_dir) / "pulled"
                file_dir.mkdir(parents=True, exist_ok=True)
                df.to_excel(file_dir / "nyfed_primary_dealers_list.xls")
        except Exception as e:
            print("Error loading Excel file:", e)
    
    return df


if __name__ == "__main__":
    # Pull and save cache of NYFED primary dealers list data
    _ = load_nyfed(
        url=url, data_dir=DATA_DIR, from_cache=False, save_cache=True)
