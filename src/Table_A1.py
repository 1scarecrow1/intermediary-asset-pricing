import pandas as pd

import numpy as np
import datetime as datetime

import config
from pathlib import Path
DATA_DIR = Path(config.DATA_DIR)
OUTPUT_DIR = Path(config.OUTPUT_DIR)

"""
After loading the NY Fed primary dealer list Excel file, clean the data from the 'Dealer Alpha' 
worksheet to replicate Table A.1. This involves formatting dates and indicating companies that 
are still active as primary dealers by marking 'Current' in their end date column.
"""

import load_nyfed

# Obtain the start and end date of primary dealers.
df_dealer_alpha = load_nyfed.load_nyfed_primary_dealers_list(url=load_nyfed.url, data_dir=DATA_DIR, 
                           save_cache=True, sheet_name = 'Dealer Alpha')
df_dealer_alpha.drop(index=0, inplace=True)
df_dealer_alpha.reset_index(drop=True, inplace=True)

df_dealer_alpha.columns = df_dealer_alpha.iloc[0]
df_dealer_alpha.drop(index=0, inplace=True)
df_dealer_alpha = df_dealer_alpha.iloc[:,:3]
df_dealer_alpha.reset_index(drop=True, inplace=True)

df_dealer_alpha['Start Date'] = pd.to_datetime(df_dealer_alpha['Start Date']).dt.strftime('%-m/%-d/%Y')
df_dealer_alpha['End Date'] = pd.to_datetime(df_dealer_alpha['End Date'], errors='coerce')
df_dealer_alpha['End Date'] = df_dealer_alpha['End Date'].apply(
    lambda x: x.strftime('%m/%d/%Y').lstrip("0").replace("/0", "/") if pd.notnull(x) else x
)
df_dealer_alpha['End Date'].fillna('Current', inplace=True)
df_dealer_alpha = df_dealer_alpha.iloc[0:167]
df_dealer_alpha.sort_values(by='Primary Dealer', inplace=True)









