import pandas as pd
import wrds
import config
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

import requests
from zipfile import ZipFile
from io import BytesIO, StringIO
from pathlib import Path

from load_fred import *


"""
Functions that pull and prepare the data for Table 03 in the intermediary asset pricing paper.
"""

DATA_DIR = Path(config.DATA_DIR)
URL_FRED_2013 = "https://www.federalreserve.gov/releases/z1/20130307/Disk/ltabs.zip"

def date_to_quarter(date):
    """
    Convert a date to a fiscal quarter in the format 'YYYYQ#'.
    """
    year = date.year
    quarter = (date.month - 1) // 3 + 1
    return f"{year}Q{quarter}"


def quarter_to_date(quarter):
    """
    Convert a fiscal quarter in the format 'YYYYQ#' to a date in the format 'YYYY-MM-DD'.
    """
    year = int(quarter[:4])
    quarter = int(quarter[-1])
    month = quarter * 3 
    return datetime(year, month, 1) + pd.DateOffset(months=1) - pd.DateOffset(days=1)


def load_macro_data():
    """
    Function to load macro data from FRED.

    Returns:
    macro_data (DataFrame): DataFrame containing macroeconomic data.
    """
    macro_data = load_fred_macro_data() 
    macro_data = macro_data.rename(columns={'UNRATE': 'unemp_rate', 
                                            'NFCI': 'nfci', 
                                            'GDP': 'nom_gdp', 
                                            'GDPC1': 'real_gdp', 
                                            'A191RL1Q225SBEA': 'real_gdp_growth',
                                            })    
    return macro_data


def load_bd_financials():
    """
    Function to load broker & dealder financial data from FRED.

    Returns:
    macro_data (DataFrame): DataFrame containing financial assets and liabilities of security brokers and dealers.
    """
    bd_financials = load_fred_bd_data(from_cache=False, save_cache=True) 
    bd_financials = bd_financials.rename(columns={'BOGZ1FL664090005Q': 'bd_fin_assets',
                                                  'BOGZ1FL664190005Q': 'bd_liabilities',
                                                  })    
    bd_financials.index = pd.to_datetime(bd_financials.index)
    bd_financials.index.name = 'datafqtr'

    return bd_financials

                                            
def load_fred_past(url=URL_FRED_2013, 
                   data_dir=DATA_DIR, 
                   prn_file_name='ltab127d.prn', 
                   csv_file_name='fred_bd_aem.csv'):
    """
    Download a ZIP file from a given URL, extract a specific .prn file,
    convert it to a .csv file, and save it to a specified directory.

    Parameters:
    - url (str): URL to download the ZIP file.
    - data_dir (Path or str): Directory where the output CSV file should be saved.
    - prn_file_name (str): Name of the .prn file to extract and convert.
    - csv_file_name (str): Name of the output .csv file.

    Returns:
    macro_data (DataFrame): DataFrame containing financial assets and liabilities of security brokers and dealers.
    """

    try:
        # Fetch the ZIP file
        response = requests.get(url)
        response.raise_for_status()

        pulled_dir = Path(data_dir) / "pulled"
        pulled_dir.mkdir(parents=True, exist_ok=True)

        # Process the ZIP file
        with ZipFile(BytesIO(response.content)) as zip_file:
            # Extract the specific .prn file
            prn_path = zip_file.extract(prn_file_name, path=str(pulled_dir))
            print(f"Extracted {prn_file_name} to {prn_path}")

        # Process DataFrame
        with open(prn_path, 'r') as file:
            first_line = file.readline().strip()
        # Split the first line on one or more spaces, considering quotes
        column_names = pd.read_csv(StringIO(first_line), sep=r'\s+', engine='python', header=None).iloc[0]
        column_names = [name.strip('"') for name in first_line.split()]

        df = pd.read_csv(prn_path, sep=r'\s+', skiprows=1, names=column_names, engine='python')
        df = df.apply(lambda x: x.str.strip('"') if x.dtype == "object" else x)
        df.set_index(df.columns[0], inplace=True)

        df.index = df.index.astype(str)
        df.index = df.index.str[:4] + 'Q' + df.index.str[5]
        df = df.loc['1969Q4':'2012Q4']
        df.index = df.index.to_series().apply(quarter_to_date)
        df.index.name = 'datafqtr'

        bd_financials = pd.DataFrame()
        bd_financials['bd_fin_assets'] = df['FL664090005.Q']
        bd_financials['bd_liabilities'] = df['FL664190005.Q']

        csv_path = Path(data_dir) / "pulled" / csv_file_name
        bd_financials.to_csv(csv_path, index=False)
        
        return bd_financials

    except Exception as e:
        print(f"Failed to download or process file: {e}")



def pull_CRSP_Value_Weighted_Index():
    """
    Pulls a value-weighted stock index from the CRSP database.

    Returns:
    - pandas.DataFrame: DataFrame containing the value-weighted stock index data.

    Note:
    This function executes a SQL query to retrieve the value-weighted stock index data from CRSP. 
    The returned DataFrame includes columns for 'date' and 'vwretd' (value-weighted return including dividends).
    """
    
    sql_query = """
        SELECT date, vwretd
        FROM crsp.msi as msi
        WHERE msi.date >= '1970-01-01' AND msi.date <= '2012-12-31'
        """
    
    data = db.raw_sql(sql_query, date_cols=["date"])
    return data