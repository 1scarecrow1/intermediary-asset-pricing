import pandas as pd
import config
from datetime import datetime

import requests
import pandas_datareader.data as web
from zipfile import ZipFile
from io import BytesIO, StringIO
from pathlib import Path

from load_fred import *

"""
Functions that pull and prepare the data for Table 03.
"""

DATA_DIR = Path(config.DATA_DIR)
URL_FRED_2013 = "https://www.federalreserve.gov/releases/z1/20130307/Disk/ltabs.zip"
URL_SHILLER = "https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/downloads/ie_data.xls"


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


def fetch_financial_data_quarterly(gvkey, start_date, end_date, db):
    """
    Fetch financial data for a given ticker and date range from the CCM database in WRDS.
    
    Parameters:
    gvkey: The gvkey symbol for the company.
    start_date: The start date for the data in YYYY-MM-DD format.
    end_date: The end date for the data in YYYY-MM-DD format or 'Current'.
    
    Returns:
    A DataFrame containing the financial data.
    """

    if not gvkey:  # Skip if no ticker is available
        return pd.DataFrame()
    
    # Convert 'Current' to today's date if necessary
    if end_date == 'Current':
        end_date = datetime.today().strftime('%Y-%m-%d')
    
    # Convert start and end dates to datetime objects
    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)
    
    # Format start and end quarters
    start_qtr = date_to_quarter(start_date_dt)
    end_qtr = date_to_quarter(end_date_dt)

    query = f"""
    SELECT datafqtr, atq AS total_assets, ltq AS book_debt, 
            COALESCE(teqq, ceqq + COALESCE(pstkq, 0) + COALESCE(mibnq, 0)) AS book_equity, 
            cshoq*prccq AS market_equity, gvkey, conm
    FROM comp.fundq as cst
    WHERE cst.gvkey = '{str(gvkey).zfill(6)}'
    AND cst.datafqtr BETWEEN '{start_qtr}' AND '{end_qtr}'
    AND indfmt='INDL'
    AND datafmt='STD'
    AND popsrc='D'
    AND consol='C'
    """
    data = db.raw_sql(query)

    return data


def fetch_data_for_tickers(ticks, db):
    """
    Function to fetch financial data for a list of tickers.

    Parameters:
    ticks (DataFrame): DataFrame containing ticker information including gvkey, start date, and end date.

    Returns:
    prim_dealers (DataFrame): DataFrame containing fetched financial data.
    empty_tickers (list): List of tickers for which no data could be fetched.
    """

    empty_tickers = []
    prim_dealers = pd.DataFrame()

    # Iterate over DataFrame rows and fetch data for each ticker
    for index, row in ticks.iterrows():
        gvkey = row['gvkey']
        start_date = row['Start Date']
        end_date = row['End Date']  # Formatting date for the query

        # Fetch financial data for the ticker if available
        new_data = fetch_financial_data_quarterly(gvkey, start_date, end_date, db)
        if isinstance(new_data, tuple):
            empty_tickers.append({row['Ticker']: gvkey})
        else:
            prim_dealers = pd.concat([new_data, prim_dealers], axis=0)
    
    return prim_dealers, empty_tickers


def load_macro_data():
    """
    Function to load macro data from FRED.

    Returns:
    macro_data (DataFrame): DataFrame containing macroeconomic data.
    """
    macro_data = load_fred_macro_data() 
    macro_data = macro_data.rename(columns={'UNRATE': 'unemp_rate', 
                                            'NFCI': 'nfci', 
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
    bd_financials = load_fred_bd_data(from_cache=False) 
    bd_financials = bd_financials.rename(columns={'BOGZ1FL664090005Q': 'bd_fin_assets',
                                                  'BOGZ1FL664190005Q': 'bd_liabilities',
                                                  })    
    bd_financials.index = pd.to_datetime(bd_financials.index)
    bd_financials = bd_financials.resample('Q').last()
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
        df = df.loc['1968Q4':'2012Q4']
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


def fetch_ff_factors(start_date, end_date):
    """
    Fetches Fama-French research data factors, adjusts dates to end of the month,
    resamples to quarterly frequency, and renames columns for ease of use.
    """
    rawdata = web.DataReader('F-F_Research_Data_5_Factors_2x3', data_source='famafrench', start=start_date, end=end_date)
    ff_facs = rawdata[0] / 100
    ff_facs.rename(columns={'Mkt-RF': 'mkt_ret'}, inplace=True)
    
    return ff_facs

def pull_shiller_pe(url=URL_SHILLER, data_dir=DATA_DIR):
    """
    Download Shiller's S&P 500 P/E list from the website and save it to a cache.
    """
    print(f"Downloading and caching from {url}")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            file_path = data_dir / "pulled" / "shiller_pe.xls"
            file_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
            with open(file_path, 'wb') as file:
                file.write(response.content)
            print(f"Data saved to cache at {file_path}")
        else:
            response.raise_for_status()
    except Exception as e:
        print(f"Error downloading or saving the Shiller PE data: {e}")
        raise

def load_shiller_pe(url=URL_SHILLER, data_dir=DATA_DIR, from_cache=True):
    """
    Load Shiller PE data from cache or directly if cache is not available.
    """
    file_path = data_dir / "pulled" / "shiller_pe.xls"
    if from_cache and file_path.exists():
        print("Loading data from cache.")
        df = pd.read_excel(file_path, sheet_name='Data', skiprows=7, usecols="A,M")
    else:
        print("Cache not found, pulling data...")
        pull_shiller_pe(url, data_dir)
        df = pd.read_excel(file_path, sheet_name='Data', skiprows=7, usecols="A,M")

    return df


def pull_CRSP_Value_Weighted_Index(db):
    """
    Pulls a value-weighted stock index from the CRSP database.
    """
    sql_query = """
        SELECT date, vwretd
        FROM crsp.msi as msi
        WHERE msi.date >= '1969-01-01' AND msi.date <= '2024-02-29'
        """
    
    data = db.raw_sql(sql_query, date_cols=["date"])
    return data