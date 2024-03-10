import pandas as pd
import wrds
import config
from datetime import datetime
import unittest
import matplotlib.pyplot as plt
import numpy as np
import load_CRSP_stock

from Table03Prep import *
from Table02Prep import prim_deal_merge_manual_data_w_linktable

"""
Reads in manual dataset for primary dealers and holding companies and matches it with linkhist entry for company. 
Compiles and prepares this data to produce Table 03 from intermediary asset pricing paper in LaTeX format.
Also creates a summary statistics table and figure in LaTeX format.
Performs unit tests to observe similarity to original table as well as other standard tests.
"""


def fetch_financial_data_quarterly(gvkey, start_date, end_date):
    """
    Fetch financial data for a given ticker and date range from the CCM database in WRDS.
    
    :param gvkey: The gvkey symbol for the company.
    :param start_date: The start date for the data in YYYY-MM-DD format.
    :param end_date: The end date for the data in YYYY-MM-DD format or 'Current'.
    :return: A DataFrame containing the financial data.
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


def fetch_data_for_tickers(ticks):
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
        new_data = fetch_financial_data_quarterly(gvkey, start_date, end_date)
        if isinstance(new_data, tuple):
            empty_tickers.append({row['Ticker']: gvkey})
        else:
            prim_dealers = pd.concat([new_data, prim_dealers], axis=0)
    
    return prim_dealers, empty_tickers



def combine_bd_financials(data_dir=DATA_DIR, UPDATED=False):
    """
    Combine broker & dealer financial data from historical sources and, if UPDATED, from more recent FRED data.

    Parameters:
    - data_dir (Path or str): Directory where the data is stored or should be saved.
    - UPDATED (bool): Whether to include data from 2013 onwards.

    Returns:
    DataFrame: Combined broker & dealer financial assets and liabilities data.
    """
    
    # Load historical data (up to 2012) from local file or fetch if necessary
    bd_financials_historical = load_fred_past(data_dir=data_dir)
    bd_financials_historical.index = pd.to_datetime(bd_financials_historical.index)
    
    if UPDATED:
        # Load recent data
        bd_financials_recent = load_bd_financials()  
        bd_financials_recent.index = pd.to_datetime(bd_financials_recent.index)
        start_date = pd.to_datetime("2012-12-31")
        bd_financials_recent = bd_financials_recent[bd_financials_recent.index > start_date]

        # Append the recent data to the historical data
        bd_financials_combined = pd.concat([bd_financials_historical, bd_financials_recent])
        
    else:
        bd_financials_combined = bd_financials_historical
    
    return bd_financials_combined    


def prep_datasets(datasets, UPDATED=False):
    """
    Function to prepare datasets by dropping duplicates, converting quarter to date, 
    and aggregating data based on specified columns.

    Parameters:
    dataset (DataFrame): DataFrame containing the dataset to be prepared.
    start_date (str): Start date for filtering the data.
    end_date (str): End date for filtering the data.

    Returns:
    prepared_dataset (DataFrame): Prepared DataFrame with specified operations applied.
    """
    # Drop duplicates and convert 'datafqtr' to date format
    datasets = datasets.drop_duplicates()
    datasets['datafqtr'] = datasets['datafqtr'].apply(quarter_to_date)
    
    # Aggregate data based on specified columns
    aggregated_dataset = datasets.groupby('datafqtr').agg({
        'total_assets': 'sum',
        'book_debt': 'sum',
        'book_equity': 'sum',
        'market_equity': 'sum'
    }).reset_index()
    
    bd_financials_combined = combine_bd_financials(UPDATED=UPDATED)
    aggregated_dataset = aggregated_dataset.merge(bd_financials_combined, left_on='datafqtr', right_index=True)

    return aggregated_dataset


def main(UPDATED=False):
    """
    Main function to execute the entire data processing pipeline.
    Returns:
    - formatted_table (pandas.DataFrame): DataFrame containing the formatted table.
    """

    db = wrds.Connection(wrds_username=config.WRDS_USERNAME)
    
    prim_dealers, _ = prim_deal_merge_manual_data_w_linktable(UPDATED=UPDATED)
    datasets, _ = fetch_data_for_tickers(prim_dealers)
    prep_datasets = prep_datasets(datasets, UPDATED=UPDATED)

    Table02Analysis.create_summary_stat_table_for_data(datasets,UPDATED=UPDATED)
    table = create_ratios_for_table(prepped_datasets,UPDATED=UPDATED)
    Table02Analysis.create_figure_for_data(table,UPDATED=UPDATED)
    formatted_table = format_final_table(table, UPDATED=UPDATED)
    convert_and_export_table_to_latex(formatted_table,UPDATED=UPDATED)
    return formatted_table
