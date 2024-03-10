import pandas as pd
import wrds
import config
from datetime import datetime
import unittest
import matplotlib.pyplot as plt
import numpy as np
import load_CRSP_stock

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.seasonal import seasonal_decompose

from Table03Prep import *
from Table02Prep import prim_deal_merge_manual_data_w_linktable, convert_and_export_table_to_latex

"""
Reads in manual dataset for primary dealers and holding companies and matches it with linkhist entry for company. 
Compiles and prepares this data to produce Table 03 from intermediary asset pricing paper in LaTeX format.
Also creates a summary statistics table and figure in LaTeX format.
Performs unit tests to observe similarity to original table as well as other standard tests.
"""


def fetch_financial_data_quarterly(gvkey, start_date, end_date, db):
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


def prep_dataset(dataset, UPDATED=False):
    """
    Function to prepare datasets by dropping duplicates, converting quarter to date, 
    and aggregating data based on specified columns.
    """
    # Drop duplicates and convert 'datafqtr' to date format
    dataset = dataset.drop_duplicates()
    dataset['datafqtr'] = dataset['datafqtr'].apply(quarter_to_date)
    
    # Aggregate data based on specified columns
    aggregated_dataset = dataset.groupby('datafqtr').agg({
        'total_assets': 'sum',
        'book_debt': 'sum',
        'book_equity': 'sum',
        'market_equity': 'sum'
    }).reset_index()
    
    bd_financials_combined = combine_bd_financials(UPDATED=UPDATED)
    aggregated_dataset = aggregated_dataset.merge(bd_financials_combined, left_on='datafqtr', right_index=True)

    return aggregated_dataset

def calculate_ratios(data):
    """
    Aggregates market cap ratio, book cap ratio, and AEM leverage for a given DataFrame.
    """
    data['market_cap_ratio'] = data['market_equity'] / (data['book_debt'] + data['market_equity'])
    data['book_cap_ratio'] = data['book_equity'] / (data['book_debt'] + data['book_equity'])
    data['aem_leverage'] = data['bd_fin_assets'] / (data['bd_fin_assets'] - data['bd_liabilities'])
    data['aem_leverage_ratio'] = 1 / data['aem_leverage']
    
    return data

def aggregate_ratios(data):
    data = calculate_ratios(data)
    data = data[['datafqtr', 'market_cap_ratio', 'book_cap_ratio', 'aem_leverage_ratio']]
    data.rename(columns={'datafqtr': 'date'}, inplace=True)
    data = data.set_index('date')
    return data

def convert_ratios_to_factors(data):
    factors_df = pd.DataFrame(index=data.index)

    # AR(1) for market capital ratio
    cleaned_data = data['market_cap_ratio'].dropna()
    model = AutoReg(cleaned_data, lags=1)
    model_fitted = model.fit()
    factors_df['innovations_mkt_cap'] = model_fitted.resid
    factors_df['market_capital_factor'] = factors_df['innovations_mkt_cap'] / data['market_cap_ratio'].shift(1)
    factors_df.drop(columns=['innovations_mkt_cap'], inplace=True)

    # AR(1) for book capital ratio
    cleaned_data = data['book_cap_ratio'].dropna()
    model = AutoReg(cleaned_data, lags=1)
    model_fitted = model.fit()
    factors_df['innovations_book_cap'] = model_fitted.resid
    factors_df['book_capital_factor'] = factors_df['innovations_book_cap'] / data['book_cap_ratio'].shift(1)
    factors_df.drop(columns=['innovations_book_cap'], inplace=True)

    # Calculate the AEM leverage factor
    factors_df['leverage_growth'] = data['aem_leverage_ratio'].pct_change().fillna(0)
    decomposition = seasonal_decompose(factors_df['leverage_growth'], model='additive', period=4)
    factors_df['seasonal'] = decomposition.seasonal
    factors_df['aem_leverage_factor'] = factors_df['leverage_growth'] - factors_df['seasonal']

    # Return only the factor columns
    return factors_df[['market_capital_factor', 'book_capital_factor', 'aem_leverage_factor']]


def calculate_ep(shiller_cape):
    """
    Processes the Shiller CAPE DataFrame and calculates the E/P ratio.
    """
    df = shiller_cape.copy()
    df.columns = ['date', 'cape']
    df['date'] = df['date'].astype(str)
    df['date'] = pd.to_datetime(df['date'], format='%Y.%m') + pd.offsets.MonthEnd(0)
    df = df.set_index('date')
    df['e/p'] = 1 / df['cape']
    
    return df

def macro_variables():
    macro_data = load_fred_macro_data()
    macro_data = macro_data.rename(columns={'UNRATE': 'unemp_rate',
                                    'NFCI': 'nfci',
                                    'GDPC1': 'real_gdp',
                                    'A191RL1Q225SBEA': 'real_gdp_growth',
                                    })
    macro_data.index = pd.to_datetime(macro_data.index)
    macro_data.rename(columns={'DATE': 'date'}, inplace=True)
    macro_quarterly = macro_data.resample('Q').mean()

    shiller_cape = load_shiller_pe()
    shiller_ep = calculate_ep(shiller_cape)
    shiller_quarterly = shiller_ep.resample('Q').mean()

    ff_facs = fetch_ff_factors(start_date='19700101', end_date='20240229')
    ff_facs_quarterly = ff_facs.to_timestamp(freq='M').resample('Q').last()

    value_wtd_indx = pull_CRSP_Value_Weighted_Index()
    value_wtd_indx['date'] = pd.to_datetime(value_wtd_indx['date'])
    annual_vol_quarterly = value_wtd_indx.set_index('date')['vwretd'].pct_change().groupby(pd.Grouper(freq='Q')).std().rename('mkt_vol')

    macro_merged = shiller_quarterly.merge(macro_quarterly, left_index=True, right_index=True, how='left')
    macro_merged = macro_merged.merge(ff_facs_quarterly[['mkt_ret']],left_index=True, right_index=True)
    macro_merged = macro_merged.merge(annual_vol_quarterly, left_index=True, right_index=True)

    return macro_merged


def create_panelA(ratios, macro):
    """
    Creates Panel A of Table 3.    
    """
    ratios_renamed = ratios.rename(columns={
        'market_cap_ratio': 'Market capital',
        'book_cap_ratio': 'Book capital',
        'aem_leverage_ratio': 'AEM leverage'
    })
    macro = macro[['e/p', 'unemp_rate', 'nfci', 'real_gdp', 'mkt_ret', 'mkt_vol']]
    macro_renamed = macro.rename(columns={
        'e/p': 'E/P',
        'unemp_rate': 'Unemployment',
        'nfci': 'Financial conditions',
        'real_gdp': 'GDP',
        'mkt_ret': 'Market excess return',
        'mkt_vol': 'Market volatility'
    })

    panelA = ratios_renamed.merge(macro_renamed, left_index=True, right_index=True)
    ordered_columns= ['Market capital', 'Book capital', 'AEM leverage',
                        'E/P', 'Unemployment', 'Financial conditions', 'GDP', 'Market excess return', 'Market volatility']
    panelA = panelA[ordered_columns]
    
    return panelA


def create_panelB(factors, macro):
    """
    Creates Panel B of Table 3.
    """
    factors_renamed = factors.rename(columns={
        'market_capital_factor': 'Market capital factor',
        'book_capital_factor': 'Book capital factor',
        'aem_leverage_factor': 'AEM leverage factor'})
    
    macro_selected = macro[['e/p', 'unemp_rate', 'nfci', 'mkt_ret', 'mkt_vol']]
    macro_growth = np.log(macro_selected / macro_selected.shift(1))
    
    macro_growth['real_gdp_growth'] = macro['real_gdp_growth']
    macro_growth_renamed = macro_growth.rename(columns={
        'e/p': 'E/P growth',
        'unemp_rate': 'Unemployment growth',
        'nfci': 'Financial conditions growth',
        'real_gdp_growth': 'GDP growth',
        'mkt_ret': 'Market excess return',
        'mkt_vol': 'Market volatility growth'
    })

    panelB = factors_renamed.merge(macro_growth_renamed, left_index=True, right_index=True)
    ordered_columns = ['Market capital factor', 'Book capital factor', 'AEM leverage factor',
                       'E/P growth', 'Unemployment growth', 'Financial conditions growth', 'GDP growth', 'Market excess return', 'Market volatility growth']
    panelB = panelB[ordered_columns]
    return panelB                      


def format_correlation_matrix(corr_matrix):
    corr_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
    return corr_matrix

def calculate_correlation_panelA(levels2):
    correlation_panelA = format_correlation_matrix(levels2.iloc[:, :3].corr())
    main_cols = levels2[['Market capital', 'Book capital', 'AEM leverage']]
    other_cols = levels2[['E/P', 'Unemployment', 'GDP', 'Financial conditions', 'Market volatility']]
    
    correlation_results_panelA = pd.DataFrame(index=main_cols.columns)
    for column in other_cols.columns:
        correlation_results_panelA[column] = main_cols.corrwith(other_cols[column])
    
    return pd.concat([correlation_panelA, correlation_results_panelA.T], axis=0)

def calculate_correlation_panelB(panelA):
    correlation_panelB = format_correlation_matrix(panelA.iloc[:, :3].corr())
    main_cols = panelA[['Market capital factor', 'Book capital factor', 'AEM leverage factor']]
    other_cols = panelA[['Market excess return', 'E/P growth', 'Unemployment growth', 'GDP growth', 'Financial conditions growth', 'Market volatility growth']]
    
    correlation_results_panelB = pd.DataFrame(index=main_cols.columns)
    for column in other_cols.columns:
        correlation_results_panelB[column] = main_cols.corrwith(other_cols[column])
    
    return pd.concat([correlation_panelB, correlation_results_panelB.T], axis=0)

def format_final_table(corrA, corrB):
    panelB_renamed = corrB.copy()
    panelB_renamed.columns = corrA.columns

    panelB_column_names = pd.DataFrame([corrB.columns], columns=corrA.columns)
    panelB_column_names.reset_index(drop=True, inplace=True)
    panelB_combined = pd.concat([panelB_column_names, panelB_renamed])

    panelA_title = pd.DataFrame({'Panel A: Correlations of levels': [np.nan, np.nan, np.nan]}, index=corrA.columns)
    panelB_title = pd.DataFrame({'Panel B: Correlations of factors': [np.nan, np.nan, np.nan]}, index=corrA.columns)
    
    full_table = pd.concat([panelA_title.T, corrA, panelB_title.T, panelB_combined])

    return full_table

def convert_and_export_tables_to_latex(corrA, corrB, UPDATED=False):
    corrA = corrA.fillna('')
    corrB = corrB.fillna('')
    
    latexA = corrA.to_latex(index=True, column_format='lcccccccccccc', float_format="%.2f")
    latexB = corrB.to_latex(index=True, column_format='lcccccccccccc', float_format="%.2f")

    start_tabular_A = latexA.find("\\begin{tabular}")
    end_tabular_A = latexA.find("\\end{tabular}") + len("\\end{tabular}")
    tabular_content_A = latexA[start_tabular_A:end_tabular_A]

    start_tabular_B = latexB.find("\\begin{tabular}")
    end_tabular_B = latexB.find("\\end{tabular}") + len("\\end{tabular}")
    tabular_content_B = latexB[start_tabular_B:end_tabular_B]

    full_latex = rf"""
    \documentclass{{article}}
    \usepackage{{booktabs}} % For better-looking tables
    \usepackage{{graphicx}} % Required for inserting images
    \usepackage{{adjustbox}} % Useful for adjusting table sizes

    \begin{{document}}
    \begin{{table}}[htbp]
      \centering
      \begin{{adjustbox}}{{max width=\textwidth}}
      \small
      Panel A: Correlation of Levels \\
      {tabular_content_A}
      \vspace{{2em}} 
      Panel B: Correlation of Factors \\
      {tabular_content_B}
      \end{{adjustbox}}
    \end{{table}}
    \end{{document}}
    """

    # Write the full LaTeX code to a .tex file
    if UPDATED:
        with open('../output/updated_table03.tex', 'w') as f:
            f.write(full_latex)
    else:
        with open('../output/table03.tex', 'w') as f:
            f.write(full_latex)


# For plotting Figure 1 
def standardize_ratios_and_factors(data):
    """
    Automatically standardizes columns that end with "ratio" in a given DataFrame
    """
    columns_to_standardize = [col for col in data.columns if col.endswith('ratio') or col.endswith('factor')]
    for col in columns_to_standardize:
        standardized_col_name = f'{col}_std'
        data[standardized_col_name] = (data[col] - data[col].mean()) / data[col].std()

    return data


def main(UPDATED=False):
    """
    Main function to execute the entire data processing pipeline.
    Returns:
    - formatted_table (pandas.DataFrame): DataFrame containing the formatted table.
    """



    prim_dealers, _ = prim_deal_merge_manual_data_w_linktable(UPDATED=UPDATED)
    db = wrds.Connection(wrds_username=config.WRDS_USERNAME)
    dataset, _ = fetch_data_for_tickers(prim_dealers, db)
    prep_datast = prep_dataset(dataset, UPDATED=UPDATED)
    ratio_dataset = aggregate_ratios(prep_datast)
    factors_dataset = convert_ratios_to_factors(ratio_dataset)
    macro_variabls = macro_variables()

    panelA = create_panelA(ratio_dataset, macro_variabls)
    panelB = create_panelB(factors_dataset, macro_variabls)
    correlation_panelA = calculate_correlation_panelA(panelA)
    correlation_panelB = calculate_correlation_panelB(panelB)
    formatted_table = format_final_table(correlation_panelA, correlation_panelB)
    convert_and_export_tables_to_latex(correlation_panelA, correlation_panelB, UPDATED=UPDATED)
    return formatted_table.style.format(na_rep='')
