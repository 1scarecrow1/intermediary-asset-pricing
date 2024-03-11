import pandas as pd
import wrds
import config
from datetime import datetime
import unittest
import matplotlib.pyplot as plt
import numpy as np

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.seasonal import seasonal_decompose

import Table03Load
from Table03Load import quarter_to_date, date_to_quarter
import Table03Analysis
import Table02Prep

"""
Reads in manual dataset for primary dealers and holding companies and matches it with linkhist entry for company. 
Calculates the market capital ratio, book capital ratio, and AEM leverage ratio and the correlation of these ratios with macroeconomic variables.
Compiles and prepares this data to produce Table 03 from intermediary asset pricing paper in LaTeX format.
Also creates a summary statistics table and figure in LaTeX format.
"""


def combine_bd_financials(UPDATED=False):
    """
    Combine broker & dealer financial data from historical sources and, if UPDATED, from more recent FRED data.
    """
    
    # Load historical data (up to 2012) from local file or fetch if necessary
    bd_financials_historical = Table03Load.load_fred_past()
    bd_financials_historical.index = pd.to_datetime(bd_financials_historical.index)
    
    if UPDATED:
        # Load recent data
        bd_financials_recent = Table03Load.load_bd_financials()  
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
    Calculates market cap ratio, book cap ratio, and AEM leverage ratio.
    """
    data['market_cap_ratio'] = data['market_equity'] / (data['book_debt'] + data['market_equity'])
    data['book_cap_ratio'] = data['book_equity'] / (data['book_debt'] + data['book_equity'])
    data['aem_leverage'] = data['bd_fin_assets'] / (data['bd_fin_assets'] - data['bd_liabilities'])
    data['aem_leverage_ratio'] = 1 / data['aem_leverage']
    
    return data


def aggregate_ratios(data):
    """
    Aggregates market cap ratio, book cap ratio, and AEM leverage ratio.
    """
    data = calculate_ratios(data)
    data = data[['datafqtr', 'market_cap_ratio', 'book_cap_ratio', 'aem_leverage_ratio']]
    data.rename(columns={'datafqtr': 'date'}, inplace=True)
    data = data.set_index('date')
    return data


def convert_ratios_to_factors(data):
    """
    Converts ratios to analytical factors.
    """

    factors_df = pd.DataFrame(index=data.index)

    # AR(1) with constant for market capital ratio
    cleaned_data = data['market_cap_ratio'].dropna()
    model = AutoReg(cleaned_data, lags=1, trend='c')
    model_fitted = model.fit()
    factors_df['innovations_mkt_cap'] = model_fitted.resid
    factors_df['market_capital_factor'] = factors_df['innovations_mkt_cap'] / data['market_cap_ratio'].shift(1)
    factors_df.drop(columns=['innovations_mkt_cap'], inplace=True)

    # AR(1) with constant for market capital ratio
    cleaned_data = data['book_cap_ratio'].dropna()
    model = AutoReg(cleaned_data, lags=1, trend='c')
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


def macro_variables(db):
    """
    Creates a table of macroeconomic variables to be used in the analysis.
    Note: The function starts gathering data from one year earlier than 1970 to allow for factor calculation where differences are used.
    """
    # Load FRED macroeconomic data and resample quarterly
    macro_data = Table03Load.load_fred_macro_data()
    macro_data = macro_data.rename(columns={'UNRATE': 'unemp_rate',
                                    'NFCI': 'nfci',
                                    'GDPC1': 'real_gdp',
                                    'A191RL1Q225SBEA': 'real_gdp_growth',
                                    })
    macro_data.index = pd.to_datetime(macro_data.index)
    macro_data.rename(columns={'DATE': 'date'}, inplace=True)
    macro_quarterly = macro_data.resample('Q').mean()

    # Load Shiller PE and calculate earnings-to-price ratio
    shiller_cape = Table03Load.load_shiller_pe()
    shiller_ep = calculate_ep(shiller_cape)
    shiller_quarterly = shiller_ep.resample('Q').mean()

    # Fetch Fama-French factors and resample quarterly
    ff_facs = Table03Load.fetch_ff_factors(start_date='19690101', end_date='20240229')
    ff_facs_quarterly = ff_facs.to_timestamp(freq='M').resample('Q').last()

    # Pull CRSP Value Weighted Index and calculate quarterly market volatility
    value_wtd_indx = Table03Load.pull_CRSP_Value_Weighted_Index(db)
    value_wtd_indx['date'] = pd.to_datetime(value_wtd_indx['date'])
    annual_vol_quarterly = value_wtd_indx.set_index('date')['vwretd'].pct_change().groupby(pd.Grouper(freq='Q')).std().rename('mkt_vol')

    # Merge all macroeconomic data
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
    panelA = panelA.loc['1970-01-01':]
    
    return panelA


def create_panelB(factors, macro):
    """
    Creates Panel B of Table 3.
    """
    factors_renamed = factors.rename(columns={
        'market_capital_factor': 'Market capital factor',
        'book_capital_factor': 'Book capital factor',
        'aem_leverage_factor': 'AEM leverage factor'})
    
    # Calculate quarterly growth rates of macroeconomic variables
    macro_growth = np.log(macro / macro.shift(1))
    macro_growth = macro_growth.fillna(0)
    macro_growth = macro_growth.loc['1970-01-01':]

    macro_growth['mkt_ret'] = macro['mkt_ret']
    macro_growth_renamed = macro_growth.rename(columns={
        'e/p': 'E/P growth',
        'unemp_rate': 'Unemployment growth',
        'nfci': 'Financial conditions growth',
        'real_gdp': 'GDP growth',
        'mkt_ret': 'Market excess return',
        'mkt_vol': 'Market volatility growth'
    })

    panelB = factors_renamed.merge(macro_growth_renamed, left_index=True, right_index=True)
    ordered_columns = ['Market capital factor', 'Book capital factor', 'AEM leverage factor',
                       'E/P growth', 'Unemployment growth', 'Financial conditions growth', 'GDP growth', 'Market excess return', 'Market volatility growth']
    panelB = panelB[ordered_columns]
    panelB = panelB.loc['1970-01-01':]  

    return panelB                      


def format_correlation_matrix(corr_matrix):
    """
    Formats the correlation matrix by masking the lower triangle.
    """
    corr_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=0).astype(np.bool_))
    return corr_matrix


def calculate_correlation_panelA(panelA):
    """
    Calculates the correlation for Panel A (capital ratio levels).
    """
    correlation_panelA = format_correlation_matrix(panelA.iloc[:, :3].corr())
    main_cols = panelA[['Market capital', 'Book capital', 'AEM leverage']]
    other_cols = panelA[['E/P', 'Unemployment', 'GDP', 'Financial conditions', 'Market volatility']]
    
    correlation_results_panelA = pd.DataFrame(index=main_cols.columns)
    for column in other_cols.columns:
        correlation_results_panelA[column] = main_cols.corrwith(other_cols[column])
    
    return pd.concat([correlation_panelA, correlation_results_panelA.T], axis=0)


def calculate_correlation_panelB(panelB):
    """
    Calculates the correlation for Panel B (capital ratio factors).
    """
    correlation_panelB = format_correlation_matrix(panelB.iloc[:, :3].corr())
    main_cols = panelB[['Market capital factor', 'Book capital factor', 'AEM leverage factor']]
    other_cols = panelB[['Market excess return', 'E/P growth', 'Unemployment growth', 'GDP growth', 'Financial conditions growth', 'Market volatility growth']]
    
    correlation_results_panelB = pd.DataFrame(index=main_cols.columns)
    for column in other_cols.columns:
        correlation_results_panelB[column] = main_cols.corrwith(other_cols[column])
    
    return pd.concat([correlation_panelB, correlation_results_panelB.T], axis=0)


def format_final_table(corrA, corrB):
    """
    Format the final correlation table.
    """
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
    """
    Convert correlation tables to LaTeX format and export to a .tex file.
    """
    # Fill NaN values with empty strings for both tables    
    corrA = corrA.round(2).fillna('')
    corrB = corrB.round(2).fillna('')
    
    # Define the caption based on whether the table is updated or not
    if UPDATED:
        caption = "Updated"
    else:
        caption = "Original"

    # Convert the correlation tables to LaTeX format without using to_latex() directly to control the structure
    # Define the column format and titles manually
    column_format = 'l' + 'c' * (len(corrA.columns))
    header_row = " & " + " & ".join(corrA.columns) + " \\\\"
    
    # Generate content rows for Panel A and B 
    panelA_rows = "\n".join([f"{index} & " + " & ".join(corrA.loc[index].astype(str)) + " \\\\" for index in corrA.index])
    panelB_rows = "\n".join([f"{index} & " + " & ".join(corrB.loc[index].astype(str)) + " \\\\" for index in corrB.index])
    
    full_latex = rf"""
    \begin{{table}}[htbp]
    \centering
    \caption{{\label{{tab:correlation}}{caption}}}
    \begin{{adjustbox}}{{max width=\textwidth}}
    \small
    \begin{{tabular}}{{{column_format}}}
        \toprule
        Panel A: Correlation of Levels \\
        \midrule
        {header_row}
        \midrule
        {panelA_rows}
        \midrule
        Panel B: Correlation of Factors \\
        \midrule
        {header_row}
        \midrule
        {panelB_rows}
        \bottomrule
    \end{{tabular}}
    \end{{adjustbox}}
    \end{{table}}
    """

    # Write the full LaTeX code to a .tex file
    if UPDATED:
        with open('../output/updated_table03.tex', 'w') as f:
            f.write(full_latex)
    else:
        with open('../output/table03.tex', 'w') as f:
            f.write(full_latex)


def main(UPDATED=False):
    """
    Main function to execute the entire data processing pipeline.
    """

    db = wrds.Connection(wrds_username=config.WRDS_USERNAME)
    
    prim_dealers, _ = Table02Prep.prim_deal_merge_manual_data_w_linktable(UPDATED=UPDATED)
    dataset, _ = Table03Load.fetch_data_for_tickers(prim_dealers, db)
    prep_datast = prep_dataset(dataset, UPDATED=UPDATED)
    ratio_dataset = aggregate_ratios(prep_datast)
    factors_dataset = convert_ratios_to_factors(ratio_dataset)
    macro_dataset = macro_variables(db)
    panelA = create_panelA(ratio_dataset, macro_dataset)
    panelB = create_panelB(factors_dataset, macro_dataset)

    Table03Analysis.create_summary_stat_table_for_data(panelB, UPDATED=UPDATED)
    Table03Analysis.plot_figure02(ratio_dataset, UPDATED=UPDATED)

    correlation_panelA = calculate_correlation_panelA(panelA)
    correlation_panelB = calculate_correlation_panelB(panelB)
    formatted_table = format_final_table(correlation_panelA, correlation_panelB)
    convert_and_export_tables_to_latex(correlation_panelA, correlation_panelB, UPDATED=UPDATED)
    return print(formatted_table.style.format(na_rep=''))
    

if __name__ == "__main__":
    main(UPDATED=False)
    print("Table 03 has been created and exported to LaTeX format.")