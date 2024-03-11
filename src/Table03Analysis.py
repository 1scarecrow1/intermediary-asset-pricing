import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

"""
Is referenced by Table03. Creates tables to understand the data and figures to understand the different ratios.
"""

def create_summary_stat_table_for_data(dataset, UPDATED=False):
    """
    Creates a summary statistics table for the dataset.
    """
    summary_df = pd.DataFrame()
    info = dataset.describe()
    info = info.drop(['25%', '50%', '75%'])
    summary_df = pd.concat([summary_df, info], axis=0)
    
    caption = "Summary statistics of capital factors and macro variables"

    latex_table = summary_df.to_latex(index=True, multirow=True, multicolumn=True, escape=False, float_format="%.2f", caption=caption, label='tab:Table 2.1')
    latex_table = latex_table.replace(r'\multirow[t]{5}{*}', '')

    if UPDATED:
        with open('../output/updated_table03_sstable.tex', 'w') as f:
            f.write(latex_table)
    else:
        with open('../output/table03_sstable.tex', 'w') as f:
            f.write(latex_table)

    

# For plotting Figure 1 and 2 
def standardize_ratios_and_factors(data):
    """
    Automatically standardizes columns that end with "ratio" or "factor" in a given DataFrame
    """
    columns_to_standardize = [col for col in data.columns if col.endswith('ratio') or col.endswith('factor')]
    for col in columns_to_standardize:
        standardized_col_name = f'{col}_std'
        data[standardized_col_name] = (data[col] - data[col].mean()) / data[col].std()

    return data


def plot_figure01(ratios, factors, UPDATED=False):
    """
    Plots the standardized market cap ratio and capital risk factor over time.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = pd.concat([ratios[['market_cap_ratio']], factors[['market_capital_factor']]], axis=1)
    data = standardize_ratios_and_factors(data)
    
    ax.plot(data.index, data['market_cap_ratio_std'], label='Market Cap Ratio')
    ax.plot(data.index, data['market_capital_factor_std'], color='orange', linestyle='--', label='Capital Risk Factor')
    
    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.set_xlabel('Date')
    ax.set_ylabel('Standardized Value')
    ax.set_ylim(-4, 4)
    ax.set_yticks([-4, -2, 0, 2, 4])
    ax.set_title('Intermediary Capital Ratio and Risk Factor of Primary Dealers')
    ax.legend(loc='best')
    # plt.show()

    if UPDATED:
        plt.savefig('../output/updated_table03_figure01.png')
    else:
        plt.savefig('../output/table03_figure01.png')


def plot_figure02(ratios, UPDATED=False):
    """
    Plots the levels of market capital ratio, book capital ratio, and AEM leverage ratio over time.
    """
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ratios = standardize_ratios_and_factors(ratios)
    ax.plot(ratios.index, ratios['market_cap_ratio']*100, label='Market Capital Ratio')
    ax.plot(ratios.index, ratios['book_cap_ratio']*100, label='Book Capital Ratio', color='green', linestyle='dotted')
    ax.plot(ratios.index, 1 / ratios['aem_leverage_ratio'], label='AEM Leverage Ratio', color='orange', linestyle='--')

    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.set_xlabel('Date')
    ax.set_yscale('log')
    ax.set_yticks([5, 10, 50, 100])
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter()) 

    ax.set_title('AEM Leverage and Intermediary Capital Ratio: Level')
    ax.legend(loc='best')
    # plt.show()

    if UPDATED:
        plt.savefig('../output/updated_table03_figure.png')
    else:
        plt.savefig('../output/table03_figure.png')