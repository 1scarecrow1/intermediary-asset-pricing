import pandas as pd
import wrds
import config
from datetime import datetime
import unittest
import matplotlib.pyplot as plt
import numpy as np

"""
Is referenced by Table02Prep. Creates tables to understand the data and figures to understand the different ratios.
"""
def create_summary_stat_table_for_data(datasets, UPDATED=False):
    summary_df = pd.DataFrame()
    for key in datasets.keys():
        dataset = datasets[key].drop(columns=['datadate'])
        info = dataset.describe()
        info = info.drop(['25%', '50%', '75%'])
        numeric_cols = info.select_dtypes(include=['float64', 'int']).columns
        info[numeric_cols] = info[numeric_cols].round(2)
        info.reset_index(inplace=True)
        info['Key'] = key
        info.set_index(['Key', 'index'], inplace=True)
        summary_df = pd.concat([summary_df, info], axis=0)
    summary_df = summary_df.round(2) # update caption
    summary_df.columns =  ['total assets', 'book debt', 'book equity', 'market equity']
    caption = 'There are significantly less entries for book equity than the other measures as shown in the count rows. There are also some negatives for book equity which is not present for other categories. '
    latex_table = summary_df.to_latex(index=True, multirow=True, multicolumn=True, escape=False, float_format="%.2f", caption=caption, label='tab:Table 2.1')
    latex_table = latex_table.replace(r'\multirow[t]{5}{*}', '')
    if UPDATED:
        with open('../output/updated_table02_sstable.tex', 'w') as f:
            f.write(latex_table)
    else:
        with open('../output/table02_sstable.tex', 'w') as f:
            f.write(latex_table)

def create_figure_for_data(ratios_dict, UPDATED=False):
    concatenated_df = pd.concat([s.rename(f"{key}_{s.name}") for key, s in ratios_dict.items()], axis=1)

    concatenated_df.sort_index(inplace=True)

    concatenated_df = concatenated_df.apply(pd.to_numeric, errors='coerce')

    concatenated_df.ffill(inplace=True)
    concatenated_df.bfill(inplace=True)

    asset_columns = [col for col in concatenated_df.columns if 'total_assets' in col]
    debt_columns = [col for col in concatenated_df.columns if 'book_debt' in col]
    equity_columns = [col for col in concatenated_df.columns if 'book_equity' in col]
    market_columns = [col for col in concatenated_df.columns if 'market_equity' in col]

    asset_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    debt_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    equity_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    market_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, columns, colors, category in zip(axes.flatten(),
                                             [asset_columns, debt_columns, equity_columns, market_columns],
                                             [asset_colors, debt_colors, equity_colors, market_colors],
                                             ['total_assets', 'book_debt', 'book_equity', 'market_equity']):
        columns.sort()
        unique_keys = [col.split('_')[-1] for col in columns]  # Extract keys from sorted column names

        for col, color, key in zip(columns, colors, unique_keys):
            ax.plot(concatenated_df.index, concatenated_df[col], label=key, color=color)  # Use unique keys as labels
        ax.set_title(f"{category.capitalize()}")  # Set subplot title with category name
        ax.legend(loc='upper left')  # Set legend location
        ax.grid(True)
        ax.set_xlabel('Date')  # Set x-axis label for each subplot
        ax.set_ylabel('Value')  # Set y-axis label for each subplot
        # Add caption
    time = datetime.now()
    caption = str(
        time) + ': From the plots above we can observe the trends of the ratios for each comparison group over time. Keep in mind that we have filled in missing values to make the lines display properly.'
    fig.text(0.5, -0.1, caption, ha='center', fontsize=8)
    if UPDATED:
        plt.savefig('../output/updated_table02_figure.png')
    else:
        plt.savefig('../output/table02_figure.png')