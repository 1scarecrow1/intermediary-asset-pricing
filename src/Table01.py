import pandas as pd
import numpy as np
import datetime as datetime

import config
from pathlib import Path
DATA_DIR = Path(config.DATA_DIR)
OUTPUT_DIR = Path(config.OUTPUT_DIR)


import load_nyfed

# Obtain the historical list of primary dealers as of Feb 2014.
df_2000s = load_nyfed.load_nyfed(url=load_nyfed.url, data_dir=DATA_DIR, 
                           save_cache=False, sheet_name = '2000s')
df_2000s = df_2000s.drop(index=[0,1])
df_2000s.reset_index(drop=True, inplace=True)

df_2000s.columns = df_2000s.iloc[0]
df_2000s.drop(index=0, inplace=True)
df_2000s.reset_index(drop=True, inplace=True)
df_2014 = df_2000s.iloc[:22,14].to_frame(name='Primary Dealer')


# Obtain the start and end date of primary dealers.
df_dealer_alpha = load_nyfed.load_nyfed(url=load_nyfed.url, data_dir=DATA_DIR, 
                           save_cache=False, sheet_name = 'Dealer Alpha')
df_dealer_alpha.drop(index=0, inplace=True)
df_dealer_alpha.reset_index(drop=True, inplace=True)

df_dealer_alpha.columns = df_dealer_alpha.iloc[0]
df_dealer_alpha.drop(index=0,inplace=True)
df_dealer_alpha = df_dealer_alpha.iloc[:,:3]
df_dealer_alpha.reset_index(drop=True, inplace=True)

df_dealer_alpha['Start Date'] = pd.to_datetime(df_dealer_alpha['Start Date'])
df_dealer_alpha['End Date'] = pd.to_datetime(df_dealer_alpha['End Date'], errors='coerce')
df_dealer_alpha['End Date'] = df_dealer_alpha['End Date'].apply(lambda x: x.date() 
                                                                if pd.notnull(x) else x)
df_dealer_alpha['End Date'] = df_dealer_alpha['End Date'].fillna('Current')


# Match primary dealer in the list with start date
df_2014['Primary_Dealer_Temp'] = df_2014['Primary Dealer'].str.strip().str.replace(r'\.$', '', regex=True).str.replace(' ', '')
df_dealer_alpha['Primary_Dealer_Temp'] = df_dealer_alpha['Primary Dealer'].str.strip().str.replace(r'\.$', '', regex=True).str.replace(' ', '')
df_dealer_alpha = df_dealer_alpha.sort_values(by='Start Date')
df_dealer_alpha = df_dealer_alpha.drop_duplicates(subset='Primary_Dealer_Temp', keep='last')
merged_df = pd.merge(df_2014, df_dealer_alpha[['Primary_Dealer_Temp', 'Start Date']],
                    on='Primary_Dealer_Temp', how='left')
merged_df.drop(columns=['Primary_Dealer_Temp'],inplace=True)
merged_df['Start Date'] = pd.to_datetime(merged_df['Start Date'])
merged_df.sort_values(by='Start Date', inplace=True)
blankIndex=[''] * len(merged_df)
merged_df.index=blankIndex
merged_df





df_level = load_fred.load_fred(data_dir=DATA_DIR).dropna()

df_quarterly = 100 * df_level.pct_change()
# df_quarterly.plot()

# Select only the values that occur in July
_df = df_level[df_level.index.month == 7]
df_annual = 100 * _df.pct_change()
# df_annual.plot()

df_quarterly.describe()
df_annual.describe()


columns_for_summary_stats = [
    'CPIAUCNS',
    'GDPC1',
    ]

# This maps the column names to their LaTeX descriptions
column_names_map = {
    'CPIAUCNS':'Inflation',
    'GDPC1':'Real GDP',
}

escape_coverter = {
    '25%':'25\\%',
    '50%':'50\\%',
    '75%':'75\\%'
}

df_annual = df_annual[columns_for_summary_stats]

## Suppress scientific notation and limit to 3 decimal places
# Sets display, but doesn't affect formatting to LaTeX
pd.set_option('display.float_format', lambda x: '%.2f' % x)
# Sets format for printing to LaTeX
float_format_func = lambda x: '{:.2f}'.format(x)

# Pooled summary stats
describe_all = (
    df_annual[columns_for_summary_stats].
    describe().T.
    rename(index=column_names_map, columns=escape_coverter)
)
describe_all['count'] = describe_all['count'].astype(int)
describe_all.columns.name = 'Full Sample: 1947 - 2023'
latex_table_string_all = describe_all.to_latex(escape=False, float_format=float_format_func)

describe1 = (
    df_annual[columns_for_summary_stats].
    describe().T.
    rename(index=column_names_map, columns=escape_coverter)
)
describe1['count'] = describe1['count'].astype(int)
describe1.columns.name = 'Subsample: 1947 - 1990'
latex_table_string1 = describe1.to_latex(escape=False, float_format=float_format_func)

describe2 = (
    df_annual.loc["1990":,columns_for_summary_stats].
    describe().T.
    rename(index=column_names_map, columns=escape_coverter)
)
describe2.columns.name = 'Subsample: 1990-2023'
latex_table_string2 = describe2.to_latex(escape=False, float_format=float_format_func)

latex_table_string_split = [
    *latex_table_string_all.split('\n')[0:-3], # Skip the \end{tabular} and \bottomrule lines
    '\\midrule',
    *latex_table_string1.split('\n')[2:-3], # Skip the \begin and \end lines
    '\\midrule',
    *latex_table_string2.split('\n')[2:] # Skip the \begin{tabular} and \toprule lines
]
latex_table_string = '\n'.join(latex_table_string_split)
# print(latex_table_string)
path = OUTPUT_DIR / f'example_table.tex'
with open(path, "w") as text_file:
    text_file.write(latex_table_string)
