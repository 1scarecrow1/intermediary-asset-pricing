import pandas as pd
import numpy as np

import config
from pathlib import Path
DATA_DIR = Path(config.DATA_DIR)
OUTPUT_DIR = Path(config.OUTPUT_DIR)

"""
Importing Table_01.py, this script uses 'Table_A1_to_latex.py' to format 'df_dealer_alpha' into LaTeX, 
saving it as 'Table_A1_to_latex.tex' in the output directory. It replaces '&' with '\&' in company names 
to avoid LaTeX errors and divides the primary dealer list at the midpoint for visual clarity, aligning 
with Table A.1's format.
"""

import Table_A1

df_copy = Table_A1.df_dealer_alpha.copy()
for col in df_copy.columns:
    if df_copy[col].dtype == object:  # Only apply to columns with object (string) type
        df_copy[col] = df_copy[col].str.replace('&', '\\&', regex=False)

midpoint = len(df_copy) // 2
df_first_half = df_copy.iloc[:midpoint].reset_index(drop=True)
df_second_half = df_copy.iloc[midpoint:].reset_index(drop=True)

# Adding a separator column for visual division
df_combined = pd.concat([df_first_half, df_second_half], axis=1)
df_combined.insert(3,'',np.nan)
df_combined.fillna('',inplace=True)

latex_table_string = df_combined.to_latex(index=False, escape=False)
print(latex_table_string)

path = OUTPUT_DIR / f'Table_A1_to_latex.tex'
with open(path, "w") as text_file:
    text_file.write(latex_table_string)