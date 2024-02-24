import pandas as pd
import numpy as np

import config
from pathlib import Path
DATA_DIR = Path(config.DATA_DIR)
OUTPUT_DIR = Path(config.OUTPUT_DIR)

import Table_01

df_copy = Table_01.merged_df_final.copy()

# Replace '&' with '\&' in the entire DataFrame
# This loop will go through each column and replace '&' with '\&'
for col in df_copy.columns:
    if df_copy[col].dtype == object:  # Only apply to columns with object (string) type
        df_copy[col] = df_copy[col].str.replace('&', '\\&', regex=False)

latex_table_string = df_copy.to_latex(index=False, escape=False)
print(latex_table_string)

path = OUTPUT_DIR / f'Table_01_to_latex.tex'
with open(path, "w") as text_file:
    text_file.write(latex_table_string)