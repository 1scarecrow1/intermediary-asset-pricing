import pandas as pd
import numpy as np
import datetime as datetime

import config
from pathlib import Path
DATA_DIR = Path(config.DATA_DIR)
OUTPUT_DIR = Path(config.OUTPUT_DIR)

"""
After loading the NY Fed primary dealer list Excel file, the process begins by obtaining the historical list of 
primary dealers as of February 2014, including their start and end dates. The next step is to match each primary 
dealer with their start date for accuracy. Subsequently, add these matched primary dealers along with their 
holding companies into the list to replicate Table 01. 
"""

import load_nyfed

# Obtain the historical list of primary dealers as of Feb 2014.
df_2000s = load_nyfed.load_nyfed_primary_dealers_list(url=load_nyfed.url, data_dir=DATA_DIR, 
                           save_cache=True, sheet_name = '2000s')
df_2000s = df_2000s.drop(index=[0,1])
df_2000s.reset_index(drop=True, inplace=True)

df_2000s.columns = df_2000s.iloc[0]
df_2000s.drop(index=0, inplace=True)
df_2000s.reset_index(drop=True, inplace=True)
df_2014 = df_2000s.iloc[:22,14].to_frame(name='Primary Dealer')


# Obtain the start and end date of primary dealers.
df_dealer_alpha = load_nyfed.load_nyfed_primary_dealers_list(url=load_nyfed.url, data_dir=DATA_DIR, 
                           save_cache=True, sheet_name = 'Dealer Alpha')
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
df_2014_temp = df_2014.copy()
df_dealer_alpha_temp = df_dealer_alpha.copy()
df_2014_temp['Primary_Dealer_Temp'] = df_2014_temp['Primary Dealer'].str.strip().str.replace(r'\.$', '', regex=True).str.replace(' ', '')
df_dealer_alpha_temp['Primary_Dealer_Temp'] = df_dealer_alpha_temp['Primary Dealer'].str.strip().str.replace(r'\.$', '', regex=True).str.replace(' ', '')
df_dealer_alpha_temp = df_dealer_alpha_temp.sort_values(by='Start Date')
df_dealer_alpha_temp = df_dealer_alpha_temp.drop_duplicates(subset='Primary_Dealer_Temp', keep='last')
merged_df = pd.merge(df_2014_temp, df_dealer_alpha_temp[['Primary_Dealer_Temp', 'Start Date']],
                    on='Primary_Dealer_Temp', how='left')
merged_df.drop(columns=['Primary_Dealer_Temp'],inplace=True)
merged_df['Start Date'] = pd.to_datetime(merged_df['Start Date'])
merged_df.sort_values(by='Start Date', inplace=True)
merged_df.reset_index(drop=True, inplace=True)


# Add matching primary dealer in the list with holding company
ticks = pd.read_csv('../data/manual/ticks.csv', sep="|")
ticks = ticks.iloc[:,:2]
ticks_temp = ticks.copy()
merged_df_temp = merged_df.copy()
ticks_temp['Primary_Dealer_Temp'] = ticks_temp['Primary Dealer'].str.strip().str.replace(r'\.$', '', regex=True).str.replace(' ', '')
merged_df_temp['Primary_Dealer_Temp'] = merged_df_temp['Primary Dealer'].str.strip().str.replace(r'\.$', '', regex=True).str.replace(' ', '')
ticks_temp = ticks_temp.drop_duplicates(subset='Primary_Dealer_Temp', keep='last')
merged_df_final = pd.merge(merged_df_temp, ticks_temp[['Primary_Dealer_Temp', 'Holding Company']],
                    on='Primary_Dealer_Temp', how='left')
merged_df_final.drop(columns=['Primary_Dealer_Temp'],inplace=True)
merged_df_final['Start Date'] = pd.to_datetime(merged_df_final['Start Date'])
merged_df_final.sort_values(by='Start Date', inplace=True)
merged_df_final['Start Date'] = merged_df_final['Start Date'].dt.strftime('%-m/%-d/%Y')
merged_df_final = merged_df_final[['Primary Dealer','Holding Company','Start Date']]
blankIndex=[''] * len(merged_df_final)
merged_df_final.index=blankIndex
merged_df_final
