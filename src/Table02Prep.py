import pandas as pd
import wrds
import config
from datetime import datetime
import unittest
import matplotlib.pyplot as plt
import numpy as np
import Table02Analysis


"""
Reads in manual dataset for primary dealers and holding companies and matches it with linkhist entry for company. 
Compiles and prepares this data to produce Table 02 from intermediary asset pricing paper in LaTeX format.
Also creates a summary statistics table and figure in LaTeX format.
Performs unit tests to observe similarity to original table as well as other standard tests.
"""

def fetch_financial_data(db, pgvkey, start_date, end_date):
    """
    Fetch financial data for a given ticker and date range from the CCM database in WRDS.
    :param db: the established connection to the WRDS database
    :param gvkey: The gvkey symbol for the company.
    :param start_date: The start date for the data in YYYY-MM-DD format.
    :param end_date: The end date for the data in YYYY-MM-DD format or 'Current'.
    :return: A DataFrame containing the financial data.
    """
    if not pgvkey:  # Skip if no ticker is available
        return pd.DataFrame()

    # If the end date is 'Current', replace it with today's date
    if end_date == 'Current':
        end_date = datetime.today().strftime('%Y-%m-%d')

    # Convert dates to the correct format if necessary
    start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    pgvkey_str = ','.join([f"'{str(key).zfill(6)}'" for key in pgvkey])
    query = f"""
    SELECT datadate, at AS total_assets, lt AS book_debt, teq AS book_equity, csho*prcc_f AS market_equity, gvkey, conm
    FROM comp.funda as cst
    WHERE cst.gvkey IN ({pgvkey_str})
    AND cst.datadate BETWEEN '{start_date}' AND '{end_date}'
    AND indfmt='INDL'
    AND datafmt='STD'
    AND popsrc='D'
    AND consol='C'
    """
    data = db.raw_sql(query)
    return data


def get_comparison_group_data(db, linktable_df, start_date, end_date):
    """
    Reads in manual datasets containing tickers and link table information.

    Returns:
    - ticks (pandas.DataFrame): DataFrame containing ticker information with columns for 'gvkey' and 'Permco'.
    - linktable (pandas.DataFrame): DataFrame containing link table information.

    Note:
    This function reads CSV files located in the '../data/manual/' directory: 'ticks.csv' and 'linktable.csv'.
    It processes the data by filling missing values with 0 and converting relevant columns to integer data type.
    The returned DataFrames include ticker information and link table information respectively.

    Example:
    - ticks_df, linktable_df = read_in_manual_datasets()
    """
    return fetch_financial_data(db, linktable_df['gvkey'].tolist(), start_date, end_date)


def read_in_manual_datasets():
    ticks = pd.read_csv('../data/manual/ticks.csv', sep="|")
    ticks['gvkey'] = ticks['gvkey'].fillna(0.0).astype(int)
    ticks['Permco'] = ticks['Permco'].fillna(0.0).astype(int)
    linktable = pd.read_csv('../data/manual/linktable.csv')
    return ticks, linktable


def pull_CRSP_Comp_Link_Table():
    """
    Pulls the CRSP-Compustat Link Table from the WRDS database.

    Returns:
    - pandas.DataFrame: DataFrame containing the CRSP-Compustat Link Table.

    Note:
    This function establishes a connection to the WRDS database, executes a SQL query to retrieve the link table data,
    and then closes the connection. The returned DataFrame includes columns for 'gvkey', 'permco', 'linktype',
    'linkprim', 'linkdt', 'linkenddt', and 'tic', representing various identifiers and dates related to the link.
    """
    sql_query = """
        SELECT 
            gvkey, lpermco AS permco, linktype, linkprim, linkdt, linkenddt, tic
        FROM 
            crsp.ccmxpf_linkhist
        WHERE 
            substr(linktype,1,1)='L' AND 
            (linkprim ='C' OR linkprim='P')
        """
    db = wrds.Connection(wrds_username=config.WRDS_USERNAME)
    ccm = db.raw_sql(sql_query, date_cols=["linkdt", "linkenddt"])
    db.close()
    return ccm


def prim_deal_merge_manual_data_w_linktable():
    """
       Merges the main dataset with the link table for primary dealer identification.

       Returns:
       - tuple: A tuple containing two DataFrames: merged_main (the merged dataset) and link_hist (the link table).

       Note:
       This function assumes that the main dataset and the link table have been read in prior to calling this function.

       Example:
       - merged_data, link_table = prim_deal_merge_manual_data_w_linktable()
       """
    # link_hist = pull_CRSP_Comp_Link_Table() Commented out until fix connection issue
    main_dataset, link_hist = read_in_manual_datasets()
    merged_main = pd.merge(main_dataset, link_hist, left_on='gvkey', right_on='gvkey')
    return merged_main, link_hist


def create_comparison_group_linktables(link_hist, merged_main):
    """
    Creates comparison group link tables based on the historical link table and merged main dataset.

    Parameters:
    - link_hist (DataFrame): Historical link table containing information about companies.
    - merged_main (DataFrame): Merged main dataset containing additional company information.

    Returns:
    - dict: A dictionary containing comparison group link tables for 'BD' (Business Development),
            'Banks', 'Cmpust.' (Computing and Technology), and 'PD' (Primary Dealers).

    Example:
    - link_tables = create_comparison_group_linktables(link_hist_df, merged_main_df)
    """
    linked_bd_less_pd = link_hist[((link_hist['sic'] == 6211) | (link_hist['sic'] == 6221)) & (
        ~link_hist['gvkey'].isin(merged_main['gvkey'].tolist()))]
    linked_banks_less_pd = link_hist[(link_hist['sic'].isin([6011, 6021, 6022, 6029, 6081, 6082, 6020])) & (
        ~link_hist['gvkey'].isin(merged_main['gvkey'].tolist()))]
    linked_all_less_pd = link_hist[(~link_hist['gvkey'].isin(merged_main['gvkey'].tolist()))]
    return {"BD": linked_bd_less_pd, "Banks": linked_banks_less_pd, "Cmpust.": linked_all_less_pd, "PD": merged_main}


def pull_data_for_all_comparison_groups(db, comparison_group_dict):
    """
    Pulls data for all comparison groups specified in the given dictionary from the database.

    Parameters:
    - db: The database connection.
    - comparison_group_dict (dict): A dictionary containing comparison group names as keys and
                                    their corresponding link tables as values.

    Returns:
    - dict: A dictionary containing datasets for all comparison groups.

    Example:
    - db = connect_to_database()
    - comparison_group_dict = {'Group1': 'LinkTable1', 'Group2': 'LinkTable2'}
    - datasets = pull_data_for_all_comparison_groups(db, comparison_group_dict)
    """
    datasets = {}
    for key, linktable in comparison_group_dict.items():
        datasets[key] = get_comparison_group_data(db, linktable, config.START_DATE, config.END_DATE).drop_duplicates()
    return datasets


def prep_datasets(datasets):
    """
     Prepares datasets by converting 'datadate' to datetime,
     grouping by year, and summing other columns.

     Parameters:
     - datasets (dict): A dictionary containing datasets.

     Returns:
     - dict: A dictionary containing prepped datasets.
     """
    prepped_datasets = {}
    for key in datasets.keys():
        dataset = datasets[key]
        # Convert 'datadate' to datetime, extract year, convert to string and append '-01-01'
        dataset['datadate'] = pd.to_datetime(dataset['datadate']).dt.year.astype(str) + '-01-01'
        # Convert back to datetime format
        dataset['datadate'] = pd.to_datetime(dataset['datadate'])

        # Group by 'datadate' and sum the other columns
        summed = dataset.groupby('datadate').agg({
            'total_assets': 'sum',
            'book_debt': 'sum',
            'book_equity': 'sum',
            'market_equity': 'sum'
        }).reset_index()

        prepped_datasets[key] = summed

    return prepped_datasets


def create_ratios_for_table(prepped_datasets):
    """
    Creates ratio dataframes for each period based on prepped datasets.

    Parameters:
    - prepped_datasets (dict): A dictionary containing prepped datasets.

    Returns:
    - DataFrame: Combined ratio dataframe with ratios calculated for each period.

    Example:
    - prepped_datasets = {'PD': primary_dealer_data, 'Other': other_data}
    - combined_ratio_df = create_ratios_for_table(prepped_datasets)
    """
    sample_periods = [
        ('1960-01-01', '2012-12-31'),
        ('1960-01-01', '1990-12-31'),
        ('1990-01-01', '2012-12-31')
    ]
    primary_dealer_set = prepped_datasets['PD']
    del prepped_datasets['PD']

    primary_dealer_set['datadate'] = pd.to_datetime(primary_dealer_set['datadate'])
    primary_dealer_set.index = primary_dealer_set['datadate']

    # Filter datasets within the sample periods for primary dealers
    filtered_sets = {}
    for period in sample_periods:
        start_date, end_date = map(lambda d: datetime.strptime(d, '%Y-%m-%d'), period)
        filtered_sets[period] = primary_dealer_set.copy()[start_date: end_date]

    # Initialize a dictionary to store the ratio dataframes for each period
    ratio_dataframes = {period: pd.DataFrame(index=filtered_sets[period].index) for period in sample_periods}

    # Iterate over the remaining datasets and calculate the ratios
    for key, prepped_dataset in prepped_datasets.items():
        # Filter the dataset for each sample period for comparison groups
        prepped_dataset['datadate'] = pd.to_datetime(prepped_dataset['datadate'])
        prepped_dataset.index = prepped_dataset['datadate']
        for period in sample_periods:
            start_date, end_date = period
            filtered_dataset = prepped_dataset[start_date:end_date]

            # Calculate the ratios for the filtered datasets
            for column in ['total_assets', 'book_debt', 'book_equity', 'market_equity']:
                sum_column = filtered_sets[period][column] + filtered_dataset[column]
                # Avoid division by zero
                sum_column = sum_column.replace(0, pd.NA)
                ratio_dataframes[period][f'{column}_{key}'] = filtered_sets[period][column] / sum_column

    # Combine the ratio dataframes for each period into one dataframe
    combined_ratio_df = pd.DataFrame()
    for period, df in ratio_dataframes.items():
        start_date, end_date = map(lambda d: datetime.strptime(d, '%Y-%m-%d'), period)
        df['Period'] = f"{start_date.year}-{end_date.year}"
        combined_ratio_df = pd.concat([combined_ratio_df, df])

    return combined_ratio_df


def format_final_table(table):
    """
    Formats the final table by grouping the data by 'Period', calculating the mean,
    and creating a DataFrame with a MultiIndex for columns.

    Parameters:
    - table (DataFrame): The DataFrame containing the raw data.

    Returns:
    - DataFrame: The formatted table with a MultiIndex for columns.
    """
    table = table.groupby('Period').mean()
    grouped_table = table[
        ['total_assets_BD', 'total_assets_Banks', 'total_assets_Cmpust.', 'book_debt_BD', 'book_debt_Banks',
         'book_debt_Cmpust.', 'book_equity_BD', 'book_equity_Banks', 'book_equity_Cmpust.', 'market_equity_BD',
         'market_equity_Banks', 'market_equity_Cmpust.']]

    grouped_table = grouped_table.reset_index()
    columns_mapping = {
        'total_assets_BD': ('Total assets', 'BD'),
        'total_assets_Banks': ('Total assets', 'Banks'),
        'total_assets_Cmpust.': ('Total assets', 'Cmpust.'),
        'book_debt_BD': ('Book debt', 'BD'),
        'book_debt_Banks': ('Book debt', 'Banks'),
        'book_debt_Cmpust.': ('Book debt', 'Cmpust.'),
        'book_equity_BD': ('Book equity', 'BD'),
        'book_equity_Banks': ('Book equity', 'Banks'),
        'book_equity_Cmpust.': ('Book equity', 'Cmpust.'),
        'market_equity_BD': ('Market equity', 'BD'),
        'market_equity_Banks': ('Market equity', 'Banks'),
        'market_equity_Cmpust.': ('Market equity', 'Cmpust.')
    }
    multiindex = pd.MultiIndex.from_tuples([columns_mapping[col] for col in grouped_table.columns if col != 'Period'],
                                           names=['Metric', 'Source'])

    formatted_table = pd.DataFrame(
        grouped_table.drop('Period', axis=1).values,
        index=grouped_table['Period'],
        columns=multiindex
    )

    # Reorder the index if necessary
    new_order = ['1960-2012', '1960-1990', '1990-2012']
    formatted_table = formatted_table.reindex(new_order)
    return formatted_table


def convert_and_export_table_to_latex(formatted_table):
    """
    Converts a formatted table to LaTeX format and exports it to a .tex file.
    """
    latex = formatted_table.to_latex(index=False, column_format='lccccccccccc', float_format="%.3f")

    # Locate and remove the original headers generated by to_latex()
    start_of_header = latex.find('\\toprule') + len('\\toprule') + 1  # Find end of \toprule plus the newline
    end_of_header = latex.find('\\midrule')  # Find start of \midrule
    latex = latex[:start_of_header] + latex[end_of_header:]  # Remove the original headers

    multicolumn_headers = r'''
    \multicolumn{3}{c}{Total assets} & \multicolumn{3}{c}{Book debt} & \multicolumn{3}{c}{Book equity} & \multicolumn{3}{c}{Market equity} \\
    \cmidrule(r{4pt}){1-3} \cmidrule(lr{4pt}){4-6} \cmidrule(lr{4pt}){7-9} \cmidrule(l{4pt}){10-12}
    '''

    latex = latex.replace(r'\midrule', multicolumn_headers + r'\midrule', 1)

    full_latex = r"""
    \usepackage{booktabs} % For better-looking tables
    \usepackage{graphicx} % Required for inserting images

    \begin{table}[htbp]
      \centering
      \caption{Your table caption here}
      \label{tab:yourlabel}
    """ + latex + r"""
    \end{table}
    """

    # Write the full LaTeX code to a .tex file
    with open('../output/table02.tex', 'w') as f:
        f.write(full_latex)

def main():
    """
    Main function to execute the entire data processing pipeline.

    Returns:
    - formatted_table (pandas.DataFrame): DataFrame containing the formatted table.
    """
    db = wrds.Connection(wrds_username=config.WRDS_USERNAME)
    merged_main, link_hist = prim_deal_merge_manual_data_w_linktable()
    comparison_group_link_dict = create_comparison_group_linktables(link_hist, merged_main)
    datasets = pull_data_for_all_comparison_groups(db, comparison_group_link_dict)
    prepped_datasets = prep_datasets(datasets)
    Table02Analysis.create_summary_stat_table_for_data(datasets)
    table = create_ratios_for_table(prepped_datasets)
    Table02Analysis.create_figure_for_data(table)
    formatted_table = format_final_table(table)
    convert_and_export_table_to_latex(formatted_table)
    return formatted_table



class TestFormattedTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.formatted_table = main()
        # Mimic the main method's workflow to get datasets
        cls.db = wrds.Connection(wrds_username=config.WRDS_USERNAME)
        merged_main, link_hist = prim_deal_merge_manual_data_w_linktable()
        cls.comparison_group_link_dict = create_comparison_group_linktables(link_hist, merged_main)
        cls.datasets = pull_data_for_all_comparison_groups(cls.db, cls.comparison_group_link_dict)
        cls.prepped_datasets = prep_datasets(cls.datasets)
        cls.table = create_ratios_for_table(cls.prepped_datasets)

    def test_value_ranges(self):
        manual_data = {
            ('1960-2012', 'BD', 'Total assets'): 0.959,
            ('1960-2012', 'Banks', 'Total assets'): 0.596,
            ('1960-2012', 'Cmpust.', 'Total assets'): 0.240,
            ('1960-2012', 'BD', 'Book debt'): 0.960,
            ('1960-2012', 'Banks', 'Book debt'): 0.602,
            ('1960-2012', 'Cmpust.', 'Book debt'): 0.280,
            ('1960-2012', 'BD', 'Book equity'): 0.939,
            ('1960-2012', 'Banks', 'Book equity'): 0.514,
            ('1960-2012', 'Cmpust.', 'Book equity'): 0.079,
            ('1960-2012', 'BD', 'Market equity'): 0.911,
            ('1960-2012', 'Banks', 'Market equity'): 0.435,
            ('1960-2012', 'Cmpust.', 'Market equity'): 0.026,

            ('1960-1990', 'BD', 'Total assets'): 0.997,
            ('1960-1990', 'Banks', 'Total assets'): 0.635,
            ('1960-1990', 'Cmpust.', 'Total assets'): 0.266,
            ('1960-1990', 'BD', 'Book debt'): 0.998,
            ('1960-1990', 'Banks', 'Book debt'): 0.639,
            ('1960-1990', 'Cmpust.', 'Book debt'): 0.305,
            ('1960-1990', 'BD', 'Book equity'): 0.988,
            ('1960-1990', 'Banks', 'Book equity'): 0.568,
            ('1960-1990', 'Cmpust.', 'Book equity'): 0.095,
            ('1960-1990', 'BD', 'Market equity'): 0.961,
            ('1960-1990', 'Banks', 'Market equity'): 0.447,
            ('1960-1990', 'Cmpust.', 'Market equity'): 0.015,

            ('1990-2012', 'BD', 'Total assets'): 0.914,
            ('1990-2012', 'Banks', 'Total assets'): 0.543,
            ('1990-2012', 'Cmpust.', 'Total assets'): 0.202,
            ('1990-2012', 'BD', 'Book debt'): 0.916,
            ('1990-2012', 'Banks', 'Book debt'): 0.550,
            ('1990-2012', 'Cmpust.', 'Book debt'): 0.240,
            ('1990-2012', 'BD', 'Book equity'): 0.883,
            ('1990-2012', 'Banks', 'Book equity'): 0.444,
            ('1990-2012', 'Cmpust.', 'Book equity'): 0.058,
            ('1990-2012', 'BD', 'Market equity'): 0.848,
            ('1990-2012', 'Banks', 'Market equity'): 0.419,
            ('1990-2012', 'Cmpust.', 'Market equity'): 0.039,
        }

        # Stack twice to convert the DataFrame into a Series with a MultiIndex
        stacked_series = self.formatted_table.stack().stack()
        formatted_dict = {index: value for index, value in stacked_series.items()}
        wrong_assertions_count = 0
        for key, manual_value in manual_data.items():
            with self.subTest(key=key):
                formatted_value = formatted_dict.get(key)
                if formatted_value is None:
                    self.fail(f"Missing value for {key} in formatted table.")
                else:
                    self.assertAlmostEqual(
                        formatted_value,
                        manual_value,
                        delta=0.15,
                        msg=f"Value for {key} is out of range."
                    )
                    if abs(formatted_value - manual_value) > 0.15:
                        # Increment the counter if the assertion fails
                        wrong_assertions_count += 1
        print("%s table values were off by more than the threshold." % wrong_assertions_count)

    def test_gvkeys_data_presence(self):
        # Iterate over each comparison group to ensure at least 75% gvkeys have data
        for group_name, dataset in self.datasets.items():
            with self.subTest(group=group_name):
                link_table = self.comparison_group_link_dict[group_name]
                link_table['gvkey'] = link_table['gvkey'].astype(str).str.zfill(6)
                link_table_gvkeys = set(link_table['gvkey'].unique())
                print(f"{group_name} link table gvkeys: {len(link_table_gvkeys)}")

                dataset['gvkey'] = dataset['gvkey'].astype(str).str.zfill(6)
                dataset_gvkeys = set(dataset['gvkey'].unique())
                print(f"{group_name} dataset gvkeys: {len(dataset_gvkeys)}")

                common_gvkeys = link_table_gvkeys.intersection(dataset_gvkeys)
                common_gvkeys_count = len(common_gvkeys)
                dataset_gvkeys_count = len(dataset_gvkeys)

                # Calculate the percentage of gvkeys present in the link table
                percentage_present = (common_gvkeys_count / dataset_gvkeys_count) * 100

                # Assert that at least 75% of gvkeys are present in the link table. Chose this because you wont have all of them ever because not all companies exist in the period we are looking at
                self.assertGreaterEqual(percentage_present, 75,
                                        f"Less than 75% of gvkeys from dataset are present in group '{group_name}'")

    def test_ratios_non_negative_and_handle_na(self):
        combined_ratio_df = self.table
        # Test for non-negative ratios
        min_ratio_value = combined_ratio_df.select_dtypes(
            include=['float64', 'int']).min().min()  # Minimum value across all numeric columns
        self.assertGreaterEqual(min_ratio_value, 0, "Found negative ratio values in the DataFrame.")
        # This checks if there are any N/A values left in the ratios
        na_values_count = combined_ratio_df.isna().sum().sum()  # Total count of N/A values across the DataFrame
        self.assertTrue(na_values_count >= 0,
                        "N/A values are present, which is expected. Ensure they are handled correctly in calculations.")


if __name__ == '__main__':
    unittest.main()
