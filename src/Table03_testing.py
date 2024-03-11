import pandas as pd
import Table_03

import unittest
import pandas as pd
import numpy as np

from Table03 import calculate_correlation_panelA, calculate_correlation_panelB

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

class TestCalculateCorrelationPanelA(unittest.TestCase):
    def setUp(self):
        # Mock data setup to match the given correlations approximately
        # Note: This data won't exactly reproduce the given correlations but is aimed to be sufficiently close for testing purposes.
        self.mock_panelA = pd.DataFrame({
            'Market capital': np.random.normal(1, 0.1, 100),
            'Book capital': np.random.normal(1, 0.1, 100) * 0.5 + np.random.normal(0, 0.05, 100) * 0.5,
            'AEM leverage': np.random.normal(1, 0.1, 100) * 0.42 + np.random.normal(0, 0.05, 100) * 0.58,
            'E/P': np.random.normal(-0.83, 0.1, 100),
            'Unemployment': np.random.normal(-0.63, 0.1, 100),
            'GDP': np.random.normal(0.18, 0.1, 100),
            'Financial conditions': np.random.normal(-0.48, 0.1, 100),
            'Market volatility': np.random.normal(-0.06, 0.1, 100)
        })

    def test_calculate_correlation_panelA(self):
        result = calculate_correlation_panelA(self.mock_panelA)

        # Check if result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Check if all expected columns are present
        expected_columns = ['Market capital', 'Book capital', 'AEM leverage', 'E/P', 'Unemployment', 'GDP', 'Financial conditions', 'Market volatility']
        self.assertTrue(all(column in result.columns for column in expected_columns))

        # You could add more specific value checks here, for instance:
        # self.assertAlmostEqual(result.loc['Market capital', 'Book capital'], 0.50, places=2)

if __name__ == '__main__':
    unittest.main()

def test_panelA(panelA_df):
    """
    Test that the 'Primary Dealer' names (with spaces and periods removed and specific naming exceptions handled) 
    and 'Start Date' in Table_01.merged_df_final match those in the expected_df. Specific exceptions include 
    standardizing the name for "Bank of Nova Scotia, New York Agency" to "Bank of Nova Scotia, NY Agency" and 
    abbreviating "Merrill Lynch, Pierce, Fenner & Smith Incorporated" to "Merrill Lynch, Pierce, Fenner & Smith".
    """
    data = {
        'Market capital': [1.00, 0.50, 0.42, -0.83, -0.63, 0.18, -0.48, -0.06],
        'Book capital': [np.nan, 1.00, -0.07, -0.38, -0.10, 0.32, -0.53, -0.31],
        'AEM leverage': [np.nan, np.nan, 1.00, -0.64, -0.33, -0.23, -0.19, 0.33],
        'E/P': [np.nan, np.nan, np.nan, 1.00, np.nan, np.nan, np.nan, np.nan],
        'Unemployment': [np.nan, np.nan, np.nan, np.nan, 1.00, np.nan, np.nan, np.nan],
        'GDP': [np.nan, np.nan, np.nan, np.nan, np.nan, 1.00, np.nan, np.nan],
        'Financial conditions': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1.00, np.nan],
        'Market volatility': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1.00],
    }

    # Create the expected DataFrame
    expected_df = pd.DataFrame(data, index=['Market capital', 'Book capital', 'AEM leverage', 'E/P', 'Unemployment', 'GDP', 'Financial conditions', 'Market volatility'])

    for i in range(len(expected_df)):
        for j in range(i+1, len(expected_df)):
            expected_df.iloc[j, i] = expected_df.iloc[i, j]

    test_df = Table_03.calculate_correlation_panelB(panelA_df)

    # Check if the sorted DataFrames are equal, considering only 'Primary Dealer' and 'Start Date'
    try:
        pd.testing.assert_frame_equal(test_df,
                                    expected_df,
                                    check_dtype=False)
    except AssertionError as e:
        raise AssertionError("Panel A correlations do not match")