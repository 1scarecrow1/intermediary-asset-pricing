import pandas as pd
import wrds
import config
import Table03
import unittest
import numpy as np
import Table02Prep
import Table03Load

"""
Tests for Table03.py
"""

class TestFormattedTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.formatted_table = Table03.main()
        # Mimic the main method's workflow to get datasets
        cls.db = wrds.Connection(wrds_username=config.WRDS_USERNAME)
        cls.prim_dealers, _ = Table02Prep.prim_deal_merge_manual_data_w_linktable()
        cls.dataset, _ = Table03Load.fetch_data_for_tickers(cls.prim_dealers, cls.db)    
        cls.prep_datast = Table03.prep_dataset(cls.dataset)
        cls.macro_dataset = Table03.macro_variables(cls.db)
        cls.ratio_dataset = Table03.aggregate_ratios(cls.prep_datast)
        cls.factors_dataset = Table03.convert_ratios_to_factors(cls.ratio_dataset)
        cls.panelA = Table03.create_panelA(cls.ratio_dataset, cls.macro_dataset)
        cls.panelB = Table03.create_panelB(cls.factors_dataset, cls.macro_dataset)
        cls.correlation_panelA = Table03.calculate_correlation_panelA(cls.panelA)
        cls.correlation_panelB = Table03.calculate_correlation_panelB(cls.panelB)
        cls.final_corr = Table03.final_correlation(cls.correlation_panelA, cls.correlation_panelB)


    def test_correlation_panelA(self):
        data_panel_a = {
            'Market capital': [1.00, 0.50, 0.42, -0.83, -0.63, 0.18, -0.48, -0.06],
            'Book capital': [np.nan, 1.00, -0.07, -0.38, -0.10, 0.32, -0.53, -0.31],
            'AEM leverage': [np.nan, np.nan, 1.00, -0.64, -0.33, -0.23, -0.19, 0.33]
        }
        index_panel_a = ['Market capital', 'Book capital', 'AEM leverage', 'E/P', 'Unemployment', 'GDP', 'Financial conditions', 'Market volatility']
        panel_a = pd.DataFrame(data_panel_a, index=index_panel_a)
        panel_a.fillna(panel_a.T, inplace=True)

        closeA = np.isclose(self.correlation_panelA, panel_a, atol=0.15)
        
        # Asserting that all values are close within the tolerance
        self.assertTrue(closeA.all(), "Not all values in generated paneA are within 0.15 of correlations in Panel 3A")
    
    
    def test_correlation_panelB(self):
        data_panel_b = {
        "Market capital factor": [1.00, 0.30, 0.14, 0.78, -0.75, -0.05, 0.20, -0.38, -0.49],
        "Book capital factor": [0.30, 1.00, -0.06, 0.10, -0.10, 0.12, 0.09, -0.29, -0.18],
        "AEM leverage factor": [0.14, -0.06, 1.00, 0.15, -0.18, -0.08, 0.04, -0.06, -0.08],
        }
        index_panel_b = ["Market capital factor", "Book capital factor", "AEM leverage factor", "Market excess return", "E/P growth", "Unemployment growth", "GDP growth", "Financial conditions growth", "Market volatility growth"]
        panel_b = pd.DataFrame(data_panel_b, index=index_panel_b)
        panel_b.fillna(panel_b.T, inplace=True)

        closeB = np.isclose(self.correlation_panelB, panel_b, atol=0.15)

        self.assertTrue(closeB.all(), "Not all values in generated panelB are within 0.15 of correlations in Panel 3B")


if __name__ == '__main__':
    unittest.main()
