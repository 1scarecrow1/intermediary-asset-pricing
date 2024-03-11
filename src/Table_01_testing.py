import pandas as pd
import Table_01

"""
Tests for Table_01.py
"""

def test_primary_dealer_start_dates(merged_df_final):
    """
    Test that the 'Primary Dealer' names (with spaces and periods removed and specific naming exceptions handled) 
    and 'Start Date' in Table_01.merged_df_final match those in the expected_df. Specific exceptions include 
    standardizing the name for "Bank of Nova Scotia, New York Agency" to "Bank of Nova Scotia, NY Agency" and 
    abbreviating "Merrill Lynch, Pierce, Fenner & Smith Incorporated" to "Merrill Lynch, Pierce, Fenner & Smith".
    """
    expected_data = [
        {'Primary Dealer': 'Goldman, Sachs & Co.', 'Start Date': '12/4/1974'},
        {'Primary Dealer': 'Barclays Capital Inc.', 'Start Date': '4/1/1998'},
        {'Primary Dealer': 'HSBC Securities (USA) Inc.', 'Start Date': '6/1/1999'},
        {'Primary Dealer': 'BNP Paribas Securities Corp.', 'Start Date': '9/15/2000'},
        {'Primary Dealer': 'Deutsche Bank Securities Inc.', 'Start Date': '3/30/2002'},
        {'Primary Dealer': 'Mizuho Securities USA Inc.', 'Start Date': '4/1/2002'},
        {'Primary Dealer': 'Citigroup Global Markets Inc.', 'Start Date': '4/7/2003'},
        {'Primary Dealer': 'UBS Securities LLC', 'Start Date': '6/9/2003'},
        {'Primary Dealer': 'Credit Suisse Securities (USA) LLC', 'Start Date': '1/16/2006'},
        {'Primary Dealer': 'Cantor Fitzgerald & Co.', 'Start Date': '8/1/2006'},
        {'Primary Dealer': 'RBS Securities Inc.', 'Start Date': '4/1/2009'},
        {'Primary Dealer': 'Nomura Securities International, Inc.', 'Start Date': '7/27/2009'},
        {'Primary Dealer': 'Daiwa Capital Markets America Inc.', 'Start Date': '4/1/2010'},
        {'Primary Dealer': 'J.P. Morgan Securities LLC', 'Start Date': '9/1/2010'},
        {'Primary Dealer': 'Merrill Lynch, Pierce, Fenner & Smith', 'Start Date': '11/1/2010'},
        {'Primary Dealer': 'RBC Capital Markets, LLC', 'Start Date': '11/1/2010'},
        {'Primary Dealer': 'SG Americas Securities, LLC', 'Start Date': '2/2/2011'},
        {'Primary Dealer': 'Morgan Stanley & Co. LLC', 'Start Date': '5/31/2011'},
        {'Primary Dealer': 'BMO Capital Markets Corp.', 'Start Date': '10/4/2011'},
        {'Primary Dealer': 'Bank Of Nova Scotia, NY Agency', 'Start Date': '10/4/2011'},
        {'Primary Dealer': 'Jefferies LLC', 'Start Date': '3/1/2013'},
        {'Primary Dealer': 'TD Securities (USA) LLC', 'Start Date': '2/11/2014'}
    ]

    # Create the expected DataFrame
    expected_df = pd.DataFrame(expected_data)

    # Normalize the 'Primary Dealer' to ensure case-insensitive comparison
    # Handle specific name abbreviation exceptions
    Table_01.merged_df_final['Primary Dealer'] = Table_01.merged_df_final['Primary Dealer'].str.lower().str.replace(' ', '').str.replace('.','')
    Table_01.merged_df_final['Primary Dealer'] = Table_01.merged_df_final['Primary Dealer'].replace({'bankofnovascotia,newyorkagency': 'bankofnovascotia,nyagency',
                                                                                   'merrilllynch,pierce,fenner&smithincorporated': 'merrilllynch,pierce,fenner&smith'})
    expected_df['Primary Dealer'] = expected_df['Primary Dealer'].str.lower().str.replace(' ', '').str.replace('.','')

    # Sort both dataframes by 'Primary Dealer' for a direct row-wise comparison
    merged_df_sorted = Table_01.merged_df_final.sort_values(by='Primary Dealer').reset_index(drop=True)
    expected_df_sorted = expected_df.sort_values(by='Primary Dealer').reset_index(drop=True)

    # Check if the sorted DataFrames are equal, considering only 'Primary Dealer' and 'Start Date'
    try:
        pd.testing.assert_frame_equal(merged_df_sorted[['Primary Dealer', 'Start Date']],
                                    expected_df_sorted[['Primary Dealer', 'Start Date']],
                                    check_dtype=False)
    except AssertionError as e:
        raise AssertionError("Either 'Primary Dealer' names or 'Start Dates' do not match between the dataframes.")