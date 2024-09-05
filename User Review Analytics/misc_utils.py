
'''
    Contains basic utility functions for user review analytics.
    
    __author__ = ''
    __email__ = ''
    __date__ = ''
    __version__ = ''
'''

import os
import gc
from typing import Dict, Any, Optional

import pyarrow.parquet as pq
import pandas as pd


def fetch_data(config, file_name: str, verbose: Optional[bool]=False) -> pd.DataFrame:
    '''
        Function to Read data from the requested file.
        Args:
            config (Dict): Configuration dictionary
            file_name (str): Name of the file to fetch data from
            verbose (Optional[bool]): Whether to print verbose information
        Returns:
            pd.DataFrame: Dataframe containing the data
    '''
    assert file_name is not None

    filepath = os.path.join(config.config_dict['extracted_data_dir'], 
                            'yelp_academic_dataset_'+file_name+'.parquet')

    if verbose: print(f"Reading file: {filepath}")
    
    if file_name == 'business':
        cols_to_read = ['business_id', 'name', 'address', 'city', 'postal_code', 'categories']
        df = pq.read_table(filepath, columns=cols_to_read).to_pandas()  # use fastparquet instead???

    elif file_name == 'review':
        cols_to_read = ['business_id', 'text']
        df = pq.read_table(filepath, columns=cols_to_read).to_pandas()
    
    if verbose: print("df.shape: ", df.shape)
    gc.collect()

    return df


def get_business_id(config: Dict, restanrant_name: str, city: str) -> Optional[str]:
    '''
        Function to get the business id for the restaurant based on the name and city.
        Returns None if no business found or multiple businesses found.
        
        Args:
            config (Dict): Configuration dictionary
            restanrant_name (str): Restaurant name
            city (str): City
        Returns:
            str: Business ID
    '''
    business_df = fetch_data(config, 'business')
    
    business_data = business_df[(business_df['name'] == restanrant_name) & 
                                (business_df['city'] == city)].business_id

    if business_data.shape[0] == 0:
        print(f"No business found for restaurant: {restanrant_name} in city: {city}")
        return None
    elif business_data.shape[0] > 1:
        print(f"Multiple businesses found for restaurant: {restanrant_name} in city: {city}")
        print("Please select one of the following business ids: ")
        print(business_data)

        return None

    id = business_data.values[0]
    del business_data
    gc.collect()

    return id
