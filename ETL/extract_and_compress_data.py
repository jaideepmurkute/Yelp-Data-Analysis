
"""
The script extracts data from a JSON file, compresses it, and writes it to a Parquet file.

Author: Jaideep Murkute
Date: 2024-03-15
Version: 1.0

"""

import gc
import os
import sys
import json
import numpy as np 
import pandas as pd 
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from CFG import Config


# ------------------------

def reduce_mem_usage(props):
    """
    Reduces the memory usage of a pandas DataFrame by converting the data types 
    of columns to lower memory alternatives.

    Args:
        props (pandas.DataFrame): The DataFrame to reduce memory usage for.

    Returns:
        tuple: A tuple containing the reduced DataFrame and a list of columns 
            that had missing values filled in.

    Raises:
        None

    Example:
        df, na_list = reduce_mem_usage(data)
    """

    start_mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in props.columns:
        gc.collect()
        if props[col].dtype != object:  # Exclude strings
            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",props[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True
            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ",props[col].dtype)
            print("******************************")
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props, NAlist


def load_data_from_json(config):
    """
    Loads data from a JSON file and returns it as a pandas DataFrame.

    Args:
        config (dict): A dictionary containing the configuration parameters.
            - file_path (str): The path to the JSON file.

    Returns:
        pandas.DataFrame: The loaded data as a DataFrame.
    """
    json_data = []
    with open(config['file_path'], encoding='utf-8', errors='ignore') as fp:
        for line in fp:
            # get 'extra data on line xyz' error if using json.loads(fp)
            # because of raw data formatting issues
            json_data.append(json.loads(line))
    
    print("Converting to dataframe...")
    df = pd.DataFrame(json_data)
    gc.collect()

    return df


def extract_data(config):
    """
    Extracts data from a JSON file, compresses it, and writes it to a Parquet file.

    Args:
        config (dict): A dictionary containing the configuration parameters.

    Returns:
        bool: True if the data extraction and compression is successful, False otherwise.
    """
    config['file_path'] = os.path.join(config['downloaded_data_dir'], config['fname']+'.json')
    pq_fpath = os.path.join(config['extracted_data_dir'], config['fname']+'.parquet')
    
    if os.path.exists(pq_fpath) and not config['force_extract_data']:
        print(f"Parquet file already exists for {config['fname']}. force_extrac_data is \
            set to False. Skipping...")
        return True

    try:
        print("Reading data from json file...")
        df = load_data_from_json(config)
        print("Compressing data...")
        df, nalist = reduce_mem_usage(df)
        print("Writing to parquet file...")
        pq.write_table(pa.Table.from_pandas(df), pq_fpath)
        
        del df
        gc.collect()
        return True
    except Exception as e:
        print("Error in extract_data: ", e)
    
    return False


# -------------------------

if __name__ == '__main__':
    
    config_ref = Config()
    config = config_ref.get_config()

    for fname in ['business', 'user', 'review', 'tip', 'checkin']:
        config['fname'] = 'yelp_academic_dataset_'+fname
        extract_data(config)
        print(f"{fname} data extracted, compressed and saved successfully...")
        print("-"*50)
    
