
"""
preprocess_data.py

This file contains functions for preprocessing data in a pandas DataFrame. 
It includes functions for reading data from a parquet file, filtering data, 
handling outliers, removing duplicates, encoding date columns, scaling data, and more.

Author: Jaideep V. Murkute
Date: 2024-03-13
Version: 1.0

Change Log:

"""

import gc
import os
import json

import numpy as np 
import pandas as pd 
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from CFG import Config

# ---------------------------------------------------------------------------

def read_data_file(config, file_name=None, verbose=False):
    """
    Reads a data file in parquet format and returns it as a pandas DataFrame.

    Args:
        config (dict): A dictionary containing configuration parameters.
        file_name (str): The name of the file to be read (without the file extension).
        verbose (bool): If True, prints additional information during the reading process.

    Returns:
        pandas.DataFrame: The data read from the file.

    Raises:
        AssertionError: If the file_name argument is not provided.

    """
    assert file_name is not None

    filepath = os.path.join(config['data_dir'], file_name+'.parquet')

    if verbose: print(f"Reading file: {filepath}")
    df = pq.read_table(filepath).to_pandas()
    if verbose: print("df.shape: ", df.shape)

    return df


def filter_data(config, df, file_name=None, drop_cols=None, dropna=False, dropinf=False, verbose=False):
    
    if verbose: print("Before filtering shape: ", df.shape)
    
    # drop columns that are not needed
    if drop_cols is not None:
        if verbose: 
            print("Before dropping columns: ", df.columns)
            print("Before dropping columns shape:", df.shape)
        df.drop(drop_cols, axis=1, inplace=True)
        if verbose: 
            print("After dropping columns: ", df.columns)
            print("After dropping columns shape:", df.shape)
        
    # Filter out rows with NaN values
    if dropna:
        if verbose: print("Before dropping NaNs shape:", df.shape)
        df.dropna(inplace=True)
        if verbose: print("After dropping NaNs shape:", df.shape)

    # Filter out rows with inf values
    if dropinf:
        if verbose: print("Before dropping Infs shape:", df.shape)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        if verbose: print("After dropping Infs shape:", df.shape)

    # Perform any file specific processing, if needed
    assert file_name is not None
    if file_name == 'business':
        if verbose: print("Before filterin out non-restaurant data shape:", df.shape)
        df = df[df['categories'].str.lower().str.contains('restaurants')]
        if verbose: print("After filterin out non-restaurant data shape:", df.shape)
    
    if verbose: print("After filtering shape: ", df.shape)

    return df


def handle_outliers_fn(df, cols, mode='drop', threshold=3, verbose=False):
    """
    Handles outliers in a DataFrame based on the specified mode.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        cols (list): A list of column names to handle outliers for.
        mode (str, optional): The mode for handling outliers. Defaults to 'drop'.
            - 'drop': Drops the rows containing outliers.
            - 'clip': Clips the outliers to the specified threshold.
        threshold (float, optional): The threshold value for identifying outliers. Defaults to 3.
        verbose (bool, optional): Whether to print additional information. Defaults to False.

    Returns:
        pandas.DataFrame: The DataFrame with outliers handled.

    """
    if verbose:
        print("Before outlier handling shape: ", df.shape)
    
    if mode == 'drop':
        print("Dropping outliers...")
        for col in cols:
            z_scores = (df[col] - df[col].mean()) / df[col].std()
            df = df[z_scores.abs() < threshold]
    elif mode == 'clip':
        print("Clipping outliers...")
        for col in cols:
            lower_bound = df[col].mean() - threshold * df[col].std()
            upper_bound = df[col].mean() + threshold * df[col].std()
            df[col] = df[col].clip(lower_bound, upper_bound)
    
    if verbose:
        print("After outlier handling shape: ", df.shape)

    return df


def remove_duplicates(df, verbose=False):
    """
    Removes duplicate rows from a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to remove duplicates from.
        verbose (bool, optional): If True, print the shape of the DataFrame 
            before and after removing duplicates. Defaults to False.

    Returns:
        pandas.DataFrame: The DataFrame with duplicate rows removed.
    """
    if verbose: 
        print("Before removing duplicates shape: ", df.shape)
    df.drop_duplicates(inplace=True)
    if verbose: 
        print("After removing duplicates shape: ", df.shape)

    return df
    

def convert_datetime_column_debug_fn(df, col_name):
    """
    A funtion to debug issues in datetime conversion. Not needed for core functionality.

    Args:
        df (pandas.DataFrame): The DataFrame containing the column to be converted.
        col_name (str): The name of the column to be converted.

    Returns:
        pandas.DataFrame: The DataFrame with the converted datetime column.

    Raises:
        None

    Example:
        df = pd.DataFrame({'date': ['2022-01-01 12:00:00', '2022-01-02 13:00:00']})
        df = convert_datetime_column(df, 'date')
        print(df['date'])

    """
    for idx, dt_str in enumerate(df[col_name]):
        try:
            df.loc[idx, col_name] = pd.to_datetime(dt_str, format='%Y-%m-%d %H:%M:%S') 
        except ValueError:
            print(f"Failed to convert datetime string at index {idx}: {dt_str}")    
    return df


def encode_dates_fn(config, df, verbose=False):
    """
    Encodes date columns in a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to encode date columns in.
        date_cols (list, optional): A list of column names to encode as dates. If not provided, columns containing the word 'date' in their name will be inferred as date columns. Defaults to None.
        verbose (bool, optional): Whether to print the shape of the DataFrame before and after encoding. Defaults to False.

    Returns:
        pandas.DataFrame: The DataFrame with encoded date columns.
    """
    if verbose:
        print("Before encoding data shape: ", df.shape)

    if config['date_cols'] is None:
        # infer date columns based on the names
        date_cols = [col for col in df.columns if 'date' in col.lower()]
    else:
        date_cols = config['date_cols']
    
    if len(date_cols) > 0:
        for col in date_cols:
            df[col] = df[col].apply(lambda x: x.lstrip(', '))
            df[col] = df[col].apply(lambda x: x.strip(' '))
            df[col] = pd.to_datetime(df[col], format='%Y-%m-%d %H:%M:%S')

    if verbose:
        print("After encoding data shape: ", df.shape)
    
    return df


def get_column_descriptor_dict(config, df, verbose=False):
    """
    Returns a dictionary containing column descriptors based on the provided DataFrame.

    Args:
        config (object): The configuration object.
        df (pandas.DataFrame): The DataFrame to analyze.
        id_cols (list, optional): A list of column names to be considered as ID columns. Defaults to None.
        verbose (bool, optional): If True, prints the column descriptors. Defaults to False.

    Returns:
        dict: A dictionary containing the following column descriptors:
            - 'id_cols': A list of column names containing 'id' in their lowercase form.
            - 'int_cols': A list of column names with integer data type.
            - 'float_cols': A list of column names with float data type.
            - 'nume_cols': A list of column names with either integer or float data type.
            - 'non_nume_cols': A list of column names with data types other than integer or float.

    """
    col_descriptor_dict = dict()

    if config['id_cols'] is None:
        col_descriptor_dict['id_cols'] = [col for col in df.columns if 'id' in col.lower()]
    else:
        col_descriptor_dict['id_cols'] = config['id_cols']
    col_descriptor_dict['int_cols'] = [col for col in df.columns if 'int' in str(df[col].dtype)]
    col_descriptor_dict['float_cols'] = [col for col in df.columns if 'float' in str(df[col].dtype)]
    col_descriptor_dict['nume_cols'] = col_descriptor_dict['int_cols'] + \
                                        col_descriptor_dict['float_cols']
    col_descriptor_dict['non_nume_cols'] = [col for col in df.columns \
                                            if col not in col_descriptor_dict['nume_cols']]

    if verbose:
        for k, v in col_descriptor_dict.items():
            print(f"{k}: ")
            print(f"{v}")
            print("-"*30)

    return col_descriptor_dict


def get_scaler(scaler_type):
    """
    Returns a scikit-learn scaler object based on the specified scaler type.

    Args:
        scaler_type (str): Type of scaler. Options are 'standard', 'min_max', 'robust'.  

    Returns:
        sklearn.preprocessing.scaler: A scikit-learn scaler object.
    """

    if scaler_type == 'standard':
        # standardize to zero mean and unit variance
        scaler = StandardScaler()  
    elif scaler_type == 'min_max':
        # scale features to be in [0, 1] range
        scaler = MinMaxScaler()  
    elif scaler_type == 'robust':
        # scale with robust outlier handling
        scaler = RobustScaler()   
    else:
        raise ValueError(f"Invalid scaler_type: {scaler_type}")  # Handle invalid input 
    
    return scaler


def scale_data_fn(df, scaler_type_cols_map, scaler_type='standard', verbose=False):
    """
    Scales the data in the given DataFrame using the specified scaler type for the specified columns.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data to be scaled.
        scaler_type_cols_map (dict): A dictionary mapping scaler types to the columns to be scaled.
        scaler_type (str, optional): The type of scaler to be used. Defaults to 'standard'.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        pandas.DataFrame: The DataFrame with the scaled data.

    """
    if verbose:
        print("Before scaling data shape: ", df.shape)
    
    for scaler_type, cols in scaler_type_cols_map.items():
        print(f"Using {scaler_type} scaler for cols: {cols}")
        scaler = get_scaler(scaler_type)
        for col in cols:
            df[[col]] = scaler.fit_transform(df[[col]])
        
    if verbose:
        print("After scaling data shape: ", df.shape)
    
    return df


def preprocess_data(config, df, encode_dates=False, date_cols=None, id_cols=None, 
                    handle_outliers=False, 
                    scale_data=False, cols_scaler_type_map=None, verbose=False):
    """
    Preprocesses the given dataframe based on the provided configuration and 
    parameters.

    Args:
        config (dict): The configuration dictionary containing information about 
            the data preprocessing steps.

        df (pandas.DataFrame): The input dataframe to be preprocessed.
        
        encode_dates (bool, optional): Whether to encode date columns. 
            Defaults to False.
        
        date_cols (list, optional): List of column names to be treated as date 
            columns. Defaults to None.
        
        id_cols (list, optional): List of column names to be treated as 
            identifier columns. Defaults to None.
        
        handle_outliers (bool, optional): Whether to handle outliers in the 
            numerical columns. Defaults to False.
        
        scale_data (bool, optional): Whether to scale the data. Defaults to False.
        
        cols_scaler_type_map (dict, optional): A dictionary mapping column 
            names to the type of scaler to be used. Defaults to None.
        
        verbose (bool, optional): Whether to print verbose output. 
            Defaults to False.

    Returns:
        tuple: A tuple containing the preprocessed dataframe and a dictionary 
                        containing column descriptors.
    """
    
    df = remove_duplicates(df, verbose=verbose)
    
    if config['encode_dates']:
        df = encode_dates_fn(df, date_cols, verbose)
    
    col_descriptor_dict = get_column_descriptor_dict(config, df, 
                    id_cols=config['id_cols'], verbose=verbose)
    
    if config['handle_outliers']:
        df = handle_outliers_fn(df, col_descriptor_dict['nume_cols'], 
                                mode='drop', threshold=3, verbose=verbose)
    
    scaler_type_cols_map = {'standard': col_descriptor_dict['float_cols']}
    if config['scale_data'] and scaler_type_cols_map:
        df = scale_data_fn(df, scaler_type_cols_map, verbose=verbose)
    
    return df, col_descriptor_dict
    

# ----------------------------------------------------------------------------------

if __name__ == '__main__':
    config_ref = Config()
    config = config_ref.get_config()
    data_preprocessing_config = config_ref.get_data_preprocessing_config()
     
    for file_name, processing_config in file_preprocessing_config_map.items():
        df = fetch_data(config)

        df, col_descriptor_dict = preprocess_data(processing_config, df, 
                        verbose=False)





















