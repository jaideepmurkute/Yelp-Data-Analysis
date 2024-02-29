"""
This module contains functions for interacting with a SQLite database, 
including authentication, data retrieval, and table creation.

Author: Jaideep Murkute
Date: 2024-02-28

"""

import gc
import os

from cryptography.fernet import Fernet
import hashlib
import pandas as pd
import pyarrow.parquet as pq
import sqlite3
from sqlite3 import Error

from CFG import Config
from security_config import SecurityConfig


def hash_password(password, salt):
    """
    Hashes the given password using SHA-256 algorithm with the provided salt.

    Args:
        password (str): The password to be hashed.
        salt (str): The salt to be combined with the password.

    Returns:
        str: The hashed password.

    """
    # Combine the password and salt
    password_salt = password + salt

    # Create a hash object
    hash_object = hashlib.sha256()

    # Update the hash object with the password and salt
    hash_object.update(password_salt.encode('utf-8'))

    # Get the hashed password
    hashed_password = hash_object.hexdigest()

    return hashed_password


def load_security_db_config():
    """
    Load the database security configuration.

    Returns:
        dict: The database security configuration.
    """
    security_config = SecurityConfig()
    security_db_config = security_config.get_config()

    return security_db_config


def authenticate_user(security_db_config, username, entered_password):
    """
    Authenticates a user by comparing the entered password with the stored 
    password hash.

    Args:
        db_security_config (dict): A dictionary containing database security 
        configuration.
        username (str): The username of the user.
        entered_password (str): The password entered by the user.

    Returns:
        bool: True if the entered password matches the stored password hash, 
            False otherwise.
    """
    key_file_path = security_db_config['encryption_key_file_path']
    creds_encryption_key = read_creds_encryption_key(file_path)

    # 1. Retrieve stored credentials - fetch_user_data
    stored_password_hash, salt = fetch_user_creds(security_db_config['db_name'], 
                                            username, creds_encryption_key)
    
    # 2. Hash the entered password using the same salt
    entered_password_hash = hash_password(entered_password, salt) 
    
    # 3. Compare hashes (constant-time comparison) and return auth result
    result = compare_hashes(stored_password_hash, entered_password_hash)

    return result


def create_db_connection(config, db_name, username, authenticate=True, password=None, 
                         create_db_if_not_exists=False):
    """
    Create a connection to the SQLite database.
    If `create_db_if_not_exists` is True; Database will be created if it does 
    not exist.

    Args:
        db_name (str): The name/path of the database.

    Returns:
    conn: SQLite connection object
    """

    if authenticate:
        assert password is not None, "Password is required for authentication."

        security_db_config = load_security_db_config()
        has_auth = authenticate_user(security_db_config, username, password)
        if not has_auth:
            print("Authentication failed.")
            return None
    
    conn = None

    db_path = os.path.join(config['db_storage_dir'], f'{db_name}.db')
    if os.path.exists(db_path):
        print("Database exists. Connecting...")
        try:
            print(f"Connecting to database {db_name}...")
            conn = sqlite3.connect(f'{db_name}.db') # use your own database name
            return conn
        except Error as e:
            print(e)
    else:
        if create_db_if_not_exists:
            try:
                print(f"Creating database {db_name}...")
                conn = sqlite3.connect(db_path) # use your own database name
                print(f"Database created successfully at: {db_path}")
                return conn
            except Error as e:
                print(e)
        else:
            print(f"Database {db_name} does not exist at: {config['db_storage_dir']}")
            return None


def close_connection(conn):
    """
    Closes the database connection.
    
    Parameters:
    conn (connection): The database connection object.
    
    Returns:
    None
    """
    try:
        conn.close()
    except Error as e:
        print(e)


def fetch_user_creds(security_db_name, username, creds_encryption_key):
    """
    Fetches the password hash and salt for a given username from the database.
    
    Args:
        username (str): The username of the user.
        decryption_key (bytes): The decryption key used to decrypt the password 
        hash.
    
    Returns:
        tuple: A tuple containing the decrypted password hash and salt. If the 
        username is not found, returns (None, None).
    """
    
    # sqlite3.connect() - does not provide user auth mechanism; since it is a 
    # file level auth framework. 
    # For our project; the auth for security db can be handled at the file level.
    conn = create_db_connection(config, security_db_name, authenticate=False,
                                create_db_if_not_exists=False)
    if conn is None:
        return None, None

    cursor = conn.cursor()
    try:
        cursor.execute("SELECT password_hash, salt FROM users WHERE \
                       username = ?", (username,))
        result = cursor.fetchone()
        if result:
            encrypted_password_hash, salt = result
            f = Fernet(creds_encryption_key)
            decrypted_password_hash = f.decrypt(encrypted_password_hash)
            return decrypted_password_hash, salt
        else:
            return None, None
    except Error as e:
        print(e)
    finally:
        cursor.close()
        close_connection(conn)


def compare_hashes(hash1, hash2):
    """
    Compare two hashes and return True if they are equal, False otherwise.
    
    Args:
        hash1 (str): The first hash to compare.
        hash2 (str): The second hash to compare.
    
    Returns:
        bool: True if the hashes are equal, False otherwise.
    """
    return hash1 == hash2


def write_creds_encryption_key(key, file_path):
    """
    Write the encryption key to a file - in plaintext.

    Args:
        key (str): The encryption key to be written.
        file_path (str): The path to the file where the key will be written.

    Returns:
        None
    """
    with open(file_path, 'wb') as file:
        file.write(key.encode())


def read_creds_encryption_key(file_path):
    """
    Reads the encryption key from the specified file.

    Args:
        file_path (str): The path to the file containing the encryption key.

    Returns:
        bytes: The encryption key read from the file.
    """
    try:
        with open(file_path, 'rb') as file:
            encryption_key = file.read()
        return encryption_key
    except IOError as e:
        print(f"Error reading encryption key from file: {e}")
        return None


def create_db_table_from_parquet(config, file_path, table_name, 
                                 recreate_if_exists=False):
    """
    Reads a parquet data file, converts it to a pandas dataframe, and inserts 
    it into a database table.

    Args:
        file_path (str): The path to the parquet data file.
        table_name (str): The name of the table to be created in the database.
        username (str): The username for authentication.
        password (str): The password for authentication.

    Returns:
        None
    """
    df = pq.read_table(file_path).to_pandas()

    conn = create_db_connection(config, 
                                db_name=config['data_db_name'], 
                                username=config['db_access_uname'],
                                password=config['db_access_pwd'],
                                authenticate=True,
                                create_db_if_not_exists=True)
    
    # Check if table already exists
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' \
                   AND name=?", (table_name,))
    result = cursor.fetchone()
    if result:
        print("Table already exists.")
        if not recreate_if_exists: 
            cursor.close()
            close_connection(conn)
            return
        else:
            print("'recreate_if_exists' is set to True. Dropping table...")
            cursor.execute(f"DROP TABLE {table_name}")
            print("Table dropped successfully.")
            print("Re-creating table...")
    
    # Create table in the database
    try:
        print(f"Creating table {table_name}...")
        df.to_sql(table_name, conn)
        print("Table created successfully.")
    except Error as e:
        print(e)
    finally:
        cursor.close()
        close_connection(conn)


def fetch_data_from_db(config, db_name, table_name, username, password, 
                       num_samples=None):
    """
    Fetches data from a specified table in the database.

    Args:
        config (dict): The configuration dictionary.
        db_name (str): The name of the database.
        table_name (str): The name of the table.
        username (str): The username for authentication.
        password (str): The password for authentication.
        num_samples (int, optional): The number of samples to fetch. 
                                    Defaults to None.

    Returns:
        pd.DataFrame: The fetched data as a pandas DataFrame.
    """
    conn = create_db_connection(config, db_name, username, authenticate=True, 
                                password=password)
    if conn is None:
        return None

    try:
        query = f"SELECT * FROM {table_name}"
        if num_samples is not None:
            query += f" LIMIT {num_samples}"
        df = pd.read_sql_query(query, conn)
        return df
    except Error as e:
        print(e)
    finally:
        close_connection(conn)


if __name__ == "__main__":
    # Create database tables from parquet files
    cfg_ref = Config()
    config = cfg_ref.get_config()
    
    security_config = load_security_db_config()

    if not os.path.exists(security_config['encryption_key_file_path']):
        write_creds_encryption_key(key='pass1234', 
                            file_path=security_config['encryption_key_file_path'])
    else:
        creds_encryption_key = read_creds_encryption_key(security_config['encryption_key_file_path'])
        
    # -------------------------
    
    for fname in ['business', 'checkin', 'review', 'tip', 'user']:
        file_path = os.path.join(config['data_dir'], fname+'.parquet')
        create_db_table_from_parquet(config, file_path, table_name=fname)
        gc.collect()

    # -------------------------
        
    # for table_name in ['business', 'checkin', 'review', 'tip', 'user']:
    table_name = 'business'
    df = fetch_data_from_db(config, config['data_db_name'], table_name, 
                            config['db_access_uname'], 
                            config['db_access_pwd'], num_samples=5)
    print(df.head())
    gc.collect()

    # -------------------------
   
   

