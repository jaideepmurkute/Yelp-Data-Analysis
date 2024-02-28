"""
This module contains functions for interacting with a SQLite database, 
including authentication, data retrieval, and table creation.

Author: Jaideep Murkute
Date: 2024-02-28

"""

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


def create_connection():
    """
    Create a connection to the SQLite database.

    Returns:
    conn: SQLite connection object
    """
    conn = None;
    try:
        conn = sqlite3.connect('your_database.db') # use your own database name
        return conn
    except Error as e:
        print(e)


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


def fetch_user_data(username, decryption_key):
    """
    Fetches the password hash and salt for a given username from the database.
    
    Args:
        username (str): The username of the user.
        decryption_key (bytes): The decryption key used to decrypt the password hash.
    
    Returns:
        tuple: A tuple containing the decrypted password hash and salt. If the username is not found, returns (None, None).
    """
    conn = create_connection()
    if conn is None:
        return None, None

    cursor = conn.cursor()
    try:
        cursor.execute("SELECT password_hash, salt FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
        if result:
            encrypted_password_hash, salt = result
            f = Fernet(decryption_key)
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


def write_encryption_key_to_file(key, file_path):
    """
    Write the encryption key to a file.

    Args:
        key (str): The encryption key to be written.
        file_path (str): The path to the file where the key will be written.

    Returns:
        None
    """
    with open(file_path, 'wb') as file:
        file.write(key.encode())


def read_encryption_key_from_file(file_path):
    """
    Reads the encryption key from the specified file.

    Args:
        file_path (str): The path to the file containing the encryption key.

    Returns:
        bytes: The encryption key read from the file.
    """
    with open(file_path, 'rb') as file:
        encryption_key = file.read()
    return encryption_key


def authenticate_user(db_security_config, username, entered_password):
    """
    Authenticates a user by comparing the entered password with the stored password hash.

    Args:
        db_security_config (dict): A dictionary containing database security configuration.
        username (str): The username of the user.
        entered_password (str): The password entered by the user.

    Returns:
        bool: True if the entered password matches the stored password hash, False otherwise.
    """
    # 1. Retrieve stored credentials - fetch_user_data
    stored_password_hash, salt = fetch_user_data(username, b'your_encryption_key')
    
    encryption_key = read_encryption_key_from_file(db_security_config['encryption_key_file_path'])

    stored_password_hash, salt = fetch_user_data(username, encryption_key)

    # 2. Hash the entered password using the same salt
    entered_password_hash = hash_password(entered_password, salt) 
    
    # 3. Compare hashes (constant-time comparison) and return authentication result
    result = compare_hashes(stored_password_hash, entered_password_hash)

    return result


def load_db_security_config():
    """
    Load the database security configuration.

    Returns:
        dict: The database security configuration.
    """
    security_config = SecurityConfig()
    db_security_config = security_config.get_security_config()

    return db_security_config


def connect_to_db(db_name, username, password):
    """
    Connects to the database using the provided username and password.

    Args:
        username (str): The username for authentication.
        password (str): The password for authentication.

    Returns:
        conn (sqlite3.Connection): The connection object to the database if authentication is successful,
                                   None otherwise.
    """
    db_security_config = load_db_security_config()
    if authenticate_user(db_security_config, username, password):
        conn = sqlite3.connect(f'{db_name}.db')
        return conn
    else:
        print("Authentication failed.")
        return None


def create_db_table_from_parquet(config, file_path, table_name):
    """
    Reads a parquet data file, converts it to a pandas dataframe, and inserts it into a database table.

    Args:
        file_path (str): The path to the parquet data file.
        table_name (str): The name of the table to be created in the database.
        username (str): The username for authentication.
        password (str): The password for authentication.

    Returns:
        None
    """
    # Read parquet file into pandas dataframe
    # df = pd.read_parquet(file_path)
    df = pq.read_table(file_path).to_pandas()

    # Connect to the database
    conn = connect_to_db(config['db_name'], config['db_access_uname'], 
                         config['db_access_pwd'])
    if conn is None:
        return

    # Check if table already exists
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' \
                   AND name=?", (table_name,))
    result = cursor.fetchone()
    if result:
        print("Table already exists.")
        cursor.close()
        close_connection(conn)
        return

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


if __name__ == "__main__":
    # Create database tables from parquet files
    cfg_ref = Config()
    config = cfg_ref.get_config()
    for fname in ['business', 'checkin', 'review', 'tip', 'user']:
        file_path = os.path.join(config['data_dir'], fname+'.parquet')
        create_db_table_from_parquet(config, file_path, table_name=fname)




