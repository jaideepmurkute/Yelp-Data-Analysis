import gc
import os

import json
import pandas as pd
import pyarrow.parquet as pq
import sqlite3
from sqlite3 import Error

from CFG import Config

# ----------------------------

class DatabaseDDLManager:
    '''
        Database Data Definition Language (DDL) Manager
            Create, Drop, Alter, Truncate, Rename, Comment, etc.
    '''
    def __init__(self, config):
        self.db_name = config['db_name']
        self.db_storage_dir = config['db_storage_dir']
        if not os.path.exists(self.db_storage_dir):
            os.makedirs(self.db_storage_dir)
        self.db_path = os.path.join(self.db_storage_dir, self.db_name)
        
        
    def open_connection(self):
        """
        Opens a connection to the database.

        Args:
            db_name (str): The name of the database to connect to.
                Or None if failed.

        Returns:
            None
        """
        try:
            self.conn = sqlite3.connect(self.db_path)
            return self.conn
        except Error as e:
            print(e)
        
        return None

    def close_connection(self):
        """
        Closes the connection to the database.
        """
        self.conn.close()
    
    def get_table_names(self):
        '''
        Retrieves the names of all tables in the database.

        Returns:
            list: A list of table names.
        '''
        table_names = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()

                table_names = [table[0] for table in tables]
        except Error as e:
            print(e)
            raise e
            
        return table_names

    
    def create_table(self, create_table_sql, from_parquet=False, \
                     parquet_file_path=None, table_name=None, exclude_columns=None, 
                     date_cols=None,
                     dtype_conversion_map=None, overwrite=False, verbose=False):
        """
        Creates a table in the database using the provided SQL statement.

        Args:
            create_table_sql (str): The SQL statement to create the table.
            Example:
                create_table_sql = 
                    CREATE TABLE IF NOT EXISTS employees (
                        id integer PRIMARY KEY,
                        name text NOT NULL,
                        salary real,
                        department_id integer,
                        FOREIGN KEY (department_id) REFERENCES departments (id)
                    );
                    
        Raises:
            Error: If there is an error executing the SQL statement.

        Returns:
            Boolean: True if the table was created successfully, False otherwise.
        """
        if table_name in self.get_table_names() and not overwrite:
            print(f"Table {table_name} already exists in the database")
            print("Either pass overwrite=True or use a different table name \
                  to create a new table. Or check your input.")
            return False

        if from_parquet:
            # Read the Parquet file into a DataFrame
            if verbose: print("Reading Parquet file into DataFrame")
            df = pd.read_parquet(parquet_file_path)
            
            # --------------------

            if verbose: print("Sanitizing DataFrame")
            
            # Drop the columns that are not required or are not supported by the database
            if exclude_columns is not None:
                df.drop(columns=exclude_columns, inplace=True)
            
            # Convert the data types of the columns to make them compatible with SQLite
            if dtype_conversion_map is not None: 
                for col, dtype in dtype_conversion_map.items():
                    if col in df.columns:
                        df[col] = df[col].astype(dtype)
                    else:
                        print(f"Column '{col}' not found in the DataFrame. Skipping data type conversion!!!")
            
            # Convert the date columns to datetime objects for SQLite compatibility
            if date_cols is not None:
                for col in date_cols:
                    df[col] = pd.to_datetime(df[col], format='%Y-%m-%d',  errors='coerce')
            
            # --------------------
            
            if verbose: print("Creating database table...")
            with sqlite3.connect(self.db_path) as conn:
                # Create the SQL table from the DataFrame
                df.to_sql(table_name, conn, if_exists='replace', index=False)
            
            if verbose: print(f"Table {table_name} created successfully!")
            
            del df
            gc.collect()
            return True
        else:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    c = conn.cursor()
                    c.execute(create_table_sql)
                print("Table created successfully")
                return True
            except Error as e:
                print(e)
            
        return False
    
    def drop_table(self, table_name):
        """
        Drops a table from the database.

        Args:
            table_name (str): The name of the table to drop.

        Raises:
            Error: If there is an error executing the SQL statement.

        Returns:
            Boolean: True if the table was dropped successfully, False otherwise.
        """
        try:
            c = self.conn.cursor()
            c.execute(f"DROP TABLE IF EXISTS {table_name}")
            return True
        except Error as e:
            print(e)
        
        return False

    def alter_table(self, table_name, alter_table_sql):
        """
        Alters a table in the database using the provided SQL statement.

        Args:
            table_name (str): The name of the table to alter.
            alter_table_sql (str): The SQL statement to alter the table.
            Example:
                alter_table_sql = 
                    ALTER TABLE employees
                    ADD COLUMN email text;
                    
        Raises:
            Error: If there is an error executing the SQL statement.

        Returns:
            Boolean: True if the table was altered successfully, False otherwise.
        """
        try:
            c = self.conn.cursor()
            c.execute(alter_table_sql)
            return True
        except Error as e:
            print(e)
        
        return False

    def truncate_table(self, table_name):
        """
        Truncates a table in the database.

        Args:
            table_name (str): The name of the table to truncate.

        Raises:
            Error: If there is an error executing the SQL statement.

        Returns:
            Boolean: True if the table was truncated successfully, False otherwise.
        """
        try:
            c = self.conn.cursor()
            c.execute(f"DELETE FROM {table_name}")
            return True
        except Error as e:
            print(e)
        
        return False

    def rename_table(self, table_name, new_table_name):
        """
        Renames a table in the database.

        Args:
            table_name (str): The name of the table to rename.
            new_table_name (str): The new name for the table.

        Raises:
            Error: If there is an error executing the SQL statement.

        Returns:
            Boolean: True if the table was renamed successfully, False otherwise.
        """
        try:
            c = self.conn.cursor()
            c.execute(f"ALTER TABLE {table_name} RENAME TO {new_table_name}")
            return True
        except Error as e:
            print(e)
        
        return False

    def comment_table(self, table_name, comment):
        """
        Comments a table in the database.

        Args:
            table_name (str): The name of the table to comment.
            comment (str): The comment for the table.

        Raises:
            Error: If there is an error executing the SQL statement.

        Returns:
            Boolean: True if the table was commented successfully, False otherwise.
        """
        try:
            c = self.conn.cursor()
            c.execute(f"COMMENT ON TABLE {table_name} IS {comment}")
            return True
        except Error as e:
            print(e)
        
        return False
    
    def execute_query(self, query_str):
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                cursor.execute(query_str)
                result = cursor.fetchall()
                column_names = [desc[0] for desc in cursor.description]
                return [column_names] + list(result)
        except Error as e:
            print("Error Message: ", e)
        
        return None



def create_table(config, data_preprocessing_config, table_name):
    try:
        print("Creating table - ", table_name)
        ddl_man.create_table(create_table_sql='', 
                            from_parquet=True, 
                            parquet_file_path=config['file_path'], 
                            table_name=table_name, 
                            exclude_columns=data_preprocessing_config['drop_cols'], 
                            dtype_conversion_map=data_preprocessing_config['dtype_conversion_map'], 
                            overwrite=False,
                            verbose=False)
        print(f"Table - {table_name} - created successfully!")
        return True
    except Error as e:
        print("Error creating table:", e)
        raise e
    
    return False


# ----------------------------

'''
        This map defines the data types of the columns in the tables which need to
    be manually converted to be supported by the sqllite database.
    
        The 'exclude_columns' key is used to exclude columns from the dataframe
    which are are either irrelevant or are not supported by the database.
    
        The 'date_cols' key is used to specify the columns which need to be
    converted to datetime objects, before being stored in the database.
'''
table_creation_config_map = {
        'business': {
            'dtype_conversion_map': {
                'business_id': 'str',
                'name': 'str',
                'address': 'str',
                'city': 'str',
                'state': 'str',
                'postal_code': 'str',
                'latitude': 'float32',
                'longitude': 'float32',
                'stars': 'uint8',
                'review_count': 'uint32',
                'is_open': 'uint8',
                'categories': 'str',
                'hours': 'str'
                 },
            'exclude_columns': ['attributes'],
            'date_cols': [],
        },
        'user': {
                'dtype_conversion_map': {
                    'user_id': 'str', 
                    'name': 'str', 
                }, 
                'exclude_columns': ['elite', 'friends'], 
                'date_cols': ['yelping_since'],
        },
        'review': {
                'dtype_conversion_map': {
                    'review_id': 'str', 
                    'user_id': 'str', 
                    'business_id': 'str', 
                    'text': 'str', 
                    'date': 'str', 
                }, 
                'exclude_columns': [], 
                'date_cols': ['date'],
        },
        'tip': {
                'dtype_conversion_map': {
                    'user_id': 'str', 
                    'business_id': 'str', 
                    'text': 'str', 
                    'compliment_count': 'int'
                }, 
                'exclude_columns': [], 
                'date_cols': ['date'],
        },
        'checkin': {
                'dtype_conversion_map': {
                    'business_id': 'str', 
                    
                    # This 'date' column has a enumeration of date-time of checkins, 
                    # not a single date-time
                    'date': 'str', 
                }, 
                'exclude_columns': [], 
                'date_cols': [],
        },
    }


if __name__ == "__main__":
    cfg = Config()
    config = cfg.get_config()
    data_preprocessing_config = cfg.get_data_preprocessing_config()
        
    ddl_man = DatabaseDDLManager(config)

    for table_name in ['business', 'user', 'review', 'tip', 'checkin']: 
        fname = 'yelp_academic_dataset_' + table_name + '.parquet'
        config['file_path'] = os.path.join(config['data_dir'], fname)

        create_table(config, data_preprocessing_config[table_name], table_name=table_name)
        
        print('-'*50)

    print("All tables created successfully!")
