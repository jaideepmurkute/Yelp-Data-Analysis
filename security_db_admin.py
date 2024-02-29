import sqlite3
import hashlib
import csv

from CFG import Config
from security_config import SecurityConfig


class DatabaseManager:
    def __init__(self, db_name):
        self.db_name = db_name
        self.conn = None
        self.cursor = None

    def connect(self):
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()

    def create_table(self, table_name):
        self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                user_name TEXT PRIMARY KEY,
                password TEXT
            )
        ''')

    def check_empty_table(self):
        self.cursor.execute('SELECT COUNT(*) FROM user_credentials')
        result = self.cursor.fetchone()
        return result[0] == 0

    def create_user(self, username, password):
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        self.cursor.execute('SELECT COUNT(*) FROM user_credentials WHERE \
                            user_name = ?', (username,))
        result = self.cursor.fetchone()
        if result[0] == 0:
            self.cursor.execute('INSERT INTO user_credentials (user_name, password) \
                                VALUES (?, ?)', (username, hashed_password))
        self.conn.commit()
    
    def delete_user(self, username):
        self.cursor.execute('DELETE FROM user_credentials WHERE user_name = ?', (username,))
        self.conn.commit()
        return self.cursor.rowcount > 0
    
    def export_db_to_csv(self, table_name, file_name):
        self.cursor.execute(f'SELECT * FROM {table_name}')
        rows = self.cursor.fetchall()

        with open(file_name, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([description[0] for description in 
                                 self.cursor.description])
            csv_writer.writerows(rows)
    
    def close(self):
        self.conn.close()


def load_security_db_config():
    """
    Load the database security configuration.

    Returns:
        dict: The database security configuration.
    """
    security_config = SecurityConfig()
    security_db_config = security_config.get_config()

    return security_db_config

def main(security_config):
    
    db_manager = DatabaseManager(f'{security_config["db_name"]}.db')
    db_manager.connect()
    db_manager.create_table(table_name='user_credentials')


    if db_manager.check_empty_table():
        db_manager.create_user('db_admin_1', 'pass1234')

    db_manager.close()


if __name__ == '__main__':
    security_config = load_security_db_config()
    main(security_config)
