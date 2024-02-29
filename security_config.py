
class SecurityConfig:
    def __init__(self):
        self.security_db_config = {
            'db_name': 'yelp_security_db',
            'credentials_table_name': 'user_credentials',
            'encryption_key_file_path': 'C:\\Users\\Vitthal\\Desktop\\projects\\yelp_data\\security\\encryption_key.bin'
        }
    
    def get_config(self):
        return self.db_security_config
    



