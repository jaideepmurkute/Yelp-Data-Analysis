
class Config:
    def __init__(self):
        self.config_dict = {
            'data_dir': 'C:\\Users\\Vitthal\\Desktop\\projects\\yelp_data',
            
            'data_db_storage_dir': 'C:\\Users\\Vitthal\\Desktop\\projects\\yelp_data\\db_storage',
            'data_db_name': 'yelp_data',
            'data_db_access_uname': 'db_admin_1', 
            'data_db_access_pwd': 'pass1234',
            'create_db_if_not_exists': True, # for data_db
        }
    
    def get_config(self):
        return self.config_dict


