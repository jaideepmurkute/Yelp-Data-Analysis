
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

        self.data_preprocessing_config = file_preprocessing_config_map = {
                'business': {
                        'drop_cols': ['attributes', 'hours'], 
                        'encode_dates': True, 
                        'date_cols': ['date'],
                        'id_cols': ['business_id'],
                        'handle_outliers': False,
                        'scale_data': False,
                        'cols_scaler_type_map': None,
                    }, 
                'user': {
                        'drop_cols': ['friends'], 
                        'encode_dates': False, 
                        'date_cols': None,
                        'id_cols': None, 
                        'handle_outliers': False,
                        'scale_data': False,
                        'cols_scaler_type_map': None,
                    }, 
                'review': {}, 
                'tip': {
                        'drop_cols': None, 
                        'encode_dates': True, 
                        'date_cols': ['date'],
                        'id_cols': None,
                        'handle_outliers': False,
                        'scale_data': False,
                        'cols_scaler_type_map': None,
                    }, 
                'checkin': {
                        'drop_cols': None, 
                        'encode_dates': False, 
                        'date_cols': None,
                        'id_cols': None,
                        'handle_outliers': False,
                        'scale_data': False,
                        'cols_scaler_type_map': None,
                    }, 
            }

    def get_config(self):
        return self.config_dict
    
    def get_data_preprocessing_config(self):
        return self.data_preprocessing_config


