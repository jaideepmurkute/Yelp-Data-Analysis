
import os

class Config:
    def __init__(self):
        self.config_dict = {
                'data_dir': os.path.join('data'),
                
                'db_storage_dir': os.path.join('data', 'db_storage'),
                'db_name': 'yelp_data',
                
                'data_db_access_uname': 'db_admin_1', 
                'data_db_access_pwd': 'pass1234',
                'create_db_if_not_exists': True, # for data_db
                
                'force_extract_data': True,
                
                'openai_api_key_path': os.path.join('..', 'openai_api_key.txt'),
            }

        self.data_preprocessing_config = {
                'business': {
                        'id_cols': ['business_id'],
                        'date_cols': ['date'],
                        'encode_dates': True, 
                        'drop_cols': ['attributes', 'hours'], 
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
                    }, 
                'user': {
                        'id_cols': None, 
                        'date_cols': ['yelping_since'],
                        'encode_dates': False, 
                        'drop_cols': ['elite', 'friends'], 
                        'dtype_conversion_map': {'user_id': 'str', 
                                                'name': 'str',
                                            }, 
                    },
                'review': {
                        'id_cols': None,
                        'date_cols': ['date'],
                        'encode_dates': True, 
                        'drop_cols': None, 
                        'dtype_conversion_map': {'review_id': 'str', 
                                                'user_id': 'str', 
                                                'business_id': 'str', 
                                                'text': 'str', 
                                                'date': 'str', 
                                                }
                    }, 
                'tip': {
                        'id_cols': None,
                        'date_cols': ['date'],
                        'encode_dates': True, 
                        'drop_cols': None, 
                        'dtype_conversion_map': {'user_id': 'str', 
                                'business_id': 'str', 
                                'text': 'str', 
                                'compliment_count': 'int'
                                }
                    }, 
                'checkin': {
                        'id_cols': None,
                        'drop_cols': None, 
                        'date_cols': ['date'],
                        'encode_dates': False, 
                        'dtype_conversion_map': {'business_id': 'str'}
                    }, 
            }

    def get_config(self):
        return self.config_dict
    
    def get_data_preprocessing_config(self):
        return self.data_preprocessing_config


