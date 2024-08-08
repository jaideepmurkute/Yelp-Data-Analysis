
import os
import gc
import pyarrow.parquet as pq
# import fastparquet as fp


def fetch_data(config, file_name, verbose=False):
    assert file_name is not None

    filepath = os.path.join(config.config_dict['data_dir'], 'yelp_academic_dataset_'+file_name+'.parquet')

    if verbose: print(f"Reading file: {filepath}")
    
    if file_name == 'business':
        cols_to_read = ['business_id', 'name', 'address', 'city', 'postal_code', 'categories']
        df = pq.read_table(filepath, columns=cols_to_read).to_pandas()  # use fastparquet instead???

    elif file_name == 'review':
        cols_to_read = ['business_id', 'text']
        df = pq.read_table(filepath, columns=cols_to_read).to_pandas()
    
    if verbose: print("df.shape: ", df.shape)
    gc.collect()

    return df


def get_business_id(config, restanrant_name, city):
    business_df = fetch_data(config, 'business')
    
    business_data = business_df[(business_df['name'] == restanrant_name) & 
                                (business_df['city'] == city)].business_id

    if business_data.shape[0] == 0:
        print(f"No business found for restaurant: {restanrant_name} in city: {city}")
        return None
    elif business_data.shape[0] > 1:
        print(f"Multiple businesses found for restaurant: {restanrant_name} in city: {city}")
        print("Please select one of the following business ids: ")
        print(business_data)

        return None

    # print('business_data: ', business_data)
    id = business_data.values[0]
    del business_data
    gc.collect()

    return id
