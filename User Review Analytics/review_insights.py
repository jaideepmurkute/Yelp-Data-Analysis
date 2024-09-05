
'''
    This script generates actionable insights based on user reviews for a restaurant.

    __author__ = ''
    __email__ = ''
    __date__ = ''
    __version__ = ''
'''

import gc
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
np.set_printoptions(linewidth=200) # default 75

from CFG import Config
from misc_utils import *
from review_insights_utils import SentimentClassifier, ReviewSummarizer, ActionRecommender
from generate_pdf_report import generate_pdf


def fetch_neigh_review_data(config: Dict, business_id: str, business_df: pd.DataFrame, review_df: pd.DataFrame, 
                            n_max_reviews: Optional[int]=50, verbose: Optional[bool]=False) -> pd.DataFrame:
    '''
        Fetches user reviews for other restaurants in the zip-code/city; from the same category.
        Args:
            business_id: str, business_id of the restaurant
            business_df: pd.DataFrame, business data
            review_df: pd.DataFrame, review data
            n_max_reviews: int, maximum number of reviews to fetch for each restaurant
            verbose: bool, print debug information
        Returns:
            neigh_review_df: pd.DataFrame, user reviews for other restaurants in the city
    '''
    if verbose: print("Fetching business data for other restaurants in the city...")
    curr_bus_data = business_df[business_df['business_id'] == business_id]
    curr_bus_categories = curr_bus_data.categories.values[0].split(', ')
    curr_bus_postal_code = curr_bus_data.postal_code.values[0]
    
    # Fetch business data for other restaurants in the city
    if verbose: print("Before filtering on zip code: ", business_df.shape)
    neigh_bus_data = business_df[business_df['postal_code'] == curr_bus_postal_code]
    if verbose and neigh_bus_data.shape[0] > 0: print("After filtering on zip code: ", neigh_bus_data.shape)
    
    if neigh_bus_data.shape[0] == 0:
        if verbose: print("No other restaurants found in the same zip code, looking for restaurants in the same city...")
        neigh_bus_data = business_df[business_df['city'] == curr_bus_data.city.values[0]]
        if verbose: print("After filtering on zip code: ", neigh_bus_data.shape)
    
    if curr_bus_categories is not None and len(curr_bus_categories) > 0:
        neigh_bus_data = neigh_bus_data[neigh_bus_data['categories'].apply(lambda x: len(x) > 0)]
        neigh_bus_data = neigh_bus_data[neigh_bus_data['categories'].apply(lambda x: any(category in x for category in curr_bus_categories))]
    
    if verbose: print("After keeping same category businesses: ", neigh_bus_data.shape)
    
    # Remove the current business
    neigh_bus_data = neigh_bus_data[neigh_bus_data['business_id'] != business_id]
    if verbose: print("After filtering out current business: ", neigh_bus_data.shape)
    gc.collect()
    
    # Fetch reviews for other restaurants in the city
    neigh_review_df = review_df[review_df['business_id'].isin(neigh_bus_data.business_id)]
    if verbose: print("city_review_df.shape: ", neigh_review_df.shape)
    gc.collect()
    
    # keep only manageable number of reviews
    if neigh_review_df.shape[0] > n_max_reviews:
        neigh_review_df = neigh_review_df.sample(n=n_max_reviews, random_state=42)
    
    return neigh_review_df
    
    
def save_actions(config: Dict, actions: str) -> str:
    # Save the generated actions string to a text file
    actions_save_path = os.path.join(config.config_dict['output_dir'], 'actions.txt')
    with open(actions_save_path, 'w', encoding='utf-8') as f:
        f.write(actions)
    print("Actions saved to: ", actions_save_path)
    return actions_save_path


def get_restaurant_category(curr_bus_data: pd.DataFrame) -> str:
    # Returns a valid category for the restaurant
    restaurant_category = None
    try:
        restaurant_category = curr_bus_data.categories.values[0]
    except:
        pass
    if restaurant_category is None or len(restaurant_category) == 0:
        restaurant_category = 'Restaurant'
    
    return restaurant_category


def generate_review_insights(config: Dict, business_id: str, restaurant_name: str, city: str, 
                             verbose: Optional[bool]=False) -> None:
    '''
        Generate actionable insights based on user reviews for a restaurant
        Code first fetches user reviews for the restaurant and then generates sentiment labels for the reviews.
        Then for each sentiment, it generates a summary of the reviews.
        
        Same process is repeated for other restaurants in the neighbourhood and from the same cateogry.
        
        Finally, it generates actionable insights based on the reviews for the restaurant and other restaurants.
        '''
    # fetch user reviews for the restaurant
    review_df = fetch_data(config, 'review')
    curr_bus_review_df = review_df[review_df['business_id'] == business_id]
    
    business_df = fetch_data(config, 'business', verbose)
    curr_bus_data = business_df[business_df['business_id'] == business_id]
    
    neigh_review_df = fetch_neigh_review_data(config, business_id, business_df, review_df, verbose=verbose)
    
    gc.collect()

    # --------------------------------------------------------------
    # keep only manageable number of reviews
    n_max_reviews = 25
    if curr_bus_review_df.shape[0] > n_max_reviews:
        curr_bus_review_df = curr_bus_review_df.sample(n=n_max_reviews, random_state=42)
    if verbose: print("After filtering to keep less reviews, if needed: ", curr_bus_review_df.shape)
    # --------------------------------------------------------------
    # User reviews sentiment classification
    
    sc = SentimentClassifier(config)
    if verbose: print("Generating sentiment labels for current business' reviews...")
    curr_bus_review_df = sc.generate_sentiment_labels(curr_bus_review_df, verbose)
    # curr_bus_review_df['sentiment_label'] = np.random.choice([1, 0], size=curr_bus_review_df.shape[0])
    gc.collect()
    
    # --------------------------------------------------------------
    
    # user reviews summary - positive and negative
    rs = ReviewSummarizer(config)
    if verbose: print("Generating review summaries for current business...")
    bus_review_summary_dict = rs.generate_per_sentiment_summary(curr_bus_review_df, verbose=verbose)
    # bus_review_summary_dict = {'pos_summary': 'dummy summary', 'neg_summary': 'dummy summary'}
    if verbose:
        print("Positive summary: ", bus_review_summary_dict['pos_summary'])
        print("-"*30)
        print("Negative summary: ", bus_review_summary_dict['neg_summary'])
        print("-"*30)
    gc.collect()
    
    # --------------------------------------------------------------
    
    
    # keep only manageable number of reviews
    n_max_reviews = 25
    if neigh_review_df.shape[0] > n_max_reviews:
        neigh_review_df = neigh_review_df.sample(n=n_max_reviews, random_state=42)
    if verbose: print("After filtering to keep less reviews, if needed: ", neigh_review_df.shape)
    # --------------------------------------------------------------
    
    # Sentiment classification for other restaurants in the city
    if verbose: print("Generating sentiment labels for other restaurants in the city...")
    neigh_review_df = sc.generate_sentiment_labels(neigh_review_df)
    # neigh_review_df['sentiment_label'] = np.random.choice([1, 0], size=neigh_review_df.shape[0])
    
    if verbose: print('neigh_review_df["sentiment_label"].value_counts(): ', neigh_review_df['sentiment_label'].value_counts())
    
    # --------------------------------------------------------------
    
    # Generate summary of reviews for other restaurants in the city in the neighbourhood and same category
    neigh_review_summary_dict = rs.generate_per_sentiment_summary(neigh_review_df, verbose=verbose)
    # neigh_review_summary_dict = {'pos_summary': 'this is a good restaurant', 'neg_summary': 'this is a bad restaurant'}
    if verbose:
        print("Neighbourhood Positive summary: ", neigh_review_summary_dict['pos_summary'])
        print("-"*30)
        print("Neighbourhood Negative summary: ", neigh_review_summary_dict['neg_summary'])
        
    gc.collect()
    
    # --------------------------------------------------------------
    # Generate actionable insights based on other restaurants in the city
    
    print("Generating actionable insights...")
    ar = ActionRecommender(config)
    actions = ar.generate_actions(bus_review_summary_dict, neigh_review_summary_dict, 
                                  curr_bus_data, verbose=True)
    actions_save_path = save_actions(config, actions)
    
    # --------------------------------------------------------------
    # Generate a pdf report with review summaries and actionable insights
    
    print("Generating PDF report...")
    restaurant_category = get_restaurant_category(curr_bus_data)
    generate_pdf(bus_id=business_id, 
                bus_name=restaurant_name, 
                city=city, 
                category=restaurant_category, # curr_bus_data.categories.values[0], 
                bus_pos_summary=bus_review_summary_dict['pos_summary'], 
                bus_neg_summary=bus_review_summary_dict['neg_summary'], 
                neigh_pos_summary=neigh_review_summary_dict['pos_summary'], 
                neigh_neg_summary=neigh_review_summary_dict['neg_summary'], 
                actions_file_path=actions_save_path)

    

if __name__ == "__main__":
    restaurant_name = 'Willie Mae\'s Scotch House'
    city = 'New Orleans'
    # id = VVH6k9-ycttH3TV_lk5WfQ

    config = Config()
    business_id = get_business_id(config, restaurant_name, city)
    
    if business_id is not None:
        generate_review_insights(config, business_id, restaurant_name, city, verbose=True)
    
    

     