
import gc

import numpy as np
import pandas as pd

from CFG import Config
from misc_utils import *
from review_insights_utils import SentimentClassifier, ReviewSummarizer, ActionRecommender

np.set_printoptions(linewidth=200) # default 75


# ----------------------------------------------------
def generate_report(config, curr_bus_data, bus_review_summary_dict: dict, other_review_summary_dict: dict, 
                    actions: str) -> None:
    # generate a pdf report with review summaries and actionable insights
    business_id = curr_bus_data.business_id.values[0]
    business_name = curr_bus_data.name.values[0]
    city = curr_bus_data.city.values[0]
    category = curr_bus_data.categories.values[0]
    
    report = f"Business ID: {business_id}\n\
                Business Name: {business_name}\n\
                City: {city}\n\
                Category: {category}\n\
                \n\
                Business' review summaries: \n\
                    \tPositive: {bus_review_summary_dict['pos_summary']};\n\
                    \tNegative: {bus_review_summary_dict['neg_summary']}.\n\
                Other Business' review summaries: \n\
                    \tPositive: {other_review_summary_dict['pos_summary']};\n\
                    \tNegative: {other_review_summary_dict['neg_summary']}.\n\
                \n\
                Actions: {actions}"
    
    # save report to a pdf file
    report_fname = f'{business_name}_review_insights.pdf'
    report_save_path = os.path.join(config['output_dir'], report_fname)
    with open(report_save_path, 'w') as f:
        f.write(report)
    print("Report saved to: ", report_save_path)
    

def generate_review_insights(config, business_id, verbose=False):
    # fetch user reviews for the restaurant
    review_df = fetch_data(config, 'review')
    
    curr_bus_review_df = review_df[review_df['business_id'] == business_id]
    gc.collect()

    # --------------------------------------------------------------
     # keep only manageable number of reviews
    n_max_reviews = 50
    if curr_bus_review_df.shape[0] > n_max_reviews:
        curr_bus_review_df = curr_bus_review_df.sample(n=n_max_reviews, random_state=42)
    print("After filtering to keep less reviews, if needed: ", curr_bus_review_df.shape)
    
    
    # user reviews sentiment classification
    sc = SentimentClassifier(config)
    print("Generating sentiment labels for current business' reviews...")
    curr_bus_review_df = sc.generate_sentiment_labels(curr_bus_review_df, verbose)
    # curr_bus_review_df['sentiment_label'] = np.random.choice([1, 0], size=curr_bus_review_df.shape[0])
    print("curr_bus_review_df.shape: ", curr_bus_review_df.shape)
    
   
    gc.collect()
    # --------------------------------------------------------------
    
    # user reviews summary - positive and negative
    rs = ReviewSummarizer(config)
    print("Generating review summaries for current business...")
    bus_review_summary_dict = rs.generate_per_sentiment_summary(curr_bus_review_df, verbose=verbose)
    # bus_review_summary_dict = {'pos_summary': 'dummy summary', 'neg_summary': 'dummy summary'}
    print("Positive summary: ", bus_review_summary_dict['pos_summary'])
    print("-"*30)
    print("Negative summary: ", bus_review_summary_dict['neg_summary'])
    
    gc.collect()
    # --------------------------------------------------------------
    
    print("Fetching business data for other restaurants in the city...")
    # actionable insights based on user reviews
    business_df = fetch_data(config, 'business', verbose)
    print("business_df.shape: ", business_df.shape)
    
    curr_bus_data = business_df[business_df['business_id'] == business_id]
    curr_bus_categories = curr_bus_data.categories.values[0].split(', ')
    curr_bus_postal_code = curr_bus_data.postal_code.values[0]
    
    
    # Fetch business data for other restaurants in the city
    # city_bus_data = business_df[business_df['city'] == curr_bus_data.city.values[0]]
    print("1. Before filtering on zip code: ", business_df.shape)
    neigh_bus_data = business_df[business_df['postal_code'] == curr_bus_postal_code]
    print("2. After filtering on zip code: ", neigh_bus_data.shape)
    
    if neigh_bus_data.shape[0] == 0:
        print("No other restaurants found in the same zip code, looking for restaurants in the same city...")
        neigh_bus_data = business_df[business_df['city'] == curr_bus_data.city.values[0]]
        print("2. After filtering on zip code: ", neigh_bus_data.shape)
    
    if curr_bus_categories is not None and len(curr_bus_categories) > 0:
        neigh_bus_data = neigh_bus_data[neigh_bus_data['categories'].apply(lambda x: len(x) > 0)]
        neigh_bus_data = neigh_bus_data[neigh_bus_data['categories'].apply(lambda x: any(category in x for category in curr_bus_categories))]
    
    print("After keeping same category businesses: ", neigh_bus_data.shape)
    
    # remove the current business
    neigh_bus_data = neigh_bus_data[neigh_bus_data['business_id'] != business_id]
    print("After filtering out current business: ", neigh_bus_data.shape)
    
    gc.collect()
    
    #  (604, 5)
    # print("neigh_bus_data.shape: ", neigh_bus_data.shape)
    
    # ----
    
    # Fetch reviews for other restaurants in the city
    neigh_review_df = review_df[review_df['business_id'].isin(neigh_bus_data.business_id)]
    print("city_review_df.shape: ", neigh_review_df.shape)
    gc.collect()
    
    # keep only manageable number of reviews
    n_max_reviews = 50
    if neigh_review_df.shape[0] > n_max_reviews:
        neigh_review_df = neigh_review_df.sample(n=n_max_reviews, random_state=42)
    
    # raise
    
    # sentiment classification for other restaurants in the city
    print("Generating sentiment labels for other restaurants in the city...")
    neigh_review_df = sc.generate_sentiment_labels(neigh_review_df)
    print('neigh_review_df["sentiment_label"].value_counts(): ', neigh_review_df['sentiment_label'].value_counts())
    
    # summary of reviews for other restaurants in the city
    neigh_review_summary_dict = rs.generate_per_sentiment_summary(neigh_review_df, verbose=verbose)
    print("Neighbourhood Positive summary: ", neigh_review_summary_dict['pos_summary'])
    print("-"*30)
    print("Neighbourhood Negative summary: ", neigh_review_summary_dict['neg_summary'])
    # neigh_review_summary_dict = {'pos_summary': 'this is a good restaurant', 'neg_summary': 'this is a bad restaurant'}
    
    gc.collect()
    
    # ----------
    # raise
    
    print("Generating actionable insights...")
    # actionable insights based on other restaurants in the city
    ar = ActionRecommender(config)
    actions = ar.generate_actions(bus_review_summary_dict, neigh_review_summary_dict, 
                                  curr_bus_data, verbose=True)
    with open('actions.txt', 'w', encoding='utf-8') as f:
        f.write(actions)
    # print("Actions: ", actions)
    
    raise

    # generate a pdf report with review summaries and actionable insights
    generate_report(config, curr_bus_data, bus_review_summary_dict, 
                      neigh_review_summary_dict, actions)


if __name__ == "__main__":
    restanrant_name = 'Willie Mae\'s Scotch House'
    city = 'New Orleans'
    # id = VVH6k9-ycttH3TV_lk5WfQ

    config = Config()
    business_id = get_business_id(config, restanrant_name, city)
    
    if business_id is not None:
        generate_review_insights(config, business_id, verbose=True)
    
    

     