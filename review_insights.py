

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import pipeline

import os
import gc
import json

torch.set_num_threads(4)
np.set_printoptions(linewidth=200) # default 75

# ----------------------------------------------------

def fetch_data(config, file_name):
    assert file_name is not None

    filepath = os.path.join(config['data_dir'], file_name+'.parquet')

    if verbose: print(f"Reading file: {filepath}")
    
    if file_name == 'business':
        cols_to_read = ['business_id', 'name', 'address', 'city']
        df = pq.read_table(filepath, columns=cols_to_read).to_pandas()

    elif file_name == 'review':
        cols_to_read = ['business_id', 'text']
        df = pq.read_table(filepath, columns=cols_to_read).to_pandas()
    
    if verbose: print("df.shape: ", df.shape)
    gc.collect()

    return df


def get_business_id(config, restanrant_name, city):
    business_df = fetch_data(config, 'business')
    
    business_data = business_df[(restaurants_df['name'] == restanrant_name) & 
                                (restaurants_df['city'] == city)].business_id

    if business_data.shape[0] == 0:
        print(f"No business found for restaurant: {restanrant_name} in city: {city}")
        return None
    elif business_data.shape[0] > 1:
        print(f"Multiple businesses found for restaurant: {restanrant_name} in city: {city}")
        print("Please select one of the following business ids: ")
        print(business_data)

        return None

    id = business_data.id.values[0]
    del business_data
    gc.collect()

    return id


def get_review_chunks(reviews, n_chunks=5, chunk_size=10):
    # Create n chunks of k reviews each - we want n review summaries 
    review_chunks = []
    for i in range(n_chunks):
        # Generate random indices
        random_indices = np.random.choice(reviews.shape[0], 
                    size=min(reviews.shape[0], chunk_size), replace=False)

        selected_texts = reviews.iloc[random_indices]['text'].tolist()

        # Merge all the text rows
        merged_text = ''.join(selected_texts)

        review_chunks.append(merged_text)

    return pd.DataFrame({'text': review_chunks})


def get_sentiment_classifier_pipeline():
    sentiment_classifier_pipeline = pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1, # Force CPU usage (0 = index of first CPU)
            truncation=True,
            max_length=512, # Increase the max length to 512
        )
    
    return sentiment_classifier_pipeline


def get_text_summarizer_pipeline():
    summarizer = pipeline(
            "summarization",
            model="t5-base",
            device=-1,  # Force CPU usage (0 = index of first CPU)
            truncation=True,
            max_length=512,  # Set the maximum output length
        )
    
    return summarizer


def summarize_reviews(df, summarizer):
    reviews = df['text'].tolist()
    
    summaries = summarizer(reviews, truncation=True, max_length=512)

    df['summary'] = [summary['summary_text'] for summary in summaries]
    
    return df


def classify_sentiment(df, classifier_pipeline):
    results = classifier_pipeline(df['text'].tolist())  
    df['sentiment_label'] = [1 if result['label'] == 'POSITIVE' else 0 for result in results]
    return df


def generate_sentiment_labels(df):
    # df = pd.DataFrame({
    #     'text': ['This product is amazing!', 'Terrible customer service.', 'It works okay.']
    # })
    sent_classifier_pipeline = get_sentiment_classifier_pipeline()

    print("Classifying reviews...")
    results = sent_classifier_pipeline(df['text'].tolist())  
    df['sentiment_label'] = [1 if result['label'] == 'POSITIVE' else 0 for result in results]
    
    return df
    

def generate_per_sentiment_summary(df, verbose=False):
    pos_reviews = review_df[review_df['sentiment_label'] == 1]['text'].to_frame()
    neg_reviews = review_df[review_df['sentiment_label'] == 0]['text'].to_frame()

    if verbose:
        print("Number of positive reviews:", len(pos_reviews))
        print("Number of negative reviews:", len(neg_reviews))

    # ----------------------

    n_chunks = 5  # How many review chunks, each of size 'each_chunk_reviews', to summaize
    chunk_size = 25  # Number of reviews in each chunk

    pos_reviews_chunk = get_review_chunks(pos_reviews, n_chunks, chunk_size)
    neg_reviews_chunk = get_review_chunks(neg_reviews, n_chunks, chunk_size)

    # ----------------------

    summarizer_pipeline = get_text_summarizer_pipeline()
    
    if verbose: print("Summarizing positive reviews...")
    pos_review_summaries = summarize_reviews(pos_reviews_chunk, summarizer_pipeline)
    pos_summary = ''.join(pos_review_summaries.summary.values)
    
    if verbose: print("Summarizing negative reviews...")
    neg_review_summaries = summarize_reviews(neg_reviews_chunk, summarizer_pipeline)
    neg_summary = ''.join(neg_review_summaries.summary.values)

    return pos_summary, neg_summary


def generate_review_insights(config, business_id, verbose=False):
    review_df = fetch_data(config, 'review')
    review_df = review_df[review_df['business_id'] == business_id]
    gc.collect()

    review_df = generate_sentiment_labels(config, review_df)

    pos_summary, neg_summary = generate_per_sentiment_summary(df, verbose=verbose)

    print("Positive summary: ", pos_summary)
    print("-"*30)
    print("Negative summary: ", neg_summary)
    


if __name__ == "__main__":
    restanrant_name = 'Willie Mae\'s Scotch House'
    city = 'New Orleans'
    # id = VVH6k9-ycttH3TV_lk5WfQ

    business_id = get_business_id(config, restanrant_name, city)
    if business_id is not None:
        generate_review_insights(config, business_id, verbose=True)

     