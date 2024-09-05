
'''
    Utility functions for the User Review Analytics project.
    Contains classes for Sentiment Classification, Review Summarization and Action Recommendation.

    __author__ = ''
    __email__ = ''
    __date__ = ''
    __version__ = ''
'''
from typing import Optional, Dict, List, Any

from openai import OpenAI
import pandas as pd
import numpy as np
import torch
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel
# torch.set_num_threads(4)


class SentimentClassifier:
    """
        Performs sentiment classification of the user reviews - positive(1) or negative(0).
    
        Attributes:
            config: Configuration object containing the necessary parameters.
        
        Methods:
            analyze: Classifies the sentiment of the reviews.
                Expects a column 'text' in the input DataFrame containing reviews.
        
        Returns:
            DataFrame with an additional column 'sentiment_label' containing the sentiment label.
        
        Example:
            df = pd.DataFrame({'text': ['This product is amazing!', 
                                        'Terrible customer service.', 
                                        'It works okay.']})
            classifier = SentimentClassifier(config)
            df = classifier.analyze(df)
            
            print(df)
            # Output:
            #        text                         sentiment_label
            # 0  This product is amazing!                1 (i.e. Positive)
            # 1  Terrible customer service.              0 (i.e. Negative)
            # 2  It works okay.                          1 (i.e. Positive)
    """
    def __init__(self, config: dict) -> None:
        self.sentiment_classifier_model_name = config.config_dict['sentiment_classifier_model_name']
        self.pipeline = self.get_sentiment_classifier_pipeline()

    def get_sentiment_classifier_pipeline(self) -> pipeline:
        return pipeline("sentiment-analysis", 
                    model=self.sentiment_classifier_model_name,
                    device=-1, # Force CPU usage (0 = index of first CPU)
                    truncation=True,
                    max_length=512, # Increase the max length to 512
                    )
        
    def generate_sentiment_labels(self, df: pd.DataFrame, verbose=False) -> pd.DataFrame:
        if verbose: print("Generating sentiment labels...")
        results = self.pipeline(df['text'].tolist(), truncation=True, max_length=512)  
        df['sentiment_label'] = [1 if result['label'] == 'POSITIVE' else 0 for result in results]
        
        return df
    

class ReviewSummarizer:
    '''
        Review Summarizer class to summarize the reviews using the HuggingFace pipeline.
    '''
    def __init__(self, config):
        self.config = config
        self.summarizer_model_name = config.config_dict['summarizer_model_name']
        self.summarizer_pipeline = self.get_text_summarizer_pipeline()
        self.max_len = 128
        
    def get_text_summarizer_pipeline(self) -> pipeline:
        '''
            Load the summarizer model pipeline.

            Args:
                None
            
            Returns:
                Summarizer model pipeline.
        '''
        summarizer = pipeline("summarization",
                            model=self.summarizer_model_name, 
                            device=-1,  # Force CPU usage (0 = index of first CPU)
                            max_length=self.max_len, 
                            # truncation=True, 
                           )
        
        return summarizer

    def summarize_reviews(self, df: pd.DataFrame) -> str:
        '''
            Call the summarizer model pipeline to summarize the reviews in the DataFrame.
            
            Args:
                df: DataFrame containing the reviews to summarize.
            
            Returns:
                List of summaries of the reviews.
        '''
        reviews = df['text'].tolist()
        reviews = ' '.join(reviews)  # Merge all the reviews chunks into a single string
        
        summaries = self.summarizer_pipeline(reviews, truncation=True, max_length=self.max_len)
        for summary in summaries:
            print("len(summary['summary_text']): ", len(summary['summary_text']))
        
        return summaries[0]['summary_text']

    def get_review_chunks(self, reviews: pd.DataFrame, n_chunks: Optional[int]=5, chunk_size: Optional[int]=10):
        '''
            Create n chunks of k reviews each - we want n review summaries.
            Reviews are randomly sampled to create each chunk are all reviews within a chunk 
            are merged into a single string.
            
            Args:
                reviews: DataFrame containing the reviews.
                n_chunks: Number of review chunks to summarize.
                chunk_size: Number of reviews in each chunk.
            
            Returns:
                DataFrame containing the review chunks.
        '''
        # Create n chunks of k reviews each - we want n review summaries 
        review_chunks = []
        num_revs_added = 0
        for i in range(n_chunks):
            # sample random indices for reviews to fetch
            random_indices = np.random.choice(reviews.shape[0], 
                        size=min(reviews.shape[0], chunk_size), replace=False)

            selected_texts = reviews.iloc[random_indices]['text'].tolist()
            num_revs_added += len(random_indices)
            
            # Merge all the text rows in this chunk
            merged_text = ''.join(selected_texts)

            review_chunks.append(merged_text)

            if num_revs_added >= 0.8*reviews.shape[0]:
                break
            
        return pd.DataFrame({'text': review_chunks})

    def generate_per_sentiment_summary(self, review_df: pd.DataFrame, n_chunks: Optional[int]=5, 
                                       chunk_size: Optional[int]=25, verbose: Optional[bool]=False):
        '''
            Generate summary of positive and negative reviews - by feeding reviews of each kind 
            to the summarizer model.
        
            Args:
                review_df: DataFrame containing the reviews and their sentiment labels.
                n_chunks: Number of review chunks to summarize.
                chunk_size: Number of reviews in each chunk.
                verbose: Whether to print the progress.
            
            Returns:
                Dictionary containing the summary of positive and negative reviews.
        '''
        pos_reviews = review_df[review_df['sentiment_label'] == 1]['text'].to_frame()
        neg_reviews = review_df[review_df['sentiment_label'] == 0]['text'].to_frame()

        if verbose:
            print("Number of positive reviews:", len(pos_reviews))
            print("Number of negative reviews:", len(neg_reviews))

        pos_reviews_chunk = self.get_review_chunks(pos_reviews, n_chunks, chunk_size)
        neg_reviews_chunk = self.get_review_chunks(neg_reviews, n_chunks, chunk_size)

        if verbose: print("Summarizing positive reviews...")
        pos_summary = self.summarize_reviews(pos_reviews_chunk)
        
        if verbose: print("Summarizing negative reviews...")
        neg_summary = self.summarize_reviews(neg_reviews_chunk)
        
        return {'pos_summary': pos_summary, 'neg_summary': neg_summary}


class ActionRecommender:
    '''
        ActionRecommender class to generate actionable insights - based on business's user review summaries - 
        using the ChatGPT model.
    '''
    def __init__(self, config):
        with open(config.config_dict['openai_api_key_path'], 'r') as f:
            api_key = f.read().strip()
        self.chat_client = OpenAI(api_key=api_key)
        self.gpt_model_name = config.config_dict['action_generator_GPT_name']
        
    def get_context_instructions(self):
        '''
            Set context instructions for ChatGPT - helps to make model output more relevant.
        '''
        return "A resutarant owner/manager has requested actionable insights based on the user reviews \
        the business received on restaurant aggregator/recommender website. \
        Following information will contain: \
            1] Business' city and category\
            2] Summary of positive and negative reviews - for the business and for other businesses, \
                from the same category and city.\
        "
    
    def get_output_instructions(self):
        '''
            Set output instructions for ChatGPT - helps to control the model output structure and content.
        '''
        return ''' Output Instructions: \
                Be concise, practical, aware of context like business category and user reviews. \
                Insights/recommendations should be clear, actionable and domain specific. \
                Do not equivocate if unsure. Have bullet points structure in response.\
                
                Goals of readers:\
                Understand and improve the customer experience and ratings.  \
                Identify areas of improvement - both comparatively to others and individually. \
                Improve the customer satisfaction and business financials. \
                
                Your Answer should of format: 
                    Point 1:
                        - Actionable Insight 1
                        - Actionable Insight 2
                    Point 2:
                        - Actionable Insight 1
                        - Actionable Insight 2
                    etc. 
                'Point', 'Actionable Insight 1' etc.are placeholders. Replace 'Point 1/2' with name of the 
                recommendation. Remove 'Actionable Insight 1/2' from your response.\
                The Point should end with a colon and the Actionable Insight should start with a hyphen.\
                    
                Do not have any text preceeding or following the actionable insights.\
                Have at least 1 point and 1 actionable insight in the point.\
                Do not have more than 5 points and no more than 3 actionable insights in the point.\
            '''
        
    def get_info_str(self, bus_review_summary_dict: Dict, other_review_summary_dict: Dict, 
                    curr_bus_data: pd.DataFrame) -> str:
        '''
            Build a string with business's review summary for the ChatGPT prompt.
        '''
        bus_city = curr_bus_data.city.values[0]
        bus_category = curr_bus_data.categories.values[0].split(', ')
        
        return f"Here is the data: \
                City: {bus_city};\
                Category: {bus_category};\n\
                Business' review summaries: \n\
                    \tPositive: {bus_review_summary_dict["pos_summary"]};\n\
                    \tNegative: {bus_review_summary_dict["neg_summary"]}.\n\
                Other Business' review summaries: \n\
                    \tPositive: {other_review_summary_dict["pos_summary"]};\n\
                    \tNegative: {other_review_summary_dict["neg_summary"]}.\
            "
    
    def generate_prompt(self, bus_review_summary_dict: Dict, other_review_summary_dict: Dict,
                        curr_bus_data: pd.DataFrame) -> str:
        '''
        Generate a prompt for the ChatGPT model.
        '''
        context_instruct = self.get_context_instructions()
        op_instruct = self.get_output_instructions()
        info_str = self.get_info_str(bus_review_summary_dict, other_review_summary_dict, 
                                     curr_bus_data)
        
        prompt = f"{context_instruct}\n{op_instruct}\n{info_str}\n"
        
        return prompt
    
    def generate_actions(self, bus_review_summary_dict: Dict, other_review_summary_dict: Dict, 
                                  curr_bus_data: pd.DataFrame, verbose: Optional[bool]=False) -> str:
        '''
            Prompt the ChatGPT model and collect the response in desired format.
        '''
        context_instruct = self.get_context_instructions()
        op_instruct = self.get_output_instructions()
        info_str = self.get_info_str(bus_review_summary_dict, other_review_summary_dict, 
                                     curr_bus_data)
        response = self.chat_client.chat.completions.create(
                model=self.gpt_model_name,
                messages=[
                    {"role": "system", "content": context_instruct + op_instruct},
                    {"role": "user", "content": info_str},
                    ]
                )
        response = response.choices[0].message.content
        # print(response)
        
        return response

        
        
    
