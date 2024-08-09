import openai
from openai import OpenAI
import pandas as pd
import numpy as np
import torch
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel
torch.set_num_threads(4)


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
        self.config = config
        self.pipeline = self.get_sentiment_classifier_pipeline()

    def get_sentiment_classifier_pipeline(self) -> pipeline:
        return pipeline("sentiment-analysis", 
                    model="distilbert-base-uncased-finetuned-sst-2-english",
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
    def __init__(self, config):
        self.config = config
    
    def get_text_summarizer_pipeline(self) -> pipeline:
        summarizer = pipeline("summarization",
                            model="t5-base",
                            device=-1,  # Force CPU usage (0 = index of first CPU)
                            # max_length=128, 
                            # truncation=True, 
                           )
        
        return summarizer

    def summarize_reviews(self, df, summarizer):
        reviews = df['text'].tolist()
        reviews = ' '.join(reviews)  # Merge all the reviews chunks into a single string
        
        summaries = summarizer(reviews, truncation=True, max_length=128)
        for summary in summaries:
            print("len(summary['summary_text']): ", len(summary['summary_text']))
        
        # df['summary'] = [summary['summary_text'] for summary in summaries]
        
        return summaries[0]['summary_text']

    def get_review_chunks(self, reviews, n_chunks=5, chunk_size=10):
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

    def generate_per_sentiment_summary(self, review_df, verbose=False):
        pos_reviews = review_df[review_df['sentiment_label'] == 1]['text'].to_frame()
        neg_reviews = review_df[review_df['sentiment_label'] == 0]['text'].to_frame()

        if verbose:
            print("Number of positive reviews:", len(pos_reviews))
            print("Number of negative reviews:", len(neg_reviews))

        # ----------------------

        n_chunks = 5  # How many review chunks, each of size 'each_chunk_reviews', to summaize
        chunk_size = 25  # Number of reviews in each chunk

        pos_reviews_chunk = self.get_review_chunks(pos_reviews, n_chunks, chunk_size)
        neg_reviews_chunk = self.get_review_chunks(neg_reviews, n_chunks, chunk_size)

        # ----------------------

        summarizer_pipeline = self.get_text_summarizer_pipeline()
        
        if verbose: print("Summarizing positive reviews...")
        # pos_review_summaries = self.summarize_reviews(pos_reviews_chunk, summarizer_pipeline)
        # pos_summary = ''.join(pos_review_summaries.summary.values)
        pos_summary = self.summarize_reviews(pos_reviews_chunk, summarizer_pipeline)
        
        if verbose: print("Summarizing negative reviews...")
        # neg_review_summaries = self.summarize_reviews(neg_reviews_chunk, summarizer_pipeline)
        # neg_summary = ''.join(neg_review_summaries.summary.values)
        neg_summary = self.summarize_reviews(neg_reviews_chunk, summarizer_pipeline)
        
        return {'pos_summary': pos_summary, 'neg_summary': neg_summary}


class ActionRecommender:
    def __init__(self, config):
        with open(config.config_dict['openai_api_key_path'], 'r') as f:
            api_key = f.read().strip()
        self.chat_client = OpenAI(api_key=api_key)
        
    def get_context_instructions(self):
        return "A resutarant owner/manager has requested actionable insights based on the user reviews \
        the business received on restaurant aggregator/recommender website. \
        Following information will contain: \
            1] Business' city and category\
            2] Summary of positive and negative reviews - for the business and for other businesses, \
                from the same category and city.\
        "
    
    def get_output_instructions(self):
        return " Output Instructions: \
                Be concise, practical, aware of context like business category and user reviews. \
                Insights/recommendations should be clear, actionable and domain specific. \
                Do not equivocate if unsure. Have bullet points structure in response.\
                Goals of readers:\
                Understand and improve the customer experience and ratings.  \
                Identify areas of improvement - both comparatively to others and individually. \
                Improve the customer satisfaction and business financials. \
            "
        
        # return ""
    
    def get_info_str(self, bus_review_summary_dict, other_review_summary_dict, 
                                     curr_bus_data):
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
    
    def generate_prompt(self, bus_review_summary_dict, other_review_summary_dict,
                        curr_bus_data):
        context_instruct = self.get_context_instructions()
        op_instruct = self.get_output_instructions()
        info_str = self.get_info_str(bus_review_summary_dict, other_review_summary_dict, 
                                     curr_bus_data)
        
        prompt = f"{context_instruct}\n{op_instruct}\n{info_str}\n"
        
        return prompt
    '''
    def generate_actions(self, bus_review_summary_dict, other_review_summary_dict, 
                                  curr_bus_data, verbose=False):
        # bus_review_summary_dict = {'pos_summary': 'this is a good restaurant', 'neg_summary': 'this is a bad restaurant'}
        # other_review_summary_dict = {'pos_summary': 'this is a good restaurant', 'neg_summary': 'this is a bad restaurant'}
        # curr_bus_data = pd.DataFrame({'city': ['Toronto'], 'categories': ['Restaurants']})
        
        prompt = self.generate_prompt(bus_review_summary_dict, other_review_summary_dict,
                                        curr_bus_data)
        print("Prompt: ", prompt)
        print("*^&%^"*10)
        
        print("len(prompt): ", len(prompt))
        print("len(prompt.split()): ", len(prompt.split()))
        
        
        # prompt = "hello"
        
        # Load pre-trained model and tokenizer
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        model.eval()
        
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # max_length = 512
        
        # Ensure input_ids are truncated to the model's maximum sequence length
        input_ids = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=True, 
                                     max_length=512, truncation=True)
        print("input_ids.shape: ", input_ids.shape)
        
        # input_ids = input_ids[:, :max_length]
        print("After slicing to max_length; input_ids.shape: ", input_ids.shape)
        
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        
        print("attention_mask.shape: ", attention_mask.shape)
        
        outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,
                num_return_sequences=5,
                do_sample=True,
                top_k=50, # limits the sampling pool to the top k most probable next tokens
                top_p=0.95,  # (nucleus sampling) smallest set of tokens whose cumulative probability is greater than or equal to p
                temperature=0.7, # affects randomness of the output, lower means more deterministic
                # An n-gram is a contiguous sequence of n items from a given sample of text
                no_repeat_ngram_size=2, # prevent the model from repeating phrases of a certain length. For example, if no_repeat_ngram_size=2, the model will avoid repeating any bigrams
                repetition_penalty=1.2 # penalty to the probability of tokens that have already been generated,
            )
        
        # Decode and print only the newly generated sequence
        decoded_outputs = []
        input_length = input_ids.shape[1]
        for i, output in enumerate(outputs):
            # Get the length of the input sequence
            
            # Slice the output to get only the newly generated tokens
            generated_sequence = output[input_length:]
            
            generated_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
            # Simple post-processing: remove any text after the first period to get a complete sentence
            generated_text = generated_text.split('.')[0] + '.'

            print(f"Generated Text {i+1}: {generated_text}")
            decoded_outputs.append(generated_text)
            
            print("-"*30)

        
        print('-'*30)
        
        return decoded_outputs
    '''
    def generate_actions(self, bus_review_summary_dict, other_review_summary_dict, 
                                  curr_bus_data, verbose=False):
        # prompt = self.generate_prompt(bus_review_summary_dict, other_review_summary_dict,
        #                                 curr_bus_data)
        '''
        # Make a request to the OpenAI GPT-4 API
        response = openai.Completion.create(
            engine="text-davinci-004",  # Use "gpt-4" for GPT-4 model
            prompt=prompt,
            
            max_tokens=150,  # The maximum number of tokens to generate.
            n=1,    # The number of responses to generate.
            
            # The sequence where the API should stop generating further tokens. If None, it will generate until it hits 
            # the max tokens limit.
            stop=None,  
            
            # Controls the randomness of the output. Lower values make the output more focused and deterministic.
            temperature=0.7  # Adjust the temperature for more creative or focused responses
        )
        
        # Extract the text response
        response = response.choices[0].text.strip()
        '''
        
        '''
        context_instruct = self.get_context_instructions()
        op_instruct = self.get_output_instructions()
        info_str = self.get_info_str(bus_review_summary_dict, other_review_summary_dict, 
                                     curr_bus_data)
        
        # Make a request to the OpenAI GPT-4 API
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Use "gpt-4" for GPT-4 model
            messages=[
                {"role": "system", "content": context_instruct + op_instruct},
                {"role": "user", "content": info_str}
            ],
            max_tokens=150,  # The maximum number of tokens to generate.
            n=1,  # The number of responses to generate.
            
            # The sequence where the API should stop generating further tokens. If None, it will generate until it hits the max tokens limit.
            stop=None,  
            
            # Controls the randomness of the output. Lower values make the output more focused and deterministic.
            temperature=0.7  # Adjust the temperature for more creative or focused responses
        )

        # Extract the response text
        response_text = response['choices'][0]['message']['content']

        # Print the response
        print(response_text)
        
        return response_text
        '''
        
        
        context_instruct = self.get_context_instructions()
        op_instruct = self.get_output_instructions()
        info_str = self.get_info_str(bus_review_summary_dict, other_review_summary_dict, 
                                     curr_bus_data)
        response = self.chat_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": context_instruct+op_instruct},
                    {"role": "user", "content": info_str},
                ]
                )
        
        response = response.choices[0].message.content
        
        print(response)
        
        return response


# ar = ActionRecommender()
# # ar.generate_actions({'pos_summary': 'this is a good restaurant', 'neg_summary': 'this is a bad restaurant'},
# #                     {'pos_summary': 'this is a good restaurant', 'neg_summary': 'this is a bad restaurant'},
# #                     pd.DataFrame({'city': ['Toronto'], 'categories': ['Restaurants']}), verbose=True)
        
# context_str = ar.get_context_instructions()
# op_str = ar.get_output_instructions()
# tot_str = context_str + op_str
# print("len(tot_str): ", len(tot_str))
# print("len(tot_str.split()): ", len(tot_str.split()))

        
        
    
