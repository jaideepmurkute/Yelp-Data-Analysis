# Yelp-Data-Analysis

## Description
This project aims to analyze Yelp data and extract meaningful insights.   
**ETL and Preprocessing Pipeline**:  
&emsp; Performs Data Extraction, Compression, Loading. Cleans up and encodes the data. Created SQL database.  
**EDA Pipeline**:   
&emsp; Generates data insights from data analysis with SQL queries and visualization with python.  
**Analytics**:   
&emsp; Performs analysis of user reviews a business recies and generates a report with insights and actionable steps for the the business.  
&emsp;    (Sentiment classification, Review Summarization, Action Generation, Report Creation)  

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/Yelp-Data-Analysis.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
### ETL Pipeline:
```bash
cd ETL
```    
1. ### Read the JSON data from Yelp, compress it and store as a parquet file
    ```bash
    python extrat_and_compress_data.py
    ```
2. #### Perform data pre-processing, duplicate cleanups, dropping irrelevant data,data format encoding etc. 
    ```bash
    python preprocess_data.py
    ```
3. #### Create SQL database from the parquet data - for easier analysis.
    ```bash
    python create_sql_db.py
    ```
---------------------------------------------------------------------------------------------------
### Exploratory Data Analysis 
```bash
cd EDA
```
1. #### Perform Exploratory data analysis with SQL:
   ```
   Run the notebook: EDA_sql.ipynb
   ```
2. #### Perform Data vizualization with Python:
   ```
   Run the notebook: EDA_visualizations.ipynb
   ```
---------------------------------------------------------------------------------------------------
### Analytics 
```bash
cd User Review Analytics
```
1. #### Perform User Review Analytics(NLP) - Sentiment classification, Summarization and Actions Recommender(Generator)    
    ```bash
    python review_insights.py
    ```

## License
This project is licensed under the [MIT License](LICENSE).
