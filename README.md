# Yelp-Data-Analysis
## Description
This project aims to analyze Yelp data and extract meaningful insights.

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
ETL Pipeline:
    ```cmd
    cd ETL
    ```
    First, read the JSON data from Yelp, compress it and store as a parquet file
    ```cmd
    python extrat_and_compress_data.py
    ```
    Perform data pre-processing, duplicate cleanups, dropping irrelevant data,data format encoding etc. 
    python preprocess_data.py
    ```
    Create SQL database from the data - for easier analysis.
    ```cmd
    python create_sql_db.py
    ```
1. Perform Exploratory data analysis with SQL:
    Run the notebook: EDA_sql.ipynb
2. Perform Data vizualization with Python:
    Run the notebook: EDA_visualizations.ipynb

2. Run the analysis script:
    ```bash
    python analyze.py
    ```

## Contributing
Contributions are welcome! Please follow the guidelines in [CONTRIBUTING.md](CONTRIBUTING.md).

## License
This project is licensed under the [MIT License](LICENSE).

## Contact
If you have any questions or suggestions, feel free to reach out to us at [email@example.com](mailto:email@example.com)."# New Repository" 
