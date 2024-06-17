import os  # Package for executing OS commands
from kaggle.api.kaggle_api_extended import KaggleApi  # For kaggle API to import datasets
import psycopg2  # Integrate with our DB
import pandas as pd  # Import pandas for Dataframe transformations
import logging # log our app
import time # to use sleep 
import urllib.request # work with urls

# Config our logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load our dataset onto a DataFrame
def load_dataset():
    # Define API
    api = KaggleApi()

    # Define our dataset
    dataset_name = 'emirhanai/social-media-usage-and-emotional-well-being'
    dataset_dir = './datasets/'

    # Create dir if doesnt exist,
    os.makedirs(dataset_dir, exist_ok=True)
    # Download dataset and unzip
    api.dataset_download_files(dataset_name, path=dataset_dir, unzip=True)

    logging.info(f"Dataset {dataset_name} downloaded to {dataset_dir}")

    # List all directory files
    dataset_dir_files = os.listdir(dataset_dir)
    dataset_dir_files

    # Define our desired dataset file
    dataset_file = f'{dataset_dir}train.csv'

    if len(dataset_dir_files) > 0:
        # read dataset from csv into pandas df
        df = pd.read_csv(dataset_file)

        logging.info(f"dataset file {dataset_file} was loaded into a DataFrame")
    else:
        logging.warning("No files found in the dataset directory.")

    return df


# Transform our dataset
def transform_dataset(dataset_df):
    transform_df = dataset_df.copy()

    # Sum all interactions - create 'Interactions_Per_Day' column
    transform_df['Interactions_Per_Day'] = transform_df[['Posts_Per_Day', 'Likes_Received_Per_Day',
                                     'Comments_Received_Per_Day', 'Messages_Sent_Per_Day']].sum(axis=1)

    logging.info(f"column 'Interactions_Per_Day' was created.")

    ### Clean irrelevant df values

    # Define accepted 'Gender' values and remove if not equal to accepted_values
    accepted_genders = ['Male', 'Female', 'Non-binary']
    transform_df = transform_df[transform_df['Gender'].isin(accepted_genders)]

    logging.info(f"removed irrelevant values for 'Gender' column")

    # Set labels for Interactions_Per_Day and Usage_Time_Per_Day
    labels = ['Average', 'High', 'Very High']

    usage_time_bins = [0, 100, 160, float('inf')]
    interactions_bins = [0, 75, 150, float('inf')]

    transform_df['Daily_Usage_Category'] = pd.cut(transform_df['Daily_Usage_Time (minutes)'], bins=usage_time_bins, labels=labels)
    transform_df['Interactions_Category'] = pd.cut(transform_df['Interactions_Per_Day'], bins=interactions_bins, labels=labels)

    logging.info(f"column 'Daily_Usage_Category' was created.")
    logging.info(f"column 'Interactions_Category' was created.")

    # Set labels for Dominant_Emotion
    negative_emotions = ['Anxiety', 'Sadness', 'Anger']
    positive_emotions = ['Happiness']
    neutral_emotions = ['Neutral', 'Boredom']

    emotion_mapping = {**{emotion: 'Negative' for emotion in negative_emotions},
                       **{emotion: 'Positive' for emotion in positive_emotions},
                       **{emotion: 'Neutral' for emotion in neutral_emotions}}

    transform_df['Emotion_Label'] = transform_df['Dominant_Emotion'].map(emotion_mapping)

    logging.info(f"column 'Emotion_Label' was mapped and created.")

    # Assert numeric values in numeric columns
    columns_to_check = ['Age', 'Daily_Usage_Time (minutes)', 'Posts_Per_Day', 'Likes_Received_Per_Day',
                        'Comments_Received_Per_Day', 'Messages_Sent_Per_Day']

    for column in columns_to_check:
        is_numeric = check_numeric(transform_df, column)
        if is_numeric:
            logging.info(f"assert numeric fields job: {column} - OK")
        else:
            logging.error(f"assert numeric fields job: {column} - ERROR")

    return transform_df


# Connect to DB for loading data
def connect_to_db():
    # Define dir path
    directory_path = os.path.join(os.getenv("APPDATA"), 'postgresql')

    # Create dir if doesnt exist
    os.makedirs(directory_path, exist_ok=True)

    # Download the cert file
    cert_url = 'https://cockroachlabs.cloud/clusters/2d752b1e-0a75-4a78-bc21-21b21f9692c9/cert'
    cert_file = os.path.join(directory_path, 'root.crt')

    urllib.request.urlretrieve(cert_url, cert_file)

    # Check certificate existance
    if os.path.exists(cert_file):
        logging.info(f"Certificate downloaded successfully to {cert_file}.")
    else:
        logging.warning(f"Failed to download cockroachdb certificate")

    # Define env var for DB connection
    os.environ['DATABASE_URL'] = 'postgresql://shoham:TsjZNqEwuYTjJoz-12MUqQ@naya-etl-10194.7tc.aws-eu-central-1.cockroachlabs.cloud:26257/defaultdb?sslmode=verify-full'

    # Define DB connection
    conn = psycopg2.connect(os.environ["DATABASE_URL"])

    # Execute an SQL Query using a cursor
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT now()")
            res = cur.fetchall()
            conn.commit()
            print(res)
    except:
        logging.error(f"Failed to connect to cockroachdb. ", e)

    return conn.cursor(), conn


def load_data(transformed_df):

    insert_df = transformed_df.copy()

    # Get cursor and psycopg2 connection of CRDB
    cur, conn = connect_to_db()

    # Create the table before inserting it's values
    create_table_query = '''
    CREATE TABLE IF NOT EXISTS social_media_usage_well_being (
        user_id INT,
        age INT,
        gender TEXT,
        platform TEXT,
        daily_usage_time INT,
        Posts_Per_Day INT,
        Likes_Received_Per_Day INT,
        Comments_Received_Per_Day INT,
        Messages_Sent_Per_Day INT,
        Dominant_Emotion TEXT,
        Interactions_Per_Day INT,
        Daily_Usage_Category TEXT,
        Interactions_Category TEXT,
        Emotion_Label TEXT
    );
    '''

    # Execute query and commit to CRDB
    cur.execute(create_table_query)
    conn.commit()

    logging.info(f"social_media_usage_well_being was created successfully")

    time.sleep(1)

    # Insert query - data from DataFrame to created 'social_media_usage_well_being' table
    insert_query = '''
    INSERT INTO social_media_usage_well_being (
        user_id, age, gender, platform, daily_usage_time, Posts_Per_Day,
        Likes_Received_Per_Day, Comments_Received_Per_Day, Messages_Sent_Per_Day,
        Dominant_Emotion, Interactions_Per_Day, Daily_Usage_Category,
        Interactions_Category, Emotion_Label
    ) 
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
    '''

    for index, row in insert_df.iterrows():
        # Convert the row to a tuple to prepare it for the SQL query
        values = tuple(row)
        
        # Check if the number of columns in the row matches the expected number
        if len(values) != 14:
            logging.error(f"Row does not contain 14 columns", row)
            continue
        try:
            # execute the SQL insert query with the values from the row
            cur.execute(insert_query, values)
        except Exception as e:
            # Print any error that occurs during the execution of the insert query
            logging.error(f"Error: {e}")
        
        # Commit the transaction after attempting to insert the row
        conn.commit()

    logging.info(f"Data was inserted to table 'social_media_usage_well_being' successfully")

# Function to check if all values in a column are numeric
def check_numeric(df_to_check, column):
    return pd.to_numeric(df_to_check[column], errors='coerce').notnull().all()


if __name__ == '__main__':
    # Extract dataset
    logging.info(f"Loading dataset from kaggle.com...")
    df = load_dataset()
    logging.info(f"Finished loading dataset!\n")
    time.sleep(1)

    # Transform dataset
    logging.info(f"Tranforming Data...")
    transformed_df = transform_dataset(df)
    logging.info(f"Dataset transformed!")
    
    # Load dataset to cockroachdb
    logging.info(f"Starting to load data to cockroachdb...")
    load_data(transformed_df)
    logging.info(f"Done loading data to cockroachdb!")
