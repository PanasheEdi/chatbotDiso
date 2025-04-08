import pandas as pd
import os
import mysql.connector
from mysql.connector import Error
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import ssl

#fixing SSL certificate issue for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

#NLTK resources
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) - {"not", "no", "never"}

def clean_text(text):
    
    if not isinstance(text, str) or not text.strip():
        return ""
    
   
    has_exclamation = '!' in text
    has_question = '?' in text
    has_caps = bool(re.search(r'[A-Z]{2,}', text))
    
   
    text = re.sub(r'[^\w\s!?.,;:)]', ' ', text)
    text = text.lower()
    words = text.split()
    
    # Skip stopword removal for very short texts
    if len(words) <= 5:
        cleaned = ' '.join(words)
    else:
        # Keep emotion-related words
        emotion_stopwords = {'not', 'no', 'never', 'none', 'nothing', 'very', 'too', 'so', 
                           'really', 'just', 'only', 'but', 'yet', 'still'}
        filtered_stopwords = [w for w in stop_words if w not in emotion_stopwords]
        words = [lemmatizer.lemmatize(word) for word in words if word not in filtered_stopwords]
        cleaned = ' '.join(words)
    
    if has_exclamation:
        cleaned += " HAS_EXCL"
    if has_question:
        cleaned += " HAS_QSTN"
    if has_caps:
        cleaned += " HAS_CAPS"
        
    return cleaned

def balance_data(df, min_samples=1000, max_samples=5000):
   #balanced dataset with logarithmic sampling
    emotion_counts = df['emotion'].value_counts()
    balanced_df = pd.DataFrame()
    
    for emotion, count in emotion_counts.items():
        if count <= min_samples:
            #taking all samples for rare classes
            samples = df[df['emotion'] == emotion]
        else:
            #calculating sample size with logarithmic scaling
            sample_size = int(min_samples + np.log(count) * (count - min_samples) / np.log(emotion_counts.max()))
            sample_size = min(sample_size, count, max_samples)
            samples = df[df['emotion'] == emotion].sample(sample_size, random_state=42)
            
        balanced_df = pd.concat([balanced_df, samples])
    
    return balanced_df.sample(frac=1, random_state=42)  #shuffle

def create_emotion_groups(df):
    """Group emotions into broader categories"""
    emotion_groups = {
        'positive': ['admiration', 'amusement', 'approval', 'caring', 'desire' 
                    'excitement', 'gratitude', 'joy', 'love', 'optimism', 'pride', 'relief'],
                    
        'negative': ['anger', 'annoyance', 'disappointment', 'disapproval', 'disgust', 
                     'grief', 'remorse', 'sadness'],

        'neutral': ['realization', 'surprise', 'curiosity','neutral',],

        'anxious_stressed': ['embarrassment','nervousness', 'restlessness','confusion','fear','sadness', 'anger','disgust']
    }
    
    # Create a mapping dictionary
    emotion_mapping = {}
    for group, emotions in emotion_groups.items():
        for emotion in emotions:
            emotion_mapping[emotion] = group
    
    # Apply mapping
    df['emotion_group'] = df['emotion'].map(emotion_mapping)
    
    return df

def prepare_data(use_grouped_emotions=True, sample_size=50000):
    """Main function to load and prepare the data"""
    # Database connection configuration
    db_config = {
        'host': 'localhost',
        'user': 'panashe',
        'password': 'Shaun2009',
        'database': 'Chatbot_database'
    }
    
    # Connect to MySQL database
    try:
        connection = mysql.connector.connect(**db_config)
        if connection.is_connected():
            print("Successfully connected to MySQL database")
    except Error as e:
        print("Error while connecting to MySQL", e)
        return None
    
    # Load data from MySQL
    query = "SELECT text, emotion FROM go_emotions"
    try:
        df = pd.read_sql(query, con=connection)
        print(f"Loaded {len(df)} records from the database")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    finally:
        if connection.is_connected():
            connection.close()
            print("MySQL connection closed")
    
    # Basic data cleaning
    df.dropna(subset=['text'], inplace=True)  # Drop rows with missing text
    df.drop_duplicates(subset=['text'], inplace=True)  # Remove duplicate texts
    
    # Clean the 'text' column using the clean_text function
    print("Cleaning text data...")
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Drop any remaining rows where 'emotion' is missing or clean_text is empty
    df.dropna(subset=['emotion'], inplace=True)
    df = df[df['clean_text'].notna() & (df['clean_text'].astype(str).str.strip() != '')]
    
    print(f"Data shape after cleaning: {df.shape}")
    
    # Create emotion groups
    df = create_emotion_groups(df)
    
    # Choose target (grouped emotions or all emotions)
    if use_grouped_emotions:
        df['target'] = df['emotion_group']
        print("Using grouped emotions as target (positive, negative, neutral)")
    else:
        df['target'] = df['emotion']
        print("Using all individual emotions as target")
    
    # Balance data
    print("Balancing dataset...")
    balanced_df = balance_data(df)
    print(f"Balanced data shape: {balanced_df.shape}")
    
    # Sample if necessary
    if sample_size and sample_size < len(balanced_df):
        print(f"Sampling {sample_size} records...")
        stratified_sample = balanced_df.groupby('target', group_keys=False).apply(
            lambda x: x.sample(min(len(x), sample_size // len(balanced_df['target'].unique())), random_state=42)
        )
        final_df = stratified_sample
    else:
        final_df = balanced_df
    
    print(f"Final dataset shape: {final_df.shape}")
    print(f"Target distribution:\n{final_df['target'].value_counts()}")
    
    return final_df

if __name__ == "__main__":
    #reminder: true = grouped emotions -positive -negative -neutral -anxious/stressed, false= individual emotions
    USE_GROUPED_EMOTIONS = True
    
    # Prepare the data
    prepared_data = prepare_data(use_grouped_emotions=USE_GROUPED_EMOTIONS)
    
    # Save processed data to CSV for later use
    if prepared_data is not None:
        prepared_data.to_csv('processed_emotion_data.csv', index=False)
        print("Processed data saved to 'processed_emotion_data.csv'")