import pandas as pd
import os
import sys
# Add the parent directory to sys.path to ensure Python can find app.py and models.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src')))
from app import app 
from models import db, GoEmotionsData  # Import from models.py in the parent directory

db_config = {
    'host': 'localhost',
    'user': 'panashe',
    'password': 'Shaun2009',
    'database': 'Chatbot_database'
}

csv_files = [
    "/Users/panashekamuteku/Documents/Brunel OneDrive/OneDrive - Brunel University London/Year 3/FYP/DatabaseForFYP/GoEmotionsData/goemotions_1.csv",
    "/Users/panashekamuteku/Documents/Brunel OneDrive/OneDrive - Brunel University London/Year 3/FYP/DatabaseForFYP/GoEmotionsData/goemotions_2.csv",
    "/Users/panashekamuteku/Documents/Brunel OneDrive/OneDrive - Brunel University London/Year 3/FYP/DatabaseForFYP/GoEmotionsData/goemotions_3.csv"
]

df_list = []
for file in csv_files:
    if not os.path.exists(file):
        continue
    try:
        df = pd.read_csv(file)
        df_list.append(df)
    except Exception as e:
        print(f"Error reading file {file}: {e}")

if not df_list:
    print("No data loaded from the CSV files. Exiting script.")
else:
    df = pd.concat(df_list, ignore_index=True)

    emotion_columns = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 
        'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 
        'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 
        'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]
    
    try:
        # With SQLAlchemy, set up a session for MySQL using Flask's app context
        with app.app_context():
            for _, row in df.iterrows():
                if 'text' in df.columns and any(emotion in df.columns for emotion in emotion_columns):
                    # Get the dominant emotion
                    emotions = row[emotion_columns]
                    dominant_emotion = emotions.idxmax()
                    dominant_value = emotions.max()

                    # Insert the data into the database
                    new_entry = GoEmotionsData(
                        text=row['text'],
                        emotion=dominant_emotion
                    )
                    db.session.add(new_entry)
            
            # Commit to save changes in MySQL
            db.session.commit()
            print("Commit successful!")

        print("Import successful!")
    
    except Exception as e:
        print(f"An error occurred during database insertion: {e}")
