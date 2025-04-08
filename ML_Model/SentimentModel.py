import pandas as pd
import numpy as np
import re
import nltk
import ssl
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE
import os

# Fix SSL certificate issue
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

#Custom transformer for text features 
#to add more features to the text data 
class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # Create DataFrame from X series
        df = pd.DataFrame({'text': X})
        
        features = pd.DataFrame()
        features['text_length'] = df['text'].apply(len)
        features['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
        features['avg_word_length'] = df['text'].apply(lambda x: 
                                                sum(len(word) for word in str(x).split()) / 
                                                max(len(str(x).split()), 1))
        features['exclamation_count'] = df['text'].apply(lambda x: str(x).count('!'))
        features['question_count'] = df['text'].apply(lambda x: str(x).count('?'))
        features['period_count'] = df['text'].apply(lambda x: str(x).count('.'))
        features['capital_ratio'] = df['text'].apply(lambda x: 
                                             sum(1 for c in str(x) if c.isupper()) / 
                                             max(len(str(x)), 1))
        
        return features.values

def train_sentiment_model(data_path=None, grid_search=False, save_output=True):
    if data_path and os.path.exists(data_path):
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
    else:
        print("No data file provided or file doesn't exist. Please run data preparation first.")
        return None, None
    
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['target'].value_counts()}")
    
    # features and targets
    df = df.dropna(subset=['target'])
    
    X = df["clean_text"]
    y = df["target"]

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"Unique classes: {label_encoder.classes_}")
    print(f"Class distribution: {np.bincount(y_encoded)}")
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Create the feature extraction pipeline
    print("\nBuilding pipeline...")
    feature_pipeline = Pipeline([
        ('features', FeatureUnion([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 4),
                min_df=3, 
                max_df=0.9, 
                max_features=3000,
                sublinear_tf=True
            )),
            ('text_features', TextFeatureExtractor()) #- keeping as seen improvements with my custom features
        ]))
    ])
    
    # Apply feature extraction
    X_train_features = feature_pipeline.fit_transform(X_train)
    X_test_features = feature_pipeline.transform(X_test)
    
    print("Unique classes in y_train:", np.unique(y_train))
    from collections import Counter
    print("Original y_train class distribution:", Counter(y_train))


    # applying SMOTE for balancing (separate from the pipeline)
    print("\nApplying SMOTE for class balancing...")

    smote_strat = {
        0:8558,
        1:8558,
        2:8558,
        3:15106

    }

    smote = SMOTE(sampling_strategy= smote_strat, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_features, y_train)
    
    print("\nTraining the classifier...")
    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        oob_score=True,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    if grid_search:
        print("\nPerforming grid search for hyperparameter tuning...")
        param_grid = {
            'n_estimators': [200, 300, 400],
            'max_depth': [20, 30, 40],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        grid_search = GridSearchCV(
            estimator=classifier,
            param_grid=param_grid,
            cv=3,
            scoring='f1_weighted',
            verbose=2,
            n_jobs=-1
        )
        
        grid_search.fit(X_train_resampled, y_train_resampled)
        print(f"Best parameters: {grid_search.best_params_}")
        classifier = grid_search.best_estimator_
    else:
        # fitting classifier directly
        classifier.fit(X_train_resampled, y_train_resampled)
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = classifier.predict(X_test_features)
    
    # Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Print Accuracy Metrics
    print("\nModel Performance:")
    print(f"Model Accuracy: {accuracy:.2f}")
    print(f"Model Precision: {precision:.2f}")
    print(f"Model Recall: {recall:.2f}")
    print(f"Model F1-Score: {f1:.2f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    if save_output:
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=label_encoder.classes_, 
                    yticklabels=label_encoder.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        print("\nConfusion matrix saved as 'confusion_matrix.png'")
        
        # Create a combined pipeline for prediction (without SMOTE)
        full_pipeline = Pipeline([
            ('features', feature_pipeline.named_steps['features']),
            ('classifier', classifier)
        ])
        
        # Save models   
        print("\nSaving model artifacts...")
        joblib.dump(full_pipeline, 'sentiment_model_pipeline.pkl', compress=3)
        joblib.dump(label_encoder, 'label_encoder.pkl', compress=3)
        print("\nModel and Label Encoder saved successfully!")
    
    full_pipeline = Pipeline([
        ('features', feature_pipeline.named_steps['features']),
        ('classifier', classifier)
    ])
        
    return full_pipeline, label_encoder

#to predict emotions with the trained model
from SentimentModel import TextFeatureExtractor
def predict_emotion(text, pipeline=None, label_encoder=None):
    """Predict emotion for a given text"""
    if pipeline is None or label_encoder is None:
        try:
            pipeline = joblib.load('sentiment_model_pipeline.pkl')
            label_encoder = joblib.load('label_encoder.pkl')
        except:
            print("Error loading model. Please train the model first.")
            return None
    
    # Clean text (simplified version for prediction)
    def quick_clean(text):
        if not isinstance(text, str) or not text.strip():
            return ""
        
        has_exclamation = '!' in text
        has_question = '?' in text
        has_caps = bool(re.search(r'[A-Z]{2,}', text))
        
        text = re.sub(r'[^\w\s!?.,;:)]', ' ', text)
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Add indicators as special tokens
        if has_exclamation:
            text += " HAS_EXCL"
        if has_question:
            text += " HAS_QSTN"
        if has_caps:
            text += " HAS_CAPS"
            
        return text
    
    # Clean and predict
    cleaned_text = quick_clean(text)
    prediction = pipeline.predict([cleaned_text])[0]
    emotion = label_encoder.inverse_transform([prediction])[0]

    emotion_groups = {
        'positive': ['admiration', 'amusement', 'approval', 'caring', 'desire', 
                    'excitement', 'gratitude', 'joy', 'love', 'optimism', 'pride', 'relief'],
        'negative': ['anger', 'annoyance', 'disappointment', 'disapproval', 'disgust', 
                     'grief', 'remorse', 'sadness'],
        'neutral': ['realization', 'surprise', 'curiosity', 'neutral'],
        'anxious_stressed': ['embarrassment', 'nervousness', 'restlessness', 'confusion', 'fear']
    }

    for group, emotions in emotion_groups.items():
        if emotion in emotions:
            emotion = group
            break
    
    return emotion

if __name__ == "__main__":
    # Set to True to perform grid search (takes longer)
    PERFORM_GRID_SEARCH = False
    
    # Check if data file exists
    data_path = 'processed_emotion_data.csv'
    if not os.path.exists(data_path):
        print(f"ERROR: Data file '{data_path}' not found. Please run data preparation first.")
        exit(1)
    
    # Train the model
    model, encoder = train_sentiment_model(
        data_path=data_path,
        grid_search=PERFORM_GRID_SEARCH,
        save_output=True
    )
    
    # Test the model with sample texts
    if model is not None and encoder is not None:
        test_texts = [
            "I felt so happy yesterday and ready for a better day today",
            "I just feel sad today idk why.",
            "Panic seized me, my breath catching in my throat as the fear consumed me.",
            "The weight of school makes me very stressed."
        ]
        
        print("\nTesting model with sample texts:")
        for text in test_texts:
            emotion = predict_emotion(text, model, encoder)
            print(f"Text: '{text}'\nPredicted emotion: {emotion}\n")