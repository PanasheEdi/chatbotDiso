import pytest
import pandas as pd
import numpy as np
from SentimentModel import train_sentiment_model, predict_emotion 
import joblib
from SentimentModel import TextFeatureExtractor
@pytest.fixture
def text_feature_extractor():
    return TextFeatureExtractor()

# Fixture to load the model and label encoder
@pytest.fixture
def sentiment_model():
    pipeline = joblib.load('sentiment_model_pipeline.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    return pipeline, label_encoder

# Test 1: Check text feature extraction

def test_text_feature_extractor(text_feature_extractor):
    data = pd.Series([
        "This is a test!",
        "Hello World",
        "I am testing my model?",
        "How is everyone today"
    ])

    features = text_feature_extractor.fit_transform(data)

    assert features.shape == (4, 7), "Feature extraction failed to produce the correct number of features."
    assert isinstance(features[0][0], (int, float)), "Text length is not numeric."
    assert isinstance(features[0][1], (int, float)), "Word count is not numeric."

# Test 2: Check if sentiment model can be loaded and returns non-None pipeline and label encoder
def test_train_sentiment_model():
    sampleData = "/Users/panashekamuteku/Documents/Brunel OneDrive/OneDrive - Brunel University London/Year 3/FYP/Dissertation/sample data/ML test data.csv"
    
    # Fixing the argument name here: change data_path to sampleData
    pipeline, label_encoder = train_sentiment_model(data_path=sampleData, grid_search=False, save_output=False)

    assert pipeline is not None, "Pipeline was not returned."
    assert label_encoder is not None, "Label encoder was not returned"
    assert 'classifier' in pipeline.named_steps, "Classifier is not found in the pipeline."


# Test 3: Check emotion prediction for non-empty text
def test_predict_emotion(sentiment_model):
    pipeline, label_encoder = sentiment_model
    text = "I am so happy!"

    emotion = predict_emotion(text, pipeline=pipeline, label_encoder=label_encoder)

    assert emotion in label_encoder.classes_, f"Predicted emotion '{emotion}' is not in the label encoder classes"

# Test 4: Check emotion prediction for empty string
def test_predict_emotion_empty_string(sentiment_model):
    pipeline, label_encoder = sentiment_model
    empty_text = ""
    emotion_empty = predict_emotion(empty_text, pipeline=pipeline, label_encoder=label_encoder)

    assert emotion_empty in label_encoder.classes_, "Predicted emotion for empty text is invalid."


def test_predict_emotion_stressed(sentiment_model):
    pipeline, label_encoder = sentiment_model
    text = "I feel so overwhelmed and stressed idk why"
    expected_label = "anxious_stressed"
    emotion = predict_emotion(text, pipeline=pipeline, label_encoder=label_encoder)

    assert emotion in label_encoder.classes_, f"Predicted emotion '{emotion}' is not in the label encoder classes"
    assert emotion == expected_label, f"predicted emotion'{emotion}'does not match expected label '{expected_label}"
