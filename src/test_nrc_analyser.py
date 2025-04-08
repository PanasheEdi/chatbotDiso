import pytest
from nrc_analyser import NRCAnalyser

"""
reminder stress is the category and anxiety is the emotion"""

@pytest.fixture 
def analyser():
    return NRCAnalyser()

def test_process_text(analyser):
    input_text1 = "I have been feeling STRESSED lately, i'm not sure why..."

    processed_text = analyser.preprocess_text(input_text1)

    assert processed_text == ['feeling', 'stressed', 'lately', 'sure'], \
        f"Expected ['i','have', 'been', 'feeling', 'stressed', 'lately', 'am', 'not', 'sure', 'why'], but got {processed_text}"


def test_categorise_emotions(analyser):
    emotion_scores = {"fear": 2, "anger":1, "joy":1}
    category,_= analyser.categorise_emotions(emotion_scores)
    assert category == "stress"

def test_generate_sentiment_anxietyStress(analyser):
#checks if the function correctly identifies the joy emotion when the sentance contains words like happy

    input_text2 =("I feel so overwhelmed, i can feel my bloodpressure rise")

    analyser.preprocess_text = lambda text: ['i', 'feel', 'overwhelmed', 'i', 'can', 'feel', 'my', 'bloodpressure','rise']
    result = analyser.generate_sentiment(input_text2)
    assert result['dominant_emotion'] == "anxiety", f"Expected 'anxiety' but got {result['dominant_emotion']}"
    assert result['dominant_emotion_score']>0, f"Expected a negative score for anxiety and stress, but got {result['dominant_emotion_score']}"
    assert result['dominant_category'] == 'stress', f"Expected 'stress' category, but got {result['dominant_category']}"

def test_generate_sentiment_mixedEmotion(analyser):
#tesing to check how the function handles sentances with multiple emotions
    input_text3 = "I feel happy but also sad"

    analyser.preprocess_text =  lambda text: ['i', 'feel', 'happy', 'but', 'also','sad']

    result = analyser.generate_sentiment(input_text3)

    assert result ['dominant_emotion'] in ['joy', 'sadness'], f"Expected either 'joy' or 'sadness', but got {result['dominant_emotion']}"
    assert result ['dominant_emotion_score'] > 0, f"Expected positive score, but got {result['dominant_emotion_score']}"

def test_generate_sentiment_neutral(analyser):
# checks how the function handles neutral sentences or words to make sure the output is neutral or with a score of 0.0
    input_text4 = "Today is just an ordinary day."

    analyser.preprocess_text = lambda text: ['today', 'is', 'just','an', 'ordinary','day']

    result = analyser.generate_sentiment(input_text4)

    assert result['dominant_emotion']=='neutral', f"Expected 'neutral', but got {result['dominant_emotion']}"
    assert result['dominant_emotion_score'] == 0.0, f"Expected score of 0.0, but got {result['dominant_emotion_score']}"
