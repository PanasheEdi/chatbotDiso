from nrclex import NRCLex
from contractions import fix
import string
import numpy as np
from scipy.spatial.distance import cosine
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class NRCAnalyser:
    def __init__(self):
        """Initialize the NRCAnalyzer."""
        self.negation_words = {"not", "no", "never", "n't"}

        self.emotion_synonyms = {
            "fear": [
            "scared", "frightened", "alarmed", "dread", "paranoid", "jumpy", 
            "terrified", "horrified", "spooked", "freaked out", "timid", 
            "trembling", "distressed"
        ],

        "anxiety": [
            "anxious", "worried", "stressed", "overwhelmed", "panicked", 
            "restless", "uneasy", "tense", "on edge", "nervous", "frazzled", 
            "apprehensive", "distraught", "shaken", "agitated", "jittery", 
            "fidgety", "disconcerted", "troubled", "concerned", "doubtful", 
            "can't shake the feeling", "something bad is going to happen", 
            "waiting for the other shoe to drop", "mind is racing", "feel overwhelmed",
            "out of control", "can't stop thinking", "what might go wrong", 
            "everything feels out of control", "bloodpressure", "panic attack", 
            "feeling tense", "nervous breakdown", "anxious thoughts"
        ],
            "joy": [
                "happy", "excited", "joy", "delighted", "pleased", "cheerful", 
                "ecstatic", "thrilled", "content", "elated", "grateful", "euphoric", 
                "overjoyed", "jovial", "blissful", "radiant", "gleeful", "satisfied", 
                "cheery", "exhilarated", "positive", "light-hearted"
            ],
            "trust": [
                "trust", "reliable", "dependable", "faith", "secure", "confident", 
                "assured", "hopeful", "optimistic", "certain", "loyal", "steadfast", 
                "devoted", "true", "believable", "supportive", "reassured", "committed"
            ],
            "sadness": [
                "sad", "down", "depressed", "unhappy", "sorrowful", "miserable", 
                "despondent", "gloomy", "hopeless", "heartbroken", "melancholy", 
                "grief-stricken", "blue", "disheartened", "downcast", "morose", 
                "dismal", "low", "desolate", "forlorn", "grieved", "doleful"
            ],
            "anger": [
                "angry", "mad", "furious", "irritated", "frustrated", "resentful", 
                "agitated", "hostile", "enraged", "infuriated", "exasperated", 
                "annoyed", "outraged", "bitter", "fuming", "incensed", "wrathful", 
                "indignant", "livid", "heated", "raging", "irate", "vexed", "irritable"
            ],
            "disgust": [
                "disgusted", "gross", "sickened", "repulsed", "revolted", "appalled", 
                "nauseated", "distasteful", "loathsome", "displeased", "offended", 
                "repellent", "yucky", "grossed out", "abhorred", "horrified", "sickened"
            ]

        }

        self.emotion_categories = {
        "stress": ["fear", "anxiety"],
        "anger": ["anger"],
        "positive": ["joy", "trust", "anticipation"],
        "negative": ["sadness", "disgust"]
        }
        self.glove_embeddings = self.load_glove_embeddings('/Users/panashekamuteku/Documents/Brunel OneDrive/OneDrive - Brunel University London/Year 3/FYP/DatabaseForFYP/glove.6B/glove.6B.50d.txt')

    def load_glove_embeddings(self,file_path):
        embeddings = {}
        with open (file_path, 'r', encoding ="utf8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embeddings[word] = vector
            return embeddings

    def preprocess_text(self, text):
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))

        print (f"original text: {text}")

        text = fix(text) 
        print(f"After fix: {text}")

        if text is None:
            raise ValueError("text became none during preprocessing.")

        text = text.lower()
        print(f"after lowecasing: {text}")

        text = text.translate(str.maketrans('', '', string.punctuation)) 
        print(f"After punctuation removal: {text}")

        tokens = word_tokenize(text)  
        print(f"Tokens: {tokens}")
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        print(f"Final tokens: {tokens}")

        bigrams = list(nltk.bigrams(tokens))
        bigram_phrases = [" ".join(bigram) for bigram in bigrams]

        return tokens + bigram_phrases

    def find_closest_emotion_word(self, word, glove_embeddings):
        max_similarity = -1
        closest_emotion = None
        word_vector = glove_embeddings.get(word)

        if word_vector is None:
            return None

        for emotion, synonyms in self.emotion_synonyms.items():
            for synonym in synonyms:
                synonym_vector = glove_embeddings.get(synonym)
                if synonym_vector is not None:
                    similarity = 1 - cosine(word_vector, synonym_vector)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        closest_emotion = emotion

        return closest_emotion

    def categorize_emotions(self, emotion_scores):
        category_scores = {category: 0 for category in self.emotion_categories}

        for emotion, score in emotion_scores.items():
            for category, emotions in self.emotion_categories.items():
                if emotion in emotions:
                    category_scores[category] += score

        dominant_category = max(category_scores, key=category_scores.get)
        return dominant_category, category_scores[dominant_category]

    def normalize_scores(self, emotion_scores, emotional_word_count):
        if emotional_word_count > 0:
            normalization_factor = len(emotion_scores) / emotional_word_count
            emotion_scores = {emotion: score * normalization_factor for emotion, score in emotion_scores.items()}
        return emotion_scores

    def generate_sentiment(self, text):
        if not text:
            return {
                "dominant_emotion": 'neutral',
                "dominant_emotion_score": 0.0,
                "dominant_category": 'neutral',
                "dominant_category_score": 0.0,
                "all_emotions": {}
            }

        preprocessed_text = self.preprocess_text(text)
        unique_tokens = set(preprocessed_text)

        word_emotion_scores = {}
        emotional_word_count = 0
        negation_flag = False

        for word in unique_tokens:
            matched = False
            #direct amatch lexicon
            for emotion, synonyms in self.emotion_synonyms.items():
                if word in synonyms:
                    weight = 1.0
                    if emotion in["anxiety, fear"]:
                        weight = 1.5
                    word_emotion_scores[emotion] = word_emotion_scores.get(emotion, 0) + weight
                    matched = True
                    emotional_word_count += 1
                    break

            if not matched:
                closest_emotion = self.find_closest_emotion_word(word, self.glove_embeddings)
                if closest_emotion:
                    weight = 1.0
                    if closest_emotion in ["anxiety", "fear"]:
                        weight = 1.5
                    word_emotion_scores[closest_emotion] = word_emotion_scores.get(closest_emotion, 0) + 1
                    emotional_word_count += 1

            if word in self.negation_words:
                negation_flag = not negation_flag
                continue

            if negation_flag:
                for emotion in word_emotion_scores:
                    word_emotion_scores[emotion] = -word_emotion_scores.get(emotion)

        if not word_emotion_scores:
            return {
                "dominant_emotion": 'neutral',
                "dominant_emotion_score": 0.0,
                "dominant_category": 'neutral',
                "dominant_category_score": 0.0,
                "all_emotions": {}
            }

        emotion_scores = self.normalize_scores(word_emotion_scores, emotional_word_count)
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        dominant_score = emotion_scores[dominant_emotion]

        dominant_category, category_score = self.categorize_emotions(emotion_scores)

        return {
            "dominant_emotion": dominant_emotion,
            "dominant_emotion_score": dominant_score,
            "dominant_category": dominant_category,
            "dominant_category_score": category_score,
            "all_emotions": emotion_scores
        }