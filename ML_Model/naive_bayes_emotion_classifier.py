import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import joblib
from scipy.sparse import csr_matrix
from imblearn.under_sampling import ClusterCentroids


# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Load data from SQLite database
db_path = "/Users/panashekamuteku/Documents/Brunel OneDrive/OneDrive - Brunel University London/Year 3/FYP/DatabaseForFYP/DatabaseForFYP.db"
engine = create_engine(f"sqlite:///{db_path}")

try:
    query = "SELECT text, emotion FROM go_emotions"
    df = pd.read_sql(query, con=engine)
    print("Cleaned data loaded successfully from the database!")
except Exception as e:
    print(f"Error loading cleaned data from database: {e}")
    exit()

# Apply text cleaning
df["clean_text"] = df["text"].apply(clean_text)

# Features and labels
X = df["clean_text"]
y = df["emotion"]

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization (Updated n-gram range to 1-3)
vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=10, max_df=0.5, sublinear_tf=True)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Convert TF-IDF to compressed sparse format (Speeds up processing & saving)
X_train_tfidf = csr_matrix(X_train_tfidf)
X_test_tfidf = csr_matrix(X_test_tfidf)

print("right before cluster centroids...")
undersampler = ClusterCentroids(random_state=42)  # Undersampling technique
X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train_tfidf, y_train)
print("cluster centroid done")

# Use a fresh model for grid search (Efficient Parallel Processing)
fresh_model = ComplementNB()

param_grid = {
    'alpha': [10.0, 20.0, 50.0],  # Reduced range for efficiency
    'fit_prior': [True, False]
}

grid_search = GridSearchCV(estimator=fresh_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_tfidf, y_train)

# Best model selection
best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# Make Predictions
y_pred = best_model.predict(X_test_tfidf)

# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print Accuracy Metrics
print(f"Model Accuracy: {accuracy:.2f}")
print(f"Model Precision: {precision:.2f}")
print(f"Model Recall: {recall:.2f}")
print(f"Model F1-Score: {f1:.2f}")

# Confusion Matrix Visualization 

plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", 
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()


# Save models without threads (one by one)
print("Saving model...")
joblib.dump(best_model, 'naive_bayes_model.pkl', compress=3)

print("Saving vectorizer...")
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl', compress=3)

print("Saving label encoder...")
joblib.dump(label_encoder, 'label_encoder.pkl', compress=3)

print("\nModel, Vectorizer, and Label Encoder saving completed!")
