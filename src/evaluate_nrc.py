import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from nrc_analyser import NRCAnalyser
import matplotlib.pyplot as plt
import seaborn as sns

df1 = pd.read_csv("/Users/panashekamuteku/Documents/Brunel OneDrive/OneDrive - Brunel University London/Year 3/FYP/DatabaseForFYP/GoEmotionsData/goemotions_1.csv")
df2 = pd.read_csv("/Users/panashekamuteku/Documents/Brunel OneDrive/OneDrive - Brunel University London/Year 3/FYP/DatabaseForFYP/GoEmotionsData/goemotions_2.csv")
df3 = pd.read_csv("/Users/panashekamuteku/Documents/Brunel OneDrive/OneDrive - Brunel University London/Year 3/FYP/DatabaseForFYP/GoEmotionsData/goemotions_3.csv")

df = pd.concat([df1,df2,df3], ignore_index=True)

nrc_analyser = NRCAnalyser()

emotion_labels = df.columns[9:]

#mapping emotions to goEmotions dataset to custom emotions and emotions in lexicon 
nrc_to_goemotions = {
    "fear":"fear",
    "joy":"joy",
    "trust":"neutral",
    "anger":"anger",
    "sadness":"sadness",
    "disgust":"disgust",
    "anxiety":"fear", #for evaluation purposes mapped anxiety/stress to fear for the purpose of this
    "stress":"fear"
}

evaluated_emotions = ['fear', 'joy', 'anger', 'sadness', 'disgust', 'neutral']

predictions = []
actual_labels = []

for _, row in df.iterrows():
    text = row['text']
    #extracting actual emotion from column == 1 
    actual = [emotion for emotion in emotion_labels if row[emotion] == 1]

    result = nrc_analyser.generate_sentiment(text)
    predicted_emotion = result["dominant_emotion"]

    mapped_emotion = nrc_to_goemotions.get(predicted_emotion,"neutral")

     # Debugging: print actual and predicted emotions for a few rows
    if _ < 5:  # Limit debug prints to first 5 rows for brevity
        print(f"Text: {text}")
        print(f"Actual Emotions: {actual}")
        print(f"Predicted Emotion: {predicted_emotion}")
        print(f"Mapped Emotion: {mapped_emotion}")

    predictions.append(mapped_emotion)
    actual_labels.append(actual)

y_true = []
y_pred = []

for actual, predicted in zip(actual_labels, predictions):
    if actual:  # skip rows with no actual emotion
        y_true.append(actual[0])
        y_pred.append(predicted)

print(f"\nLength of y_true: {len(y_true)}")
print(f"Length of y_pred: {len(y_pred)}")

# Generating the classification report and converting it to a dictionary
report_dict = classification_report(y_true, y_pred, labels=evaluated_emotions, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report_dict).transpose()
metrics_df = report_df.loc[evaluated_emotions, ['precision', 'recall', 'f1-score']]

print("\nPer-Emotion Precision, Recall, and F1-Score:")
print(metrics_df.round(3))

# Overall scores
overall_metrics = {}

# Check if the report contains accuracy and other overall metrics
if "accuracy" in report_dict:
    overall_metrics["accuracy"] = report_dict["accuracy"]

if "macro avg" in report_dict:
    overall_metrics["macro avg"] = {
        k: v for k, v in report_dict["macro avg"].items() if k in ['precision', 'recall', 'f1-score']
    }

if "weighted avg" in report_dict:
    overall_metrics["weighted avg"] = {
        k: v for k, v in report_dict["weighted avg"].items() if k in ['precision', 'recall', 'f1-score']
    }

# Convert the overall metrics into a DataFrame for easy viewing
overall_df = pd.DataFrame(overall_metrics).T
accuracy = overall_metrics.get('accuracy', None)
if accuracy is not None:
    print(f"Model Accuracy: {accuracy:.3f}")
else:
    print("Model Accuracy: N/A")

print(f"Model Precision: {overall_df.loc['macro avg', 'precision']:.3f}")
print(f"Model Recall: {overall_df.loc['macro avg', 'recall']:.3f}")
print(f"Model F1-Score: {overall_df.loc['macro avg', 'f1-score']:.3f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred, labels=evaluated_emotions)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=evaluated_emotions, yticklabels=evaluated_emotions)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()