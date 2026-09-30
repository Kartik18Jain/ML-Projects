import pandas as pd
import re

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import seaborn as sns
import matplotlib.pyplot as plt


# Load dataset
data = pd.read_excel("dataset/AI-Powered Chatbot.xlsx")


# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


data["Clean Message"] = data["User Message"].apply(clean_text)


# Encode intents
label_encoder = LabelEncoder()

data["Intent Encoded"] = label_encoder.fit_transform(
    data["Intent"]
)


# Handle rare intents
intent_counts = data["Intent"].value_counts()

rare_data = data[
    data["Intent"].map(intent_counts) == 1
]

normal_data = data[
    data["Intent"].map(intent_counts) > 1
]


# Train-test split
normal_train, normal_test = train_test_split(
    normal_data,
    test_size=0.20,
    random_state=42,
    stratify=normal_data["Intent"]
)

train_data = pd.concat(
    [normal_train, rare_data],
    ignore_index=True
)

test_data = normal_test.reset_index(drop=True)


X_train = train_data["Clean Message"]
y_train = train_data["Intent Encoded"]

X_test = test_data["Clean Message"]
y_test = test_data["Intent Encoded"]


# TF-IDF
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Train SVM
model = LinearSVC()

model.fit(
    X_train_tfidf,
    y_train
)


# Predictions
predictions = model.predict(
    X_test_tfidf
)


# Accuracy
accuracy = accuracy_score(
    y_test,
    predictions
)

print("===== MODEL EVALUATION =====")
print("Accuracy:", round(accuracy, 4))


# Classification report
print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        predictions,
        labels=sorted(y_test.unique()),
        target_names=label_encoder.inverse_transform(
            sorted(y_test.unique())
        ),
        zero_division=0
    )
)


# Confusion Matrix
cm = confusion_matrix(
    y_test,
    predictions,
    labels=sorted(y_test.unique())
)


plt.figure(figsize=(14, 10))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.inverse_transform(
        sorted(y_test.unique())
    ),
    yticklabels=label_encoder.inverse_transform(
        sorted(y_test.unique())
    )
)

plt.xlabel("Predicted Intent")
plt.ylabel("Actual Intent")
plt.title("SVM Confusion Matrix")

plt.xticks(rotation=90)
plt.yticks(rotation=0)

plt.tight_layout()

plt.show()