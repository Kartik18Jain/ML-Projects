import pandas as pd
import re

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score, f1_score


# ==============================
# 1. Load Dataset
# ==============================

data = pd.read_excel("dataset/AI-Powered Chatbot.xlsx")


# ==============================
# 2. Clean Text
# ==============================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


data["Clean Message"] = data["User Message"].apply(clean_text)


# ==============================
# 3. Encode Intent
# ==============================

label_encoder = LabelEncoder()

data["Intent Encoded"] = label_encoder.fit_transform(
    data["Intent"]
)


# ==============================
# 4. Handle Rare Intents
# ==============================

intent_counts = data["Intent"].value_counts()

rare_data = data[
    data["Intent"].map(intent_counts) == 1
]

normal_data = data[
    data["Intent"].map(intent_counts) > 1
]


# ==============================
# 5. Train-Test Split
# ==============================

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


# ==============================
# 6. TF-IDF
# ==============================

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)


# ==============================
# 7. Define Models
# ==============================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000
        ),

    "Naive Bayes":
        MultinomialNB(),

    "Linear SVM":
        LinearSVC()
}


# ==============================
# 8. Train & Compare
# ==============================

print("===== MODEL COMPARISON =====\n")

for name, model in models.items():

    # Train
    model.fit(
        X_train_tfidf,
        y_train
    )

    # Predict
    predictions = model.predict(
        X_test_tfidf
    )

    # Metrics
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    print(name)
    print("Accuracy:", round(accuracy, 4))
    print("F1 Score:", round(f1, 4))
    print("-" * 30)