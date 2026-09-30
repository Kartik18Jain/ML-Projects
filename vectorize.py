import pandas as pd
import re

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


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
# 4. Separate Rare Intents
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


# Put one-example intents directly
# into the training set
train_data = pd.concat(
    [normal_train, rare_data],
    ignore_index=True
)

test_data = normal_test.reset_index(drop=True)


# ==============================
# 6. Features and Target
# ==============================

X_train = train_data["Clean Message"]
y_train = train_data["Intent Encoded"]

X_test = test_data["Clean Message"]
y_test = test_data["Intent Encoded"]


# ==============================
# 7. TF-IDF
# ==============================

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)


X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)


# ==============================
# 8. Display Results
# ==============================

print("===== TRAIN-TEST SPLIT =====")

print("Total samples:", len(data))

print("Training samples:", len(X_train))

print("Testing samples:", len(X_test))


print("\n===== TF-IDF =====")

print("Training matrix shape:", X_train_tfidf.shape)

print("Testing matrix shape:", X_test_tfidf.shape)

print(
    "\nNumber of features:",
    len(vectorizer.get_feature_names_out())
)


print("\n===== RARE INTENTS =====")

print(
    rare_data["Intent"].value_counts()
)


print("\n===== SAMPLE FEATURES =====")

print(
    vectorizer.get_feature_names_out()[:20]
)