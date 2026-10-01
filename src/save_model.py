import pandas as pd
import re
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


# ==============================
# 1. Load Dataset
# ==============================

data = pd.read_excel(
    "dataset/AI-Powered Chatbot.xlsx"
)


# ==============================
# 2. Clean Text
# ==============================

def clean_text(text):
    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


data["Clean Message"] = data["User Message"].apply(
    clean_text
)


# ==============================
# 3. Encode Intent
# ==============================

label_encoder = LabelEncoder()

data["Intent Encoded"] = label_encoder.fit_transform(
    data["Intent"]
)


# ==============================
# 4. Prepare Data
# ==============================

X = data["Clean Message"]

y = data["Intent Encoded"]


# ==============================
# 5. TF-IDF
# ==============================

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_tfidf = vectorizer.fit_transform(X)


# ==============================
# 6. Train Final SVM
# ==============================

model = LinearSVC()

model.fit(
    X_tfidf,
    y
)


# ==============================
# 7. Save Model
# ==============================

with open(
    "models/chatbot_model.pkl",
    "wb"
) as file:

    pickle.dump(
        model,
        file
    )


# ==============================
# 8. Save Vectorizer
# ==============================

with open(
    "models/tfidf_vectorizer.pkl",
    "wb"
) as file:

    pickle.dump(
        vectorizer,
        file
    )


# ==============================
# 9. Save Label Encoder
# ==============================

with open(
    "models/label_encoder.pkl",
    "wb"
) as file:

    pickle.dump(
        label_encoder,
        file
    )


print("================================")
print("Final model saved successfully!")
print("================================")

print("\nSaved files:")

print("models/chatbot_model.pkl")
print("models/tfidf_vectorizer.pkl")
print("models/label_encoder.pkl")

print("\nTraining samples:", len(X))

print(
    "Number of intents:",
    len(label_encoder.classes_)
)
