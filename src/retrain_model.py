import pandas as pd
import re
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


# ==============================
# 1. Load Augmented Dataset
# ==============================

data = pd.read_excel(
    "dataset/training_data.xlsx"
)

print("Dataset loaded:", len(data), "rows")


# ==============================
# 2. NLP Text Preprocessing
# ==============================

def clean_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    # Keep letters and spaces
    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


data["Clean_Message"] = data["User Message"].apply(
    clean_text
)


# ==============================
# 3. Encode Intent
# ==============================

label_encoder = LabelEncoder()

data["Intent_Encoded"] = label_encoder.fit_transform(
    data["Intent"]
)


# ==============================
# 4. Input and Target
# ==============================

X = data["Clean_Message"]

y = data["Intent_Encoded"]


# ==============================
# 5. TF-IDF
# ==============================

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_tfidf = vectorizer.fit_transform(X)


print("TF-IDF shape:", X_tfidf.shape)


# ==============================
# 6. Train Linear SVM
# ==============================

model = LinearSVC(
    C=1.5
)

model.fit(
    X_tfidf,
    y
)

print("Model training completed!")


# ==============================
# 7. Save Improved Model
# ==============================

with open(
    "models/chatbot_model_improved.pkl",
    "wb"
) as file:
    pickle.dump(model, file)


with open(
    "models/tfidf_vectorizer_improved.pkl",
    "wb"
) as file:
    pickle.dump(vectorizer, file)


with open(
    "models/label_encoder_improved.pkl",
    "wb"
) as file:
    pickle.dump(label_encoder, file)


print("\n================================")
print("Improved model saved!")
print("================================")

print(
    "models/chatbot_model_improved.pkl"
)

print(
    "models/tfidf_vectorizer_improved.pkl"
)

print(
    "models/label_encoder_improved.pkl"
)

print("\nTraining samples:", len(X))
print(
    "Number of intents:",
    len(label_encoder.classes_)
)