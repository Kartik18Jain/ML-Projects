import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder


# Load dataset
data = pd.read_excel("dataset/AI-Powered Chatbot.xlsx")


# Text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


data["Clean Message"] = data["User Message"].apply(clean_text)


# Create label encoder
label_encoder = LabelEncoder()

# Convert Intent into numbers
data["Intent Encoded"] = label_encoder.fit_transform(data["Intent"])


# Display mapping
print("===== INTENT → ENCODED LABEL =====")

for intent, label in zip(
    label_encoder.classes_,
    range(len(label_encoder.classes_))
):
    print(f"{intent} → {label}")


print("\n===== SAMPLE DATA =====")
print(
    data[["User Message", "Intent", "Intent Encoded"]]
    .head(10)
    .to_string(index=False)
)