import pandas as pd
import re

# Load dataset
data = pd.read_excel("dataset/AI-Powered Chatbot.xlsx")


# Text cleaning function
def clean_text(text):
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Apply cleaning
data["Clean Message"] = data["User Message"].apply(clean_text)


# Display original and cleaned messages
print("===== ORIGINAL vs CLEANED =====")

print(
    data[["User Message", "Clean Message"]].head(10).to_string(index=False)
)


# Check for empty messages after cleaning
print("\n===== EMPTY CLEANED MESSAGES =====")
print(
    "Empty messages:",
    (data["Clean Message"] == "").sum()
)