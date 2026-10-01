import pandas as pd
import pickle
import re


# ==============================
# 1. Load Saved Model
# ==============================

with open("models/chatbot_model.pkl", "rb") as file:
    model = pickle.load(file)


# ==============================
# 2. Load TF-IDF Vectorizer
# ==============================

with open("models/tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


# ==============================
# 3. Load Label Encoder
# ==============================

with open("models/label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)


# ==============================
# 4. Load Dataset
# ==============================

data = pd.read_excel(
    "dataset/AI-Powered Chatbot.xlsx"
)


# ==============================
# 5. Text Cleaning
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


# ==============================
# 6. Get Bot Response
# ==============================

def get_response(user_message):

    # Clean message
    cleaned_message = clean_text(user_message)

    # Convert message to TF-IDF
    message_vector = vectorizer.transform(
        [cleaned_message]
    )

    # Predict intent
    predicted_label = model.predict(
        message_vector
    )[0]

    # Convert number back to intent
    predicted_intent = label_encoder.inverse_transform(
        [predicted_label]
    )[0]

    # Find matching response
    matching_rows = data[
        data["Intent"] == predicted_intent
    ]

    if len(matching_rows) > 0:

        response = matching_rows.iloc[0]["Bot Response"]

    else:

        response = "Sorry, I don't know how to answer that."


    return predicted_intent, response


# ==============================
# 7. Chatbot
# ==============================

print("================================")
print("      UNIVERSITY AI CHATBOT")
print("================================")
print("Type 'exit' to stop the chatbot.\n")


while True:

    user_message = input("You: ")

    if user_message.lower() == "exit":
        print("Bot: Goodbye! 👋")
        break

    intent, response = get_response(
        user_message
    )

    print("\nPredicted Intent:", intent)
    print("Bot:", response)
    print()