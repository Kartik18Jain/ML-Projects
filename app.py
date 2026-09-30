import streamlit as st
import pandas as pd
import pickle
import re

from sklearn.metrics.pairwise import cosine_similarity


# ==============================
# Load Model
# ==============================

with open("models/chatbot_model_improved.pkl", "rb") as file:
    model = pickle.load(file)

with open("models/tfidf_vectorizer_improved.pkl", "rb") as file:
    vectorizer = pickle.load(file)

with open("models/label_encoder_improved.pkl", "rb") as file:
    label_encoder = pickle.load(file)


# ==============================
# Load Dataset
# ==============================

data = pd.read_excel(
    "dataset/training_data.xlsx"
)


# ==============================
# Text Cleaning
# ==============================

def clean_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    # Keep only letters and spaces
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


# ==============================
# Get Response
# ==============================

def get_response(user_message):

    # Clean user message
    cleaned_message = clean_text(
        user_message
    )

    # Convert user message to TF-IDF
    message_vector = vectorizer.transform(
        [cleaned_message]
    )

    # Predict intent
    predicted_label = model.predict(
        message_vector
    )[0]

    predicted_intent = label_encoder.inverse_transform(
        [predicted_label]
    )[0]


    # ==============================
    # Find Matching Intent Examples
    # ==============================

    matching_rows = data[
        data["Intent"] == predicted_intent
    ].copy()


    # If no matching intent exists
    if len(matching_rows) == 0:

        return (
            "unknown",
            "I'm sorry, but I couldn't find "
            "a suitable answer."
        )


    # ==============================
    # Calculate Similarity
    # ==============================

    matching_vectors = vectorizer.transform(
        matching_rows["User Message"].apply(
            clean_text
        )
    )

    similarities = cosine_similarity(
        message_vector,
        matching_vectors
    )[0]


    # ==============================
    # Find Best Matching Question
    # ==============================

    best_index = similarities.argmax()

    best_similarity = similarities[best_index]

    response = matching_rows.iloc[
        best_index
    ]["Bot Response"]


    # ==============================
    # Fallback
    # ==============================

    # Only reject extremely unrelated questions.
    # A low threshold keeps spelling mistakes
    # and slightly different wording working.

    SIMILARITY_THRESHOLD = 0.05

    if best_similarity < SIMILARITY_THRESHOLD:

        return (
            "unknown",
            "I'm sorry, but I couldn't find "
            "a reliable answer to that question "
            "in the available university information."
        )


    return (
        predicted_intent,
        response
    )


# ==============================
# Streamlit UI
# ==============================

st.set_page_config(
    page_title="University AI Chatbot",
    page_icon="🎓"
)

st.title(
    "🎓 University AI Chatbot"
)

st.write(
    "Ask questions about university services, "
    "courses, exams, facilities, and more."
)


# ==============================
# Chat Input
# ==============================

user_message = st.chat_input(
    "Ask your question..."
)


if user_message:

    # User message
    st.chat_message(
        "user"
    ).write(
        user_message
    )


    # Get chatbot response
    intent, response = get_response(
        user_message
    )


    # Assistant response
    with st.chat_message(
        "assistant"
    ):

        st.write(
            response
        )

        st.caption(
            f"Detected intent: {intent}"
        )