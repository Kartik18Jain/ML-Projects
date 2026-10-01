import pickle
import re


# Load improved model
with open("models/chatbot_model_improved.pkl", "rb") as file:
    model = pickle.load(file)

# Load improved vectorizer
with open("models/tfidf_vectorizer_improved.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# Load improved label encoder
with open("models/label_encoder_improved.pkl", "rb") as file:
    label_encoder = pickle.load(file)


# Text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Prediction function
def predict_intent(message):

    cleaned = clean_text(message)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    intent = label_encoder.inverse_transform(
        [prediction]
    )[0]

    return intent


# Test questions
questions = [
    "How can I appeal a grade?",
    "I want to dispute my marks",
    "My grade is incorrect",
    "How do I register for a course?",
    "When are my exams?",
    "How can I apply for financial aid?"
]


print("===== IMPROVED MODEL TEST =====\n")

for question in questions:

    intent = predict_intent(question)

    print("Question:", question)
    print("Predicted Intent:", intent)
    print("-" * 50)