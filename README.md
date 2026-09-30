# 🎓 AI-Powered University Chatbot

An NLP-based chatbot designed to answer common university student queries related to courses, examinations, academic policies, financial aid, campus facilities, student services, and other university-related topics.

## 🚀 Features

- Intent classification using Machine Learning
- TF-IDF based text vectorization
- Linear SVM for intent prediction
- Similarity-based response retrieval
- Handles common student queries
- Fallback response for unknown questions
- Streamlit web interface
- Local machine learning implementation
- No paid API required

## 🧠 Technologies Used

- Python
- Pandas
- Scikit-learn
- NumPy
- Streamlit
- OpenPyXL

## 📊 Dataset

The chatbot uses a university student-support dataset containing:

- User messages
- Intent labels
- Topics
- Sentiment information
- Bot responses

The training dataset contains 217 examples across 26 different intents after adding additional training examples for certain intents.

## 🔄 Project Workflow

1. Dataset Loading
2. Exploratory Data Analysis
3. Text Preprocessing
4. Intent Label Encoding
5. Train-Test Split
6. TF-IDF Vectorization
7. Model Training and Comparison
8. Model Evaluation
9. Model Saving using Pickle
10. Chatbot Response Logic
11. Streamlit Deployment

## 🤖 Machine Learning Model

Different classification algorithms were compared during development:

- Logistic Regression
- Naive Bayes
- Linear Support Vector Machine (Linear SVM)

Linear SVM was selected as the final intent-classification model.

TF-IDF with unigram and bigram features is used to convert user messages into numerical features.

## 💬 Response Generation

After predicting the user's intent, the chatbot searches the training data for responses belonging to that intent.

Cosine similarity is then used to find the response most similar to the user's question.

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Kartik18Jain/ML-Projects.git
cd ML-Projects