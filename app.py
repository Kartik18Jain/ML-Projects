import streamlit as st

import pandas as pd

import pickle

import re

from sklearn.metrics.pairwise import cosine_similarity





# ============================================================

# PAGE CONFIGURATION

# ============================================================



st.set_page_config(

    page_title="UniAssist AI",

    page_icon="🎓",

    layout="wide",

    initial_sidebar_state="expanded"

)





# ============================================================

# CUSTOM CSS

# ============================================================



st.markdown("""

<style>



.stApp {

    background: #f5f7fb;

}



.block-container {

    padding-top: 1.5rem;

    padding-bottom: 4rem;

    max-width: 1150px;

}





/* ================= HERO ================= */



.hero {

    background: linear-gradient(135deg, #172554 0%, #2563eb 100%);

    padding: 32px 36px;

    border-radius: 20px;

    color: white;

    margin-bottom: 24px;

    box-shadow: 0 10px 30px rgba(37, 99, 235, 0.20);

}



.hero-title {

    font-size: 38px;

    font-weight: 750;

    margin: 0;

    color: white;

    line-height: 1.2;

}



.hero-subtitle {

    font-size: 17px;

    margin-top: 10px;

    margin-bottom: 0;

    color: #dbeafe;

}



.hero-badge {

    display: inline-block;

    margin-top: 18px;

    padding: 6px 12px;

    border-radius: 20px;

    background: rgba(255, 255, 255, 0.14);

    color: #e0f2fe;

    font-size: 13px;

}





/* ================= WELCOME CARD ================= */



.welcome-card {

    background: white;

    padding: 25px 28px;

    border-radius: 18px;

    border: 1px solid #e2e8f0;

    margin-bottom: 20px;

    box-shadow: 0 4px 15px rgba(15, 23, 42, 0.04);

}



.welcome-title {

    color: #172554;

    font-size: 23px;

    font-weight: 700;

    margin: 0 0 8px 0;

}



.welcome-text {

    color: #64748b;

    font-size: 15px;

    line-height: 1.7;

    margin: 0;

}





/* ================= SIDEBAR ================= */



section[data-testid="stSidebar"] {

    background: #ffffff;

    border-right: 1px solid #e2e8f0;

}



.sidebar-title {

    color: #172554;

    font-size: 23px;

    font-weight: 750;

    margin-bottom: 3px;

}



.sidebar-subtitle {

    color: #64748b;

    font-size: 13px;

    margin-bottom: 18px;

}



.sidebar-section {

    color: #172554;

    font-size: 16px;

    font-weight: 650;

    margin-top: 8px;

    margin-bottom: 10px;

}





/* ================= BUTTONS ================= */



.stButton > button {

    width: 100%;

    border-radius: 10px;

    border: 1px solid #dbe3ef;

    background: white;

    color: #1e3a8a;

    font-weight: 500;

    min-height: 42px;

    transition: all 0.2s ease;

}



.stButton > button:hover {

    border-color: #2563eb;

    color: #2563eb;

    background: #eff6ff;

    transform: translateY(-1px);

}





/* ================= CHAT ================= */



[data-testid="stChatMessage"] {

    border-radius: 16px;

    padding: 12px 16px;

    margin-bottom: 12px;

}



[data-testid="stChatInput"] {

    border-radius: 14px;

}





/* ================= INFO CARDS ================= */



.mini-card {

    background: #eff6ff;

    border: 1px solid #dbeafe;

    border-radius: 12px;

    padding: 12px 14px;

    margin-top: 10px;

    color: #1e3a8a;

    font-size: 13px;

}





/* ================= FOOTER ================= */



.footer {

    text-align: center;

    color: #94a3b8;

    font-size: 12px;

    margin-top: 35px;

    padding-top: 18px;

    border-top: 1px solid #e2e8f0;

    line-height: 1.7;

}



</style>

""", unsafe_allow_html=True)





# ============================================================

# LOAD MODEL

# ============================================================



with open("models/chatbot_model.pkl", "rb") as file:

    model = pickle.load(file)



with open("models/tfidf_vectorizer.pkl", "rb") as file:

    vectorizer = pickle.load(file)



with open("models/label_encoder.pkl", "rb") as file:

    label_encoder = pickle.load(file)



data = pd.read_excel("dataset/training_data.xlsx")





# ============================================================

# TEXT CLEANING

# ============================================================



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





# ============================================================

# CHATBOT RESPONSE

# ============================================================



def get_response(user_message):
    cleaned_message = clean_text(user_message)

    generic_inputs = {
    "same", "same same", "test", "testing", "asdf", "ok", "okay",
    "hmm", "hmmm", "nothing", "random", "xyz",
    "is", "are", "am", "the", "a", "an", "hello", "hi", "hey"
}
    sorry_message = (
    "I am not able to find a suitable answer to that question "
    "in the available university information. Please try asking "
    "a university-related question."
)

    if not cleaned_message or cleaned_message in generic_inputs:
        return sorry_message

    message_vector = vectorizer.transform([cleaned_message])

    if message_vector.nnz == 0:
        return sorry_message

    predicted_label = model.predict(message_vector)[0]
    predicted_intent = label_encoder.inverse_transform([predicted_label])[0]

    matching_rows = data[data["Intent"] == predicted_intent].copy()

    if len(matching_rows) == 0:
        return sorry_message

    matching_vectors = vectorizer.transform(
        matching_rows["User Message"].apply(clean_text)
    )
    similarities = cosine_similarity(message_vector, matching_vectors)[0]
    best_index = similarities.argmax()
    best_similarity = similarities[best_index]

    all_vectors = vectorizer.transform(
        data["User Message"].apply(clean_text)
    )
    overall_similarity = cosine_similarity(message_vector, all_vectors)[0].max()

    response = matching_rows.iloc[best_index]["Bot Response"]

    # Reject unrelated questions instead of forcing them into an intent.
    SIMILARITY_THRESHOLD = 0.12

    if best_similarity < SIMILARITY_THRESHOLD or overall_similarity < SIMILARITY_THRESHOLD:
        return sorry_message

    return response


# SESSION STATE

# ============================================================



if "messages" not in st.session_state:

    st.session_state.messages = []





# ============================================================

# SIDEBAR

# ============================================================



with st.sidebar:



    st.markdown(

        '<div class="sidebar-title">🎓 UniAssist AI</div>',

        unsafe_allow_html=True

    )



    st.markdown(

        '<div class="sidebar-subtitle">University Student Support Assistant</div>',

        unsafe_allow_html=True

    )



    st.divider()



    st.markdown(

        '<div class="sidebar-section">💡 Try asking</div>',

        unsafe_allow_html=True

    )



    example_questions = [

        "How do I register for a course?",

        "When are my exams?",

        "How can I appeal a grade?",

        "How can I apply for financial aid?",

        "What facilities are available on campus?",

        "How can I get academic advising?"

    ]



    for question in example_questions:



        if st.button(

            question,

            use_container_width=True

        ):



            st.session_state.messages.append(

                {

                    "role": "user",

                    "content": question

                }

            )



            response = get_response(question)



            st.session_state.messages.append(

                {

                    "role": "assistant",

                    "content": response

                }

            )



            st.rerun()



    st.divider()



    if st.button(

        "🧹 Clear Conversation",

        use_container_width=True

    ):



        st.session_state.messages = []



        st.rerun()



    st.divider()



    st.markdown(

        """

<div class="mini-card">

    🤖 <b>AI Model</b><br>

    TF-IDF + Linear SVM

</div>

""",

        unsafe_allow_html=True

    )



    st.markdown(

        """

<div class="mini-card">

    📚 <b>Knowledge Base</b><br>

    University Student Support Dataset

</div>

""",

        unsafe_allow_html=True

    )







# ============================================================

# MAIN HERO HEADER

# ============================================================



st.markdown(

    '<div class="hero">'

    '<div class="hero-title">🎓 UniAssist AI</div>'

    '<div class="hero-subtitle">Your intelligent university student support assistant</div>'

    '<div class="hero-badge">✨ Ask questions about university services, academics and more</div>'

    '</div>',

    unsafe_allow_html=True

)





# ============================================================

# WELCOME MESSAGE

# ============================================================



if len(st.session_state.messages) == 0:



    st.markdown(

        '<div class="welcome-card">'



        '<div class="welcome-title">'

        '👋 Welcome to UniAssist!'

        '</div>'



    '<div class="welcome-text">'

    'Ask me about courses, examinations, academic policies,'

    'financial aid, campus facilities, student services,'

    'study abroad programs and more.'

    '</div>'



'</div>'

,

        unsafe_allow_html=True

    )





# ============================================================

# DISPLAY CHAT HISTORY

# ============================================================



for message in st.session_state.messages:



    with st.chat_message(message["role"]):



        st.write(

            message["content"]

        )





# ============================================================

# CHAT INPUT

# ============================================================



user_message = st.chat_input(

    "Ask your university question..."

)





if user_message:



    st.session_state.messages.append(

        {

            "role": "user",

            "content": user_message

        }

    )



    with st.chat_message("user"):



        st.write(user_message)



    response = get_response(

        user_message

    )



    st.session_state.messages.append(

        {

            "role": "assistant",

            "content": response

        }

    )



    with st.chat_message("assistant"):



        st.write(response)





# ============================================================

# FOOTER

# ============================================================



st.markdown(

    '<div class="footer">'

    '<b>UniAssist AI</b> • P_200 University Student Support Project'

    '<br>'

    'NLP-based intent classification using TF-IDF and Linear SVM'

    '</div>',

    unsafe_allow_html=True

)