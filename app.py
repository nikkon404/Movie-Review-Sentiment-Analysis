import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# App title and description with emoji
st.set_page_config(page_title="IMDB Sentiment Analysis", page_icon="🎬")
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write(
    "Welcome! Enter a movie review below and let our AI tell you the sentiment. 😊"
)


# Load word index and reverse index
@st.cache_resource
def get_word_index():
    return imdb.get_word_index()


word_index = get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


# Load the trained model
@st.cache_resource
def get_model():
    return load_model("simple_rnn_model_imdb.h5")


model = get_model()


def preprocess_text(text, maxlen=500):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=maxlen)
    return padded_review


def predict_sentiment(review):
    processed_review = preprocess_text(review)
    prediction = model.predict(processed_review)
    score = prediction[0][0]
    if score > 0.6:
        sentiment = "Positive 😊"
        icon = "✅"
    elif score < 0.4:
        sentiment = "Negative 😞"
        icon = "❌"
    else:
        sentiment = "Neutral 😐"
        icon = "⚪"
    return sentiment, score, icon


# Streamlit UI
st.write("👇 Type your review and click Predict!")

user_review = st.text_area("✍️ Movie Review", "")

if st.button("🔍 Predict Sentiment"):
    if user_review.strip() == "":
        st.warning("⚠️ Please enter a review.")
    else:
        with st.spinner("Analyzing sentiment... 🔄"):
            sentiment, score, icon = predict_sentiment(user_review)
        st.success(f"{icon} **Predicted Sentiment:** {sentiment}")
        st.info(f"📊 **Confidence Score:** `{score:.2f}`")
