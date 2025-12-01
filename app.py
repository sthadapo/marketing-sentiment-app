import streamlit as st
import joblib

# Load model + vectorizer
model = joblib.load("sentiment_model.joblib")
tfidf = joblib.load("tfidf_vectorizer.joblib")

st.title("Marketing Tweet Sentiment Analyzer")
st.write("This app predicts if a marketing-related tweet is positive or negative.")

user_text = st.text_area("Enter a tweet or message:")

if st.button("Analyze"):
    if user_text.strip() == "":
        st.write("Please type something first.")
    else:
        vect = tfidf.transform([user_text])
        pred = model.predict(vect)[0]
        sentiment = "Positive 😊" if pred == 1 else "Negative 😕"
        st.subheader(f"Sentiment: {sentiment}")
