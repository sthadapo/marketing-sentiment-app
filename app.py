import streamlit as st
import joblib
import re

# -------------------------
# Load model + vectorizer
# -------------------------
tfidf = joblib.load("tfidf_vectorizer.joblib")
model = joblib.load("sentiment_model.joblib")

# -------------------------
# Text cleaning function
# -------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -------------------------
# Title / UI
# -------------------------
st.title("Marketing Tweet Sentiment + Persona Classifier 🔍📊")
st.write("Analyze a marketing-related tweet and classify its sentiment as **Negative**, **Neutral**, or **Positive**.")

user_input = st.text_area("Enter a tweet or message:", height=150)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter text to analyze.")
    else:
        cleaned = clean_text(user_input)
        vectorized = tfidf.transform([cleaned])
        pred = model.predict(vectorized)[0]

        label_map = {
    0: ("Negative 😞", "This message reflects customer dissatisfaction."),
    4: ("Positive 😄", "This message reflects customer satisfaction.")
}

sentiment_label, explanation = label_map.get(
    int(pred),
    ("Unknown 🤔", "The model predicted a label I didn't expect.")
)

        st.subheader(f"Sentiment: **{sentiment_label}**")
        st.write(explanation)
