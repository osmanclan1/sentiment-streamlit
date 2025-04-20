import streamlit as st
import joblib

# === Load model and vectorizer ===
model = joblib.load("logistic_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# === Streamlit UI ===
st.title("yaqubs Sentiment Classifier")
st.subheader("TF-IDF + Logistic Regression")

# Text input
user_input = st.text_area("Enter a movie review:")

# Predict button
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Vectorize input
        input_vec = vectorizer.transform([user_input])

        # Predict
        prediction = model.predict(input_vec)[0]
        proba = model.predict_proba(input_vec)[0]

        # Interpret result
        sentiment = "Positive 😄" if prediction == 1 else "Negative 😡"
        confidence = proba[prediction] * 100

        st.markdown(f"### Sentiment: **{sentiment}**")
        st.markdown(f"Confidence: **{confidence:.2f}%**")
