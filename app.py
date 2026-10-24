import re
import joblib
import streamlit as st
from nltk.corpus import stopwords

# Page configuration
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered"
)

# Load trained model and TF-IDF vectorizer
@st.cache_resource
def load_models():
    model = joblib.load("models/spam_logistic_regression.pkl")
    vectorizer = joblib.load("models/spam_tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_models()


# Load English stopwords
@st.cache_resource
def load_stopwords():
    try:
        return set(stopwords.words("english"))
    except LookupError:
        import nltk
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))

STOP_WORDS = load_stopwords()


# Preprocess email text
def preprocess_email(email):
    email = email.lower()
    email = re.sub(r"\s+", " ", email).strip()
    tokens = re.findall(r"\b[a-z0-9]+\b", email)
    tokens = [word for word in tokens if word not in STOP_WORDS]
    return " ".join(tokens)


# App interface
st.title("📧 Email Spam Detector")
st.write(
    "Enter an email message below to check whether it is Spam or Ham (not spam)."
)

email_text = st.text_area(
    "Email Message",
    height=220,
    placeholder="Paste your email message here..."
)

if st.button("Check Email", type="primary", use_container_width=True):
    if not email_text.strip():
        st.warning("Please enter an email message first.")
    else:
        processed_text = preprocess_email(email_text)
        features = vectorizer.transform([processed_text])

        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        classes = list(model.classes_)
        ham_probability = probabilities[classes.index(0)] * 100
        spam_probability = probabilities[classes.index(1)] * 100

        if prediction == 1:
            st.error("🚨 Prediction: SPAM")
        else:
            st.success("✅ Prediction: HAM (Not Spam)")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Ham Probability", f"{ham_probability:.2f}%")

        with col2:
            st.metric("Spam Probability", f"{spam_probability:.2f}%")

        st.caption(
            "This is a machine-learning prediction, not a guarantee. "
            "Always review suspicious emails carefully."
        )