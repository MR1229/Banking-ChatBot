"""
Banking Support Chatbot
------------------------
An AI/ML-based banking FAQ assistant.

Pipeline: clean text -> TF-IDF vectorize -> Naive Bayes intent
classification -> confidence-checked response lookup -> Gradio chat UI.

Run with:  python banking_chatbot.py
"""

import os
import re

import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

import gradio as gr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH = "banking_support.csv"
MODEL_PATH = "banking_chatbot_model.pkl"
VECTORIZER_PATH = "banking_chatbot_vectorizer.pkl"

CONFIDENCE_THRESHOLD = 0.06
FALLBACK_MESSAGE = (
    "I'm sorry, I couldn't confidently understand that question. "
    "Could you try rephrasing it? You can also call our 24x7 helpline "
    "at 1800-419-1234 for further assistance."
)

BOT_NAME = "Banking Support Assistant"

nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

def clean_text(text):
    """Lowercase, strip punctuation/numbers, and remove stopwords."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [word for word in text.split() if word not in STOPWORDS]
    return " ".join(words)


# ---------------------------------------------------------------------------
# Load data and build features
# ---------------------------------------------------------------------------

def load_dataset(path=DATA_PATH):
    df = pd.read_csv(path)
    df["clean_question"] = df["question"].apply(clean_text)
    return df


def build_vectorizer():
    # unigrams + bigrams give the model short phrases like "block card"
    # in addition to single words, which helps with short queries
    return TfidfVectorizer(max_features=1500, ngram_range=(1, 2), min_df=1)


# ---------------------------------------------------------------------------
# Train, evaluate, and persist the model
# ---------------------------------------------------------------------------

def train_and_save_model(df):
    vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(df["clean_question"])
    y = df["category"]

    # Honest train/test split purely to report a realistic accuracy figure
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    eval_model = MultinomialNB()
    eval_model.fit(X_train, y_train)
    test_accuracy = accuracy_score(y_test, eval_model.predict(X_test))
    print(f"Held-out test accuracy: {test_accuracy * 100:.1f}% "
          f"({len(y_test)} unseen questions, {y.nunique()} categories)")

    # Final model used in production is retrained on the FULL dataset,
    # so every category benefits from all of its available examples
    model = MultinomialNB()
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Vectorizer saved to {VECTORIZER_PATH}")

    return model, vectorizer


def load_or_train_model(df):
    """Reuse a saved model/vectorizer if present, otherwise train fresh ones."""
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        print("Loading saved model and vectorizer...")
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    print("No saved model found. Training a new one...")
    return train_and_save_model(df)


# ---------------------------------------------------------------------------
# Predict intent and generate a response
# ---------------------------------------------------------------------------

def get_response(question, model, vectorizer, df, threshold=CONFIDENCE_THRESHOLD):
    """
    Clean -> vectorize -> predict category -> confidence check -> response lookup.
    Returns (response_text, predicted_category, confidence).
    """
    cleaned = clean_text(question)
    vector = vectorizer.transform([cleaned])

    # The question shares no words at all with the training vocabulary
    if vector.sum() == 0:
        return FALLBACK_MESSAGE, None, 0.0

    probabilities = model.predict_proba(vector)[0]
    best_index = probabilities.argmax()
    confidence = probabilities[best_index]
    predicted_category = model.classes_[best_index]

    # The model recognized some words, but its best guess is still weak
    if confidence < threshold:
        return FALLBACK_MESSAGE, predicted_category, confidence

    response = df.loc[df["category"] == predicted_category, "response"].iloc[0]
    return response, predicted_category, confidence


# ---------------------------------------------------------------------------
# Gradio chat interface
# ---------------------------------------------------------------------------

def build_chat_function(model, vectorizer, df):
    def chat_fn(message, history):
        response, _category, _confidence = get_response(message, model, vectorizer, df)
        return response
    return chat_fn


def launch_app(model, vectorizer, df):
    chat_fn = build_chat_function(model, vectorizer, df)

    demo = gr.ChatInterface(
        fn=chat_fn,
        title=f"🏦 {BOT_NAME}",
        description=(
            "Ask me about accounts, cards, loans, UPI, KYC, net banking, "
            "and other everyday banking topics. I'm an AI assistant trained "
            "on common support questions — I can't access live account data."
        ),
        textbox=gr.Textbox(
            placeholder="e.g. How do I check my account balance?",
            scale=7,
        ),
        examples=[
            "How do I check my account balance?",
            "My debit card got blocked, what should I do?",
            "What documents are needed to open an account?",
            "How can I transfer money using UPI?",
            "What is the interest rate on a personal loan?",
        ],
    )
    demo.launch(share=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    df = load_dataset()
    model, vectorizer = load_or_train_model(df)
    launch_app(model, vectorizer, df)


if __name__ == "__main__":
    main()
