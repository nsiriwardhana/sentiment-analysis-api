import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model.pkl")

pipeline = joblib.load(MODEL_PATH)

vectorizer = pipeline["vectorizer"]
model = pipeline["model"]

def predict_sentiment(text: str):

    text_tfidf = vectorizer.transform([text])

    prediction = model.predict(text_tfidf)[0]

    probabilities = model.predict_proba(text_tfidf)[0]

    sentiment = "positive" if prediction == 1 else "negative"

    confidence_score = round(float(probabilities[prediction]), 2)

    return sentiment, confidence_score