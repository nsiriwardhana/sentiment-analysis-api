import joblib
import os
import re
import sys

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model.pkl")

# Load the model with error handling
try:
    pipeline = joblib.load(MODEL_PATH)
    vectorizer = pipeline["vectorizer"]
    model = pipeline["model"]
    print("✓ Model loaded successfully")
except FileNotFoundError:
    print(f"ERROR: Model file not found at {MODEL_PATH}")
    print("Please run 'python train.py' to train and save the model first.")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Failed to load model: {e}")
    sys.exit(1)

def clean_text(text):
    """Clean text using the same preprocessing as training."""
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove special characters (keep only letters and spaces)
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_sentiment(text: str):
    """
    Predict sentiment of the given text.
    
    Supports three sentiment classes:
    - positive: High positive probability
    - negative: High negative probability  
    - neutral: Probabilities are close (difference < 0.2)
    
    Args:
        text: Input text to analyze
        
    Returns:
        tuple: (sentiment, confidence_score)
            sentiment: 'positive', 'negative', or 'neutral'
            confidence_score: float between 0 and 1
        
    Raises:
        ValueError: If text is empty or too long
    """
    # Input validation
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    
    if len(text) > 10000:
        raise ValueError("Text is too long (max 10000 characters)")
    
    # Clean the input text first
    text_cleaned = clean_text(text)
    
    # Check if cleaning resulted in empty text
    if not text_cleaned:
        raise ValueError("Text contains no valid content after preprocessing")
    
    text_tfidf = vectorizer.transform([text_cleaned])

    # Get probabilities for both classes
    probabilities = model.predict_proba(text_tfidf)[0]
    
    # Extract probabilities for each class
    # Index 0 = negative, Index 1 = positive
    negative_probability = float(probabilities[0])
    positive_probability = float(probabilities[1])
    
    # Calculate difference between probabilities
    probability_difference = abs(positive_probability - negative_probability)
    
    # Determine sentiment based on confidence threshold
    if probability_difference < 0.2:
        # Close to 0.5/0.5 split indicates neutral sentiment
        sentiment = "neutral"
        confidence_score = round(max(positive_probability, negative_probability), 2)
    elif positive_probability > negative_probability:
        sentiment = "positive"
        confidence_score = round(positive_probability, 2)
    else:
        sentiment = "negative"
        confidence_score = round(negative_probability, 2)

    return sentiment, confidence_score