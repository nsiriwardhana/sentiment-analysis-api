import pandas as pd
import numpy as np
import re
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
df = pd.read_csv('data/IMDB_Dataset.csv')

print(df.head())
print(df.shape)
print(df.columns)
print(df["sentiment"].value_counts())

def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove special characters (keep only letters and spaces)
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Create cleaned dataset
df_clean = df.copy()
df_clean["review"] = df_clean["review"].apply(clean_text)

print(df_clean.head())

# Convert sentiment labels to numeric values
df_clean["sentiment"] = df_clean["sentiment"].map({"positive": 1, "negative": 0})
print("\nSentiment label conversion:")
print(df_clean["sentiment"].value_counts())

# Split dataset into training and testing sets
X = df_clean["review"]
y = df_clean["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# Convert text to numerical features using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"\nTF-IDF feature shape: {X_train_tfidf.shape}")
print(f"Test TF-IDF feature shape: {X_test_tfidf.shape}")

# Train Logistic Regression classifier
print("\nTraining Logistic Regression model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)
print("Model training completed!")

# Make predictions on test set
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
print("\n== Model Evaluation ==")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# Save the trained model and vectorizer
print("\nSaving model...")
os.makedirs('model', exist_ok=True)
pipeline = {
    'vectorizer': tfidf,
    'model': model
}
joblib.dump(pipeline, 'model/model.pkl')
print("Model saved to model/model.pkl")