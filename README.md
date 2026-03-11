# Sentiment Analysis API

## Project Overview

This project builds a sentiment classification model and exposes it through a FastAPI REST API. The API analyzes text and returns sentiment predictions classified as **Positive**, **Negative**, or **Neutral**, along with a confidence score.

The model is trained on the IMDb movie review dataset using TF-IDF vectorization and Logistic Regression. Neutral sentiment is inferred using probability thresholds, as the training dataset contains only binary labels (positive/negative).

## Requirements

- **Python 3.10 or newer**
- **pip** package manager

## Project Structure

```
sentiment-analysis-api
│
├── app
│   ├── main.py           # FastAPI application with endpoints
│   ├── model.py          # Model loading and prediction logic
│   └── schemas.py        # Pydantic models for request/response
│
├── data
│   └── IMDB_Dataset.csv  # Training dataset
│
├── model
│   └── model.pkl         # Trained model and vectorizer (generated)
│
├── train.py              # Model training script
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Setup Instructions

### Step 1 – Clone the repository

```bash
git clone <repo-url>
cd sentiment-analysis-api
```

### Step 2 – Create a virtual environment

```bash
python -m venv venv
```

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

### Step 3 – Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 – Train the model

```bash
python train.py
```

This will train the TF-IDF + Logistic Regression model and save it to `model/model.pkl`.

**Note:** Training takes approximately 2-3 minutes on a modern CPU.

### Step 5 – Start the API server

```bash
uvicorn app.main:app --reload
```

The API will start at:
- **Base URL:** http://127.0.0.1:8000

Interactive API documentation is available at:
- **Swagger UI:** http://127.0.0.1:8000/docs
- **ReDoc:** http://127.0.0.1:8000/redoc

## API Endpoints

### Health Check
```
GET /health
```

### Single Prediction
```
POST /predict
```

### Batch Prediction
```
POST /predict/batch
```

## Example API Request

**Single prediction:**

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "text": "I absolutely love how fast the delivery was!"
}'
```

**Example response:**

```json
{
  "text": "I absolutely love how fast the delivery was!",
  "sentiment": "positive",
  "confidence": 0.97
}
```

**Batch prediction:**

```bash
curl -X POST "http://127.0.0.1:8000/predict/batch" \
-H "Content-Type: application/json" \
-d '{
  "texts": [
    "This movie was terrible",
    "I loved every minute of it",
    "It was okay, nothing special"
  ]
}'
```

## Approach

The model uses TF-IDF vectorization to convert text into numerical features and Logistic Regression for classification. The IMDb dataset contains 50,000 labeled movie reviews that allow the model to learn patterns associated with positive and negative sentiment. Neutral sentiment is inferred using probability thresholds—when the model's confidence is low (probability difference < 0.2), the prediction is classified as neutral. Logistic Regression was chosen because it is efficient, interpretable, and performs well for text classification tasks with TF-IDF features. The model achieves approximately 89% accuracy on the test set. With more time, the system could be improved by using transformer-based models such as BERT and adding better neutral sentiment detection through a multi-class training approach.

## Model Performance

The trained model achieves the following metrics on the test set:

- **Accuracy:** ~0.89
- **Precision:** ~0.89
- **Recall:** ~0.89
- **F1 Score:** ~0.89

## Technologies Used

- **FastAPI** - Web framework for building APIs
- **scikit-learn** - Machine learning library
- **pandas** - Data manipulation
- **joblib** - Model serialization
- **uvicorn** - ASGI server

## License

This project is for educational purposes.

