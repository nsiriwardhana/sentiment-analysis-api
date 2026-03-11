# Sentiment Analysis API

A machine learning-powered REST API for sentiment analysis built with FastAPI and scikit-learn. This project trains a sentiment classifier on the IMDb movie review dataset and exposes predictions through a fast, production-ready API.

## Project Overview

This project provides a complete end-to-end sentiment analysis solution:
- **Data Processing**: Cleans and preprocesses text data (HTML removal, lowercasing, special character removal)
- **Model Training**: Trains a Logistic Regression classifier using TF-IDF features
- **API Service**: FastAPI endpoints for single and batch predictions
- **Model Persistence**: Saves trained model and vectorizer for deployment

The API classifies text as **positive** or **negative** and returns a confidence score.

## Technologies Used

- **Python 3.8+**
- **FastAPI** - Modern web framework for building APIs
- **scikit-learn** - Machine learning library for training and inference
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **joblib** - Model serialization
- **uvicorn** - ASGI server for running FastAPI
- **Pydantic** - Data validation using Python type hints

## Project Structure

```
sentiment-analysis-api/
├── app/
│   ├── main.py           # FastAPI application with endpoints
│   ├── model.py          # Model loading and prediction logic
│   └── schemas.py        # Pydantic models for request/response
├── data/
│   └── IMDB_Dataset.csv  # Training dataset
├── model/
│   └── model.pkl         # Trained model and vectorizer (generated)
├── train.py              # Model training script
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore file
└── README.md            # This file
```

## Setup Instructions

### 1. Clone the repository
```bash
git clone <repository-url>
cd sentiment-analysis-api
```

### 2. Create a virtual environment
```bash
python -m venv venv
```

### 3. Activate the virtual environment
**Windows:**
```bash
venv\Scripts\activate
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Download the dataset
Place the `IMDB_Dataset.csv` file in the `data/` directory. You can download it from [Kaggle]

## How to Train the Model

Run the training script to preprocess the data, train the model, and save it to disk:

```bash
python train.py
```

This script will:
1. Load the IMDb dataset (50,000 movie reviews)
2. Clean and preprocess the text
3. Convert sentiment labels to numeric values (positive=1, negative=0)
4. Split data into 80% training and 20% testing
5. Convert text to TF-IDF features (max 5000 features)
6. Train a Logistic Regression classifier
7. Evaluate the model and print metrics
8. Save the trained pipeline to `model/model.pkl`

**Training time:** ~2-3 minutes on a modern CPU

## How to Run the API

Start the FastAPI server using uvicorn:

```bash
uvicorn app.main:app --reload
```

The API will be available at: `http://127.0.0.1:8000`

**Interactive API documentation:**
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

### Available Endpoints

#### Health Check
```
GET /health
```
Returns: `{"status": "ok"}`

#### Single Prediction
```
POST /predict
```
Request body:
```json
{
  "text": "This movie was absolutely fantastic!"
}
```

Response:
```json
{
  "text": "This movie was absolutely fantastic!",
  "sentiment": "positive",
  "confidence": 0.94
}
```

#### Batch Prediction
```
POST /predict/batch
```
Request body:
```json
{
  "texts": [
    "This movie was terrible",
    "I loved every minute of it",
    "Not my cup of tea"
  ]
}
```

Response:
```json
{
  "predictions": [
    {
      "text": "This movie was terrible",
      "sentiment": "negative",
      "confidence": 0.89
    },
    {
      "text": "I loved every minute of it",
      "sentiment": "positive",
      "confidence": 0.95
    },
    {
      "text": "Not my cup of tea",
      "sentiment": "negative",
      "confidence": 0.72
    }
  ]
}
```

## Example API Usage

### Using curl

**Single prediction:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"This movie was absolutely fantastic!\"}"
```

**Batch prediction:**
```bash
curl -X POST "http://127.0.0.1:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d "{\"texts\": [\"Great movie!\", \"Terrible film.\"]}"
```

### Using Python requests
```python
import requests

# Single prediction
response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"text": "This movie was absolutely fantastic!"}
)
print(response.json())

# Batch prediction
response = requests.post(
    "http://127.0.0.1:8000/predict/batch",
    json={"texts": ["Great movie!", "Terrible film."]}
)
print(response.json())
```

## Model Choice: TF-IDF + Logistic Regression

### Why this approach?

**TF-IDF (Term Frequency-Inverse Document Frequency):**
- Converts text into numerical features
- Weights important words higher while downweighting common words
- Captures word importance in the context of the entire dataset
- Fast and memory-efficient for inference

**Logistic Regression:**
- Simple yet powerful linear classifier
- Fast training and prediction
- Interpretable coefficients
- Works well with high-dimensional sparse data (like TF-IDF)
- Low risk of overfitting with proper regularization
- Provides probability scores (confidence)

This combination offers an excellent **baseline model** that is:
- **Fast**: Sub-millisecond predictions
- **Lightweight**: Small model size (~10MB)
- **Scalable**: Can handle high request volumes
- **Performant**: Achieves ~88-90% accuracy on sentiment classification

## Model Evaluation Metrics

The trained model achieves the following performance on the test set (10,000 reviews):

| Metric    | Score  | Description |
|-----------|--------|-------------|
| **Accuracy**  | ~0.89  | Overall correct predictions |
| **Precision** | ~0.89  | Proportion of positive predictions that were correct |
| **Recall**    | ~0.89  | Proportion of actual positives correctly identified |
| **F1 Score**  | ~0.89  | Harmonic mean of precision and recall |

These metrics indicate a well-balanced model with strong performance on both positive and negative sentiment classification.

## Future Improvements

### Model Enhancements
- [ ] Experiment with deep learning models (LSTM, BERT, RoBERTa)
- [ ] Add support for neutral sentiment
- [ ] Implement ensemble methods (combining multiple models)
- [ ] Fine-tune hyperparameters using grid search
- [ ] Add n-gram features (bigrams, trigrams)
- [ ] Implement cross-validation for more robust evaluation

### API Features
- [ ] Add rate limiting and authentication
- [ ] Implement caching for repeated queries
- [ ] Add logging and monitoring (Prometheus, Grafana)
- [ ] Create async batch processing for large requests
- [ ] Add confidence threshold filtering
- [ ] Support for multiple languages

### Infrastructure
- [ ] Containerize with Docker
- [ ] Deploy to cloud (AWS, GCP, Azure)
- [ ] Add CI/CD pipeline
- [ ] Implement A/B testing for model versions
- [ ] Add unit and integration tests
- [ ] Create performance benchmarks

### Data
- [ ] Expand to multi-domain sentiment analysis
- [ ] Add data augmentation techniques
- [ ] Implement active learning for model improvement
- [ ] Handle emojis and slang better

## License

This project is for educational purposes.

## Acknowledgments

- IMDb dataset from Kaggle
- FastAPI documentation and community
- scikit-learn contributors
