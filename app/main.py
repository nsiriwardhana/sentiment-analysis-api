from fastapi import FastAPI, HTTPException
from app.schemas import PredictRequest, PredictResponse, BatchPredictRequest, BatchPredictResponse
from app.model import predict_sentiment

# Create FastAPI application
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for analyzing sentiment of text using machine learning",
    version="1.0.0"
)

@app.get("/health")
def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Predict sentiment of the given text.
    
    Args:
        request: PredictRequest containing the text to analyze
        
    Returns:
        PredictResponse with sentiment prediction and confidence score
        
    Raises:
        HTTPException: If text validation fails
    """
    try:
        # Call predict_sentiment function
        sentiment, confidence = predict_sentiment(request.text)
        
        # Return response
        return PredictResponse(
            text=request.text,
            sentiment=sentiment,
            confidence=confidence
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest):
    """
    Predict sentiment for multiple texts.
    
    Args:
        request: BatchPredictRequest containing a list of texts to analyze
        
    Returns:
        BatchPredictResponse with a list of predictions
        
    Raises:
        HTTPException: If any text validation fails
    """
    predictions = []
    
    for i, text in enumerate(request.texts):
        try:
            sentiment, confidence = predict_sentiment(text)
            predictions.append(
                PredictResponse(
                    text=text,
                    sentiment=sentiment,
                    confidence=confidence
                )
            )
        except ValueError as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Error at index {i}: {str(e)}"
            )
    
    return BatchPredictResponse(predictions=predictions)
