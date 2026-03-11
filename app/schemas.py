from pydantic import BaseModel, field_validator
from typing import List

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float

class BatchPredictRequest(BaseModel):
    texts: List[str]
    
    @field_validator('texts')
    @classmethod
    def validate_texts_not_empty(cls, v):
        if not v:
            raise ValueError('texts list cannot be empty')
        return v

class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]
