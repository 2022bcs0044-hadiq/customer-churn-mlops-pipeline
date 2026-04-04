from pydantic import BaseModel

class PredictionRequest(BaseModel):
    # Add input features
    pass

class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
