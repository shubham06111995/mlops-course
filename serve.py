"""
serve.py
A simple REST API that serves predictions from our trained Animal Classifier.
This is what "model serving" means: turning a saved model into a live,
callable service that other applications can send requests to.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Animal Classifier API")

# Load the trained model once, when the API starts up
model = joblib.load("model.pkl")

# Define what a valid request looks like (data validation happens automatically)
class AnimalTraits(BaseModel):
    has_fur: int
    can_fly: int
    lays_eggs: int
    has_fins: int
    breathes_air: int
    moist_skin: int
    legs: int

# A simple "is this thing alive" check - common in real production APIs
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# The actual prediction endpoint
@app.post("/predict")
def predict(traits: AnimalTraits):
    sample = pd.DataFrame([traits.dict()])
    prediction = model.predict(sample)[0]
    return {"predicted_class": prediction}
