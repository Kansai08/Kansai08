import uvicorn
from fastapi import FastAPI, HTTPException
from joblib import load
import numpy as np

# Initialize FastAPI app
app = FastAPI(
    title="Heart Attack Prediction API",
    description="API for predicting heart attack risk using the best model (Random Forest)",
    version="1.0"
)

# Load the best model (Random Forest) and scaler
best_model = load('random_forest_model.joblib')  # Random Forest Model (the best model)
scaler = load('scaler.joblib')  # Scaler used for data preprocessing

# Define root endpoint
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Heart Attack Prediction API using the Best Model (Random Forest)!"}

# Define prediction endpoint
@app.post('/predict', tags=["predictions"])
async def get_prediction(
    age: float,
    sex: int,
    cp: int,
    trestbps: float,
    chol: float,
    fbs: int,
    restecg: int,
    thalach: float,
    exang: int,
    oldpeak: float,
    slope: int,
    ca: int,
    thal: int
):
    try:
        # Input features
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        # Preprocess input features (scaling)
        scaled_features = scaler.transform(features)

        # Make prediction using the best model (Random Forest)
        prediction = best_model.predict(scaled_features)

        # Prepare result
        result = "Heart Attack Risk" if prediction[0] == 1 else "No Heart Attack Risk"

        return {"Best Model Prediction": result}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)
