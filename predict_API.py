from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
from joblib import load

# Define the application
app = FastAPI(title="Heart Disease Prediction API")

# Define the model input as a Pydantic model
class HeartDiseaseModelInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

# Load the pre-trained models
try:
    loaded_models = {
        'decision_tree': load('DecisionTreeClassifier.joblib'),
        'random_forest': load('RandomForestClassifier.joblib'),
    }
except FileNotFoundError as e:
    print(f"Error loading models: {e}")
    raise HTTPException(status_code=500, detail="Model files not found")

# Define the prediction endpoint
@app.post("/predict/{model_type}")
def predict(model_type: str, input: HeartDiseaseModelInput):
    if model_type not in loaded_models:
        raise HTTPException(status_code=400, detail="Model type not supported")

    # Convert the input data into an array for the model
    input_data = np.array([[
        input.age, input.sex, input.cp, input.trestbps, input.chol,
        input.fbs, input.restecg, input.thalach, input.exang,
        input.oldpeak, input.slope, input.ca, input.thal
    ]])

    # Make the prediction
    model = loaded_models[model_type]
    prediction = model.predict(input_data)

    # Return the prediction result
    result = "The person has no Heart disease" if prediction[0] == 0 else "The person is diabetic"
    return {"prediction": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
