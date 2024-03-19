import numpy as np
from joblib import load
import streamlit as st

# Load the pre-trained models
try:
    loaded_models = {
        'decision_tree': load('DecisionTreeClassifier.joblib'),
        'random_forest': load('RandomForestClassifier.joblib'),
    }
except FileNotFoundError as e:
    st.error(f"Error loading models: {e}")
    st.stop()

def get_numeric_input(user_input):
    try:
        return float(user_input)
    except ValueError:
        return None

def predict(model_type: str, input_data):
    if model_type not in loaded_models:
        return {"prediction": "Model type not supported"}

    input_array = np.array([input_data])
    model = loaded_models[model_type]
    prediction = model.predict(input_array)

    result = "The person has no heart disease" if prediction[0] == 0 else "The person has heart disease"
    return {"prediction": result}

def main():
    st.title('Heart Disease Prediction Web App')
    with st.form("prediction_form"):
        st.selectbox("Choose the model for prediction", ['random_forest', 'decision_tree'], key="model_type")
        
        instructions = {
            "age": "The age of the individual.",
            "sex": "The gender of the individual (1 = male; 0 = female).",
            "cp": "Chest pain type (0 to 3).",
            "trestbps": "Resting blood pressure (in mm Hg on admission to the hospital).",
            "chol": "Serum cholesterol in mg/dl.",
            "fbs": "Fasting blood sugar > 120 mg/dl (1 = true; 0 = false).",
            "restecg": "Resting electrocardiographic results.",
            "thalach": "Maximum heart rate achieved.",
            "exang": "Exercise-induced angina (1 = yes; 0 = no).",
            "oldpeak": "ST depression induced by exercise relative to rest.",
            "slope": "The slope of the peak exercise ST segment.",
            "ca": "Number of major vessels (0-3) colored by fluoroscopy.",
            "thal": "Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)."
        }
        
        inputs = []
        for field, desc in instructions.items():
            user_input = st.text_input(f'Enter {field}', key=field)
            st.caption(desc)  # Show instructions below each input
            numeric_input = get_numeric_input(user_input)
            if numeric_input is None and user_input:  # Check if input is not empty and invalid
                st.error(f"Invalid input for {field}, please enter a number.")
                st.stop()
            inputs.append(numeric_input)
        
        submitted = st.form_submit_button("Heart Disease Prediction")
        if submitted:
            model_type = st.session_state.model_type
            diagnosis = predict(model_type, inputs)
            st.success(diagnosis['prediction'])

if __name__ == "__main__":
    main()
