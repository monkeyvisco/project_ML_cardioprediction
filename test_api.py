import requests

# Define the URL of the FastAPI application
url = 'http://127.0.0.1:8000/predict/random_forest'

# Define the input data matching the HeartDiseaseModelInput schema
data = {
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1
}

# Send a POST request to the API
response = requests.post(url, json=data)

# Check if the request was successful
if response.status_code == 200:
    # Print the prediction result
    print("Response from API:", response.json())
else:
    print("Failed to get a response from the API, status code:", response.status_code)
