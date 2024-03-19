# Heart Disease Prediction Project

## Project Overview
This project aims to develop a machine learning model to predict the presence of heart disease in individuals based on various medical parameters. We use the `heart_data` dataset, which contains clinical measurements related to heart health.

## Dataset Description
The `heart_data` dataset includes the following features:

- `age`: The age of the individual.
- `sex`: The gender of the individual (1 = male; 0 = female).
- `cp`: Chest pain type (0 to 3).
- `trestbps`: Resting blood pressure (in mm Hg on admission to the hospital).
- `chol`: Serum cholesterol in mg/dl.
- `fbs`: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false).
- `restecg`: Resting electrocardiographic results.
- `thalach`: Maximum heart rate achieved.
- `exang`: Exercise-induced angina (1 = yes; 0 = no).
- `oldpeak`: ST depression induced by exercise relative to rest.
- `slope`: The slope of the peak exercise ST segment.
- `ca`: Number of major vessels (0-3) colored by fluoroscopy.
- `thal`: Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect).
- `target`: Presence of heart disease (1 = yes; 0 = no).

## Prerequisites
- Python 3.6 or above
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, streamlit

## Installation
To set up the project environment, run the following command to install the required packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn

Additional Installation
For deploying the machine learning model as an API and testing it locally, the following libraries are also required:

Flask: A lightweight WSGI web application framework.
Gunicorn: A Python WSGI HTTP Server for UNIX.
These can be installed using the following command in bash:
pip install flask gunicorn


## Dataset Source
The dataset was obtained from the UCI Machine Learning Repository:

Janosi, Andras, Steinbrunn, William, Pfisterer, Matthias, and Detrano, Robert. (1988). Heart Disease. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X

BibTeX citation:
```bibtex
@misc{misc_heart_disease_45,
  author       = {Janosi, Andras and Steinbrunn, William and Pfisterer, Matthias and Detrano, Robert},
  title        = {{Heart Disease}},
  year         = {1988},
  howpublished = {UCI Machine Learning Repository},
  note         = {{DOI}: https://doi.org/10.24432/C52P4X}
}
## Models Evaluated
The following machine learning models were evaluated:

Logistic Regression
Decision Tree Classifier
Support Vector Classifier (SVC)
K-Nearest Neighbors (KNN)
Random Forest Classifier
Evaluation Metrics
The models were evaluated based on Accuracy, Precision, Recall, and F1 Score.

## Results
The Random Forest and Decision Tree models demonstrated the highest accuracy in predicting the presence of heart disease.
Contributing
Contributions to this project are welcome. Please fork the repository and submit a pull request.


Model Deployment and API
We have developed an API to make predictions using the trained model. This API is built using Flask and can be tested on a local host.

Saving the Trained Model
The trained model is saved using joblib, which allows us to serialize Python objects. This makes it easy to load the model for future predictions without retraining.

Creating the API
The API accepts JSON input containing the necessary features for prediction and returns the prediction result. The Flask framework is used to create the API, allowing for easy testing and deployment.

Testing the API Locally
To test the API locally, run the Flask application. You can then use tools like Postman or a simple curl command to send requests to your local server and receive predictions.

Interactive Web Application
1.This repository includes a sophisticated web application built with Streamlit, which provides a dynamic and intuitive interface for heart disease risk prediction. To utilize the web application:
"pip install streamlit"
2.To initiate the web application, execute the command below in your terminal:
First, ensure Streamlit is installed by running the following command:
"streamlit run streamlit2_web.py"
3.Upon launch, the application will prompt you to input relevant health metrics. After entering the data, click on the 'Predict' button to generate your heart disease risk assessment.
*****NOTE***: The provided predictive model is for educational purposes and should not be construed as medical advice. Always consult healthcare professionals for health-related inquiries.


## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments
Heart Disease UCI Dataset from Kaggle
Contributions from the open-source community




