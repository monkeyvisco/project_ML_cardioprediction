#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


# In[3]:


#read the cv file and show the data columns
heart_data=pd.read_csv("heart.csv")
heart_data.head()


# In[4]:


heart_data.shape


# In[5]:


#getting some info regarding the data
heart_data.info()


# In[6]:


#checking for missing values
heart_data.isnull().sum()


# In[7]:


#staistical measure for data
heart_data.describe()


# In[ ]:


#checking the distribution for target variable
heart_data['target'].value_counts()


# In[ ]:


# prompt: check for correlation between colums in targets by coefficient and 

corr_matrix = heart_data.corr()
print(corr_matrix['target'].sort_values(ascending=False))


# In[ ]:


# prompt: plot cell 19 

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f")
plt.show()


# In[ ]:


# Bar plots for categorical variables
categorical_vars = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
for i, var in enumerate(categorical_vars):
    row = i // 3
    col = i % 3
    sns.countplot(x=var, data=heart_data, ax=axes[row, col])

plt.tight_layout()
plt.show()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Assuming 'heart_data' is your DataFrame loaded with the dataset information

# Selecting a subset of categorical variables for visualization
categorical_vars_subset = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

# Selecting a subset of continuous variables to explore against categorical variables
continuous_vars_subset = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Create box plots
fig, axes = plt.subplots(len(continuous_vars_subset), len(categorical_vars_subset), figsize=(20, 15))

for i, cont_var in enumerate(continuous_vars_subset):
    for j, cat_var in enumerate(categorical_vars_subset):
        sns.boxplot(ax=axes[i, j], x=heart_data[cat_var], y=heart_data[cont_var])
        axes[i, j].set_title(f'{cont_var} by {cat_var}')
        axes[i, j].set_xlabel('')
        axes[i, j].set_ylabel('')

# Adjust layout
plt.tight_layout()
plt.show()


# In[ ]:


#  write a function for examining data in terms of shape , null values and describe and type 

def examine_data(df):
  """
  This function examines a DataFrame and provides information about its shape, null values, describe and type.

  Args:
      df: A Pandas DataFrame.

  Returns:
      None
  """
  # Print the shape of the DataFrame
  print("Shape:", df.shape)

  # Check for null values
  print("Null Values:")
  print(df.isnull().sum())

  # Print the describe of the DataFrame
  print("Describe:")
  print(df.describe())

  # Print the type of the DataFrame
  print("Type:")
  print(df.dtypes)

# Use the function on the heart_data DataFrame
examine_data(heart_data)


# In[ ]:


def preprocess_data(df, target_column):
  """
  Preprocesses data for machine learning.

  Args:
      df: The pandas DataFrame containing the data.
      target_column: The name of the target column.

  Returns:
      A pandas DataFrame containing the preprocessed data.
  """

  # Drop the target column
  

  # Separate features and target
  X = df.drop(columns=[target_column])
  y = df[target_column]

  # Impute missing values
  if X.isnull().sum().sum() > 0:
    X = X.select_dtypes(include=["object"]).fillna(df.mode())
    
  else:
    X=X.select_dtypes(include=['int64','float64']).fillna(df.mean())

  # Encode categorical columns
  # Select categorical columns
  categorical_columns = df.select_dtypes(include=["object"]).columns

  # Encode categorical columns
  ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
  encoded_X = pd.concat([X.drop(categorical_columns, axis=1),
                            pd.DataFrame(ohe.fit_transform(X[categorical_columns]))], axis=1)

  # Scale numerical columns
  numerical_columns = X.select_dtypes(exclude=["object"]).columns
  scaler = StandardScaler()
  scaled_X = pd.DataFrame(scaler.fit_transform(encoded_X[numerical_columns]), columns=numerical_columns)

  # Check if oversampling is needed
  target_counts = y.value_counts()
  majority_class_count = target_counts.max()
  minority_class_count = target_counts.min()
  imbalance_ratio = majority_class_count / minority_class_count

  if imbalance_ratio > 1.5:
    # Oversample minority class
    oversampler = RandomOverSampler()
    scaled_X, y = oversampler.fit_resample(scaled_X,y)
  else:
    y = df[target_column]

  return scaled_X, y
  
# Use the function on the heart_data DataFrame
scaled_X, y = preprocess_data(heart_data, "target")

# Examine the preprocessed data
examine_data(scaled_X)


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import pandas as pd

def split_and_train(scaled_X,y, model_list):
    """
    This function splits a DataFrame into training and testing sets, and then trains multiple models on the training data.
    It also evaluates each model on the testing set and returns their performance metrics.

    Args:
        df: A Pandas DataFrame.
        target_column: The name of the target column.
        model_list: A list of machine learning models.

    Returns:
        A dictionary with trained models as keys and their evaluation metrics as values.
    """
    # Separate the features and the target variable
    

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=42)

    # Initialize a dictionary to hold trained models and their evaluation metrics
    model_performance = {}

    # Train each model on the training data and evaluate
    for model in model_list:
        model.fit(X_train, y_train)
        model_name = model.__class__.__name__

        # Predict on the test set
        y_pred = model.predict(X_test)

        # Calculate the evaluation metrics
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'F1 Score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }

        # Store the model and its performance metrics
        model_performance[model_name] = {'Model': model, 'Metrics': metrics}

        # Optionally print the results
        print(f"Model: {model_name}")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.2f}")
        print()

    return model_performance


# In[ ]:


# Instantiate each model
model_list = [
    LogisticRegression(max_iter=1000),  # Increase max_iter if needed for convergence
    DecisionTreeClassifier(),
    SVC(probability=True),  # Enable probability for SVC to use methods like predict_proba
    KNeighborsClassifier(),
    RandomForestClassifier(),]
    
# Assuming 'preprocessed_heart_data' is your DataFrame and 'target' is the target column
split_and_train(scaled_X,y, model_list)



# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

# split  preprocessed dataset
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=42)

# Initialize the models
decision_tree = DecisionTreeClassifier()
random_forest = RandomForestClassifier()

# Define a function to perform cross-validation and print scores
def perform_cross_validation(model, scaled_X, y, cv=5):
    scoring = {'accuracy': 'accuracy',
               'precision': make_scorer(precision_score, average='weighted'),
               'recall': make_scorer(recall_score, average='weighted'),
               'f1': make_scorer(f1_score, average='weighted')}

    scores = cross_val_score(model, scaled_X, y, cv=cv, scoring='accuracy')
    print(f"Accuracy (mean): {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")

    for metric, scorer in scoring.items():
        scores = cross_val_score(model, scaled_X, y, cv=cv, scoring=scorer)
        print(f"{metric.capitalize()} (mean): {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")

# Perform cross-validation for Decision Tree
print("Decision Tree Classifier:")
perform_cross_validation(decision_tree, X_train, y_train)

# Perform cross-validation for Random Forest
print("\nRandom Forest Classifier:")
perform_cross_validation(random_forest, X_train, y_train)


# In[ ]:


import matplotlib.pyplot as plt

# Model names
models = ['Logistic Regression', 'Decision Tree', 'SVC', 'KNN', 'Random Forest']

# Scores for each model
accuracy_scores = [0.80, 0.99, 0.89, 0.83, 0.99]
precision_scores = [0.80, 0.99, 0.89, 0.84, 0.99]
recall_scores = [0.80, 0.99, 0.89, 0.83, 0.99]
f1_scores = [0.79, 0.99, 0.89, 0.83, 0.99]

# Setting the positions and width for the bars
positions = list(range(len(models)))

# Plotting each metric
plt.figure(figsize=(10, 6))

plt.plot(models, accuracy_scores, marker='o', linestyle='-', label='Accuracy')
plt.plot(models, precision_scores, marker='s', linestyle='-', label='Precision')
plt.plot(models, recall_scores, marker='^', linestyle='-', label='Recall')
plt.plot(models, f1_scores, marker='*', linestyle='-', label='F1 Score')

# Adding some text for labels, title, and custom x-axis tick labels, etc.
plt.ylabel('Scores')
plt.title('Performance Comparison Across Models')
plt.xticks(positions, models, rotation=45)
plt.legend()

# Adding a grid
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Show the plot
plt.tight_layout()
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# Define the scores
scores = {
    'Logistic Regression': {'Accuracy': 0.80, 'Precision': 0.80, 'Recall': 0.80, 'F1 Score': 0.79},
    'Decision Tree': {'Accuracy': 0.99, 'Precision': 0.99, 'Recall': 0.99, 'F1 Score': 0.99},
    'SVC': {'Accuracy': 0.89, 'Precision': 0.89, 'Recall': 0.89, 'F1 Score': 0.89},
    'KNN': {'Accuracy': 0.83, 'Precision': 0.84, 'Recall': 0.83, 'F1 Score': 0.83},
    'Random Forest': {'Accuracy': 0.99, 'Precision': 0.99, 'Recall': 0.99, 'F1 Score': 0.99}
}

metrics = list(scores['Logistic Regression'].keys())
n_metrics = len(metrics)
n_models = len(scores)

# Create an array with the positions of each group on the x-axis
barWidth = 0.15
r = np.arange(n_metrics)

# Create the bar plots
plt.figure(figsize=(14, 8))

for i, (model, model_scores) in enumerate(scores.items()):
    plt.bar(r + i * barWidth, model_scores.values(), width=barWidth, edgecolor='grey', label=model)

# Add labels to the plot
plt.xlabel('Metric', fontweight='bold', fontsize=15)
plt.xticks([r + barWidth for r in range(n_metrics)], metrics)
plt.ylabel('Scores', fontweight='bold', fontsize=15)
plt.title('Comparison of Model Performance', fontweight='bold', fontsize=16)

# Create legend & Show graphic
plt.legend()
plt.show()


# In[ ]:




