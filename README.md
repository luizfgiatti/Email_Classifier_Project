# AI-Powered Email Classification

## Project Overview
This project implements a machine learning model to classify IT service request emails into different categories automatically. It uses Natural Language Processing (NLP) and Naïve Bayes classification to categorize emails efficiently.

## Folder Structure
Email_Classifier_Project/  
│── data/ (Contains datasets: cleaned, preprocessed, balanced)  
│── models/ (Saved ML models: Naïve Bayes, TF-IDF, Label Encoder)  
│── notebooks/ (Jupyter notebooks for exploratory data analysis)  
│── scripts/ (Python scripts for preprocessing, training, and testing)  
│── README.md (Project documentation)  
│── requirements.txt (Required Python packages)  

## Installation Guide  
Clone the repository:  
`git clone https://github.com/luizfgiatti/Email_Classifier_Project.git`  
`cd Email_Classifier_Project`  

Create a virtual environment:  
`python -m venv .venv`  
Activate the environment:  
`source .venv/bin/activate` (Mac/Linux)  
On Windows: `.venv\Scripts\activate`  

Install dependencies:  
`pip install -r requirements.txt`  

## Usage Guide  
Preprocess the data:  
`python scripts/data_preprocessing.py`  

Train the model:  
`python scripts/model_training.py`  

Test the model:  
`python scripts/load_model.py`  

## Model Performance  
Model Type: Naïve Bayes Classifier  
Feature Extraction: TF-IDF (10,000 features)  
Accuracy: 68%  
Categories: Problem/Fault, Others, Suggestion  

## Future Improvements  
Implement SVM and BERT-based classification for better accuracy.  
Deploy the model using Flask/FastAPI as an API.  
Perform hyperparameter tuning for better results.  

## Authors  
Luiz Fernando Alves  
Romulo Gomes da Silva  
National College of Ireland  
