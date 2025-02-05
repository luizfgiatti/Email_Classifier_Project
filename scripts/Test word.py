# Create a new comprehensive CA report document

doc = Document()

# Title
doc.add_heading("Continuous Assessment Report: AI-Powered Email Classification", level=1)

# Author and Course Details
doc.add_paragraph("Prepared by: Luiz Fernando Alves & Romulo Gomes da Silva")
doc.add_paragraph("Course: Engineering and Evaluating Artificial Intelligence Systems")
doc.add_paragraph("Instructor: Dr. Abdul Razzaq")
doc.add_paragraph("National College of Ireland")
doc.add_paragraph("Date: January 2025")

# Section 1: Introduction
doc.add_heading("1. Introduction", level=2)
doc.add_paragraph(
    "This report presents the development of an AI-powered email classification system "
    "that automates the categorization of IT service request emails. The goal is to improve "
    "response efficiency and accuracy by leveraging Natural Language Processing (NLP) and "
    "Machine Learning (ML) techniques. The project follows a structured AI workflow, including "
    "data preprocessing, feature extraction, model selection, and performance optimization."
)

# Section 2: Problem Statement
doc.add_heading("2. Problem Statement", level=2)
doc.add_paragraph(
    "Organizations receive large volumes of IT service request emails, which require manual "
    "classification before they can be assigned to appropriate teams. This manual process "
    "is time-consuming and error-prone. The objective of this project is to develop a machine "
    "learning model that can automatically classify emails based on their content, reducing "
    "human intervention and improving efficiency."
)

# Section 3: Data Preprocessing
doc.add_heading("3. Data Preprocessing", level=2)
doc.add_paragraph("The dataset used consists of service request emails with multilingual content. "
                  "The preprocessing steps include:")

doc.add_paragraph("1. **Translation**: Converting non-English emails into English for standardization.")
doc.add_paragraph("2. **Noise Removal**: Removing unwanted characters, punctuation, and numbers.")
doc.add_paragraph("3. **Tokenization**: Splitting text into meaningful words.")
doc.add_paragraph("4. **Stopword Removal**: Eliminating common words that do not contribute to classification.")
doc.add_paragraph("5. **TF-IDF Vectorization**: Converting text into numerical representations for machine learning.")

doc.add_heading("3.1 Data Preprocessing Code", level=3)
preprocessing_code = """
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
file_path = "../data/AppGallery_cleaned.csv"
df = pd.read_csv(file_path)

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'\\d+', '', text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df["Processed_Text"] = df["Ticket Summary"].apply(preprocess_text)
df.to_csv("../data/AppGallery_preprocessed.csv", index=False)
"""
doc.add_paragraph(preprocessing_code, style="Normal")

# Section 4: Feature Engineering
doc.add_heading("4. Feature Engineering", level=2)
doc.add_paragraph("To convert the preprocessed text into a format that can be used for machine learning, "
                  "we used the Term Frequency-Inverse Document Frequency (TF-IDF) method. This technique "
                  "assigns importance scores to words based on their frequency in a document relative "
                  "to other documents in the dataset.")

# Section 5: Model Training
doc.add_heading("5. Model Training", level=2)
doc.add_paragraph("We trained a Naïve Bayes classification model using 'Type 2' as the target variable. "
                  "The categories included: 'Problem/Fault', 'Suggestion', and 'Others'. The dataset was "
                  "split into training and testing sets, and the model was evaluated based on accuracy "
                  "and classification metrics.")

doc.add_heading("5.1 Model Training Code", level=3)
model_training_code = """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
file_path = "../data/AppGallery_final_preprocessed.csv"
df = pd.read_csv(file_path)

df = df.dropna(subset=["Type 2"])
label_encoder = LabelEncoder()
df["Type 2 Encoded"] = label_encoder.fit_transform(df["Type 2"])

X = df["Processed_Text"]
y = df["Type 2 Encoded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
"""
doc.add_paragraph(model_training_code, style="Normal")

# Section 6: Model Evaluation
doc.add_heading("6. Model Evaluation", level=2)
doc.add_paragraph("After training, the model was evaluated using accuracy, precision, recall, and F1-score. "
                  "The classification report showed that the model achieved 68% accuracy, with varying performance "
                  "across different categories.")

doc.add_paragraph("- **Problem/Fault:** High recall (100%) but moderate precision (64%).")
doc.add_paragraph("- **Others:** High precision (100%) but low recall (22%).")
doc.add_paragraph("- **Suggestion:** Limited data (only 2 samples).")

# Section 7: Model Improvement
doc.add_heading("7. Model Improvement", level=2)
doc.add_paragraph("To improve the model, we considered the following strategies:")
doc.add_paragraph("1. **Balancing the dataset** to ensure equal representation of all categories.")
doc.add_paragraph("2. **Trying alternative models**, such as Support Vector Machines (SVM) and Random Forest.")
doc.add_paragraph("3. **Hyperparameter tuning** to optimize Naïve Bayes performance.")

# Section 8: Conclusion
doc.add_heading("8. Conclusion", level=2)
doc.add_paragraph("This project successfully demonstrated how NLP and Machine Learning can automate email classification "
                  "for IT service requests. Future work includes testing additional models, refining hyperparameters, "
                  "and integrating the model into a chatbot system for real-time classification.")

# Save the document
file_path = "/mnt/data/AI_Email_Classification_Step_By_Step_Report.docx"
doc.save(file_path)

# Provide the file for download
file_path