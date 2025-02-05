import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the preprocessed dataset
file_path = "../data/AppGallery_final_preprocessed.csv"
df = pd.read_csv(file_path)

# Ensure "Processed_Text" and "Type 2" columns exist
if "Processed_Text" not in df.columns or "Type 2" not in df.columns:
    raise ValueError("Required columns not found in dataset.")

# Drop any rows where Type 2 is missing
df = df.dropna(subset=["Type 2"])

# Convert "Type 2" into numerical labels
label_encoder = LabelEncoder()
df["Type 2 Encoded"] = label_encoder.fit_transform(df["Type 2"])

# Split data into train/test sets
X = df["Processed_Text"]  # Features (preprocessed text)
y = df["Type 2 Encoded"]  # Labels (Type 2 categories)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text into TF-IDF numerical representation
vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Naïve Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.4f}")

print("\n🔹 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save the model and vectorizer for future use
import joblib
joblib.dump(model, "../models/naive_bayes_model.pkl")
joblib.dump(vectorizer, "../models/tfidf_vectorizer.pkl")
joblib.dump(label_encoder, "../models/label_encoder.pkl")

print("\n✅ Model training complete. Model saved in 'models/' folder.")