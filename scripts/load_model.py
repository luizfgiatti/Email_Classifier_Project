import joblib
import pandas as pd

# Load the saved model, vectorizer, and label encoder
model = joblib.load("../models/naive_bayes_model.pkl")
vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")
label_encoder = joblib.load("../models/label_encoder.pkl")

# List of sample emails to classify
sample_emails = [
    "My app is not working. I can't install new updates.",
    "I want a refund for my purchase. The app charged me twice!",
    "The interface is very confusing. You should improve navigation.",
    "I lost all my reward points. How can I recover them?",
    "My internet connection is fine, but the app is not loading.",
    "I would like to suggest a dark mode feature for better usability.",
]

# Convert text into numerical format using TF-IDF
sample_tfidf = vectorizer.transform(sample_emails)

# Make predictions
predicted_labels = model.predict(sample_tfidf)

# Convert numeric predictions back to category names
predicted_categories = label_encoder.inverse_transform(predicted_labels)

# Display results
print("✅ Classification Results:")
for email, category in zip(sample_emails, predicted_categories):
    print(f"\n📩 **Email:** {email}\n🔹 **Predicted Category:** {category}")