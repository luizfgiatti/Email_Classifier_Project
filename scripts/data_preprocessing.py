import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure required resources are downloaded
nltk.download('stopwords')
nltk.download('wordnet')

# Load the cleaned dataset
file_path = "../data/AppGallery_cleaned.csv"  # Adjust path if needed
df = pd.read_csv(file_path)

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Define a function to preprocess text (WITHOUT word_tokenize)
def preprocess_text(text):
    if pd.isna(text):  # Handle missing values
        return ""

    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation

    words = text.split()  # 🚀 Simple tokenization (splitting by spaces)
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatization
    return " ".join(words)

# Apply text preprocessing
df["Processed_Text"] = df["Ticket Summary"].apply(preprocess_text)

# Convert text to numerical representation using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Limit features for efficiency
X_tfidf = vectorizer.fit_transform(df["Processed_Text"])

# Save the transformed data
df.to_csv("../data/AppGallery_preprocessed.csv", index=False)
print("✅ Popen model_training.pyreprocessing complete. Data saved as 'AppGallery_preprocessed.csv'.")