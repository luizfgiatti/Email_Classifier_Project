import pandas as pd
from deep_translator import GoogleTranslator

# Load the preprocessed dataset
file_path = "../data/AppGallery_preprocessed.csv"
df = pd.read_csv(file_path)

# Initialize the translator
translator = GoogleTranslator(source="auto", target="en")

# Function to translate text if it's not already in English
def translate_text(text):
    try:
        translated = translator.translate(text)
        return translated
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return original text if translation fails

# Apply translation to the 'Processed_Text' column
print("🔹 Translating text... This may take a few minutes.")
df["Translated_Text"] = df["Processed_Text"].apply(translate_text)

# Save the translated dataset
translated_file_path = "../data/AppGallery_translated.csv"
df.to_csv(translated_file_path, index=False)

print(f"✅ Translation complete! Translated data saved as '{translated_file_path}'.")