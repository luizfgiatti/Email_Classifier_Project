import pandas as pd

# Load dataset
file_path = "/Users/luizfernandogiatti/Documents/Estudos/NCI_AI Post/Engineering and Evaluating AI/Email_Classifier_Project/data/AppGallery.csv"
df = pd.read_csv(file_path)

# Drop unnecessary columns
df = df.drop(columns=['Unnamed: 11'], errors='ignore')

# Remove duplicate rows
df = df.drop_duplicates()

# Fill missing values with appropriate defaults
df['Ticket Summary'].fillna("No summary provided", inplace=True)
df['Interaction content'].fillna("No content available", inplace=True)
df['Type 3'].fillna("Unknown", inplace=True)
df['Type 4'].fillna("Unknown", inplace=True)

# Convert 'Interaction date' to datetime format
df['Interaction date'] = pd.to_datetime(df['Interaction date'], errors='coerce')

# Trim whitespace from all string columns
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Convert categorical columns to category data type for memory efficiency
df['Type 3'] = df['Type 3'].astype('category')
df['Type 4'] = df['Type 4'].astype('category')

# Save the cleaned dataset
cleaned_file_path = "/Users/luizfernandogiatti/Documents/Estudos/NCI_AI Post/Engineering and Evaluating AI/Email_Classifier_Project/data/AppGallery_cleaned.csv"
df.to_csv(cleaned_file_path, index=False)

# Print dataset summary
print("Data Cleaning Completed")
print(df.info())
print("\nSample Data After Cleaning:")
print(df.head())