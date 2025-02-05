import pandas as pd

# Load the dataset
file_path = "../data/AppGallery.csv"  # Adjusted relative path
df = pd.read_csv(file_path)

# Show basic info
print("🔍 Dataset Info:")
print(df.info())

# Show first 5 rows
print("\n📊 Sample Data:")
print(df.head())

# Check missing values
print("\n⚠️ Missing Values:")
print(df.isnull().sum())

# Check unique values per column
print("\n🔢 Unique Values per Column:")
print(df.nunique())