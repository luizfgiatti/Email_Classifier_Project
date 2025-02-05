import pandas as pd

# Load dataset
file_path = "../data/AppGallery_final_preprocessed.csv"
df = pd.read_csv(file_path)

# Check initial class distribution
print("\n🔹 Class Distribution in Type 2 (Before Balancing):")
print(df["Type 2"].value_counts())

# Balance the dataset by oversampling minority classes
max_samples = df["Type 2"].value_counts().max()

df_balanced = df.groupby("Type 2", group_keys=False).apply(lambda x: x.sample(max_samples, replace=True))

# Save the balanced dataset
df_balanced.to_csv("../data/AppGallery_balanced.csv", index=False)

# Check the new distribution
print("\n✅ Balanced Dataset Created! New Class Distribution:")
print(df_balanced["Type 2"].value_counts())