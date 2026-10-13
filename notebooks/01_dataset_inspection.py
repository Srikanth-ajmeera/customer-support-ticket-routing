import pandas as pd

# Original BANKING77 CSV files
TRAIN_URL = (
    "https://raw.githubusercontent.com/"
    "PolyAI-LDN/task-specific-datasets/master/"
    "banking_data/train.csv"
)

TEST_URL = (
    "https://raw.githubusercontent.com/"
    "PolyAI-LDN/task-specific-datasets/master/"
    "banking_data/test.csv"
)

# Load the official training and test files
train_df = pd.read_csv(TRAIN_URL)
test_df = pd.read_csv(TEST_URL)

# Save local copies of the original dataset splits
train_df.to_csv("data/raw/train.csv", index=False)
test_df.to_csv("data/raw/test.csv", index=False)

print("\nDataset files saved successfully.")
print("Training file: data/raw/train.csv")
print("Test file: data/raw/test.csv")

print("TRAINING DATA")
print("Shape:", train_df.shape)
print("Columns:", train_df.columns.tolist())
print(train_df.head())

print("\nTEST DATA")
print("Shape:", test_df.shape)
print("Columns:", test_df.columns.tolist())
print(test_df.head())

# Inspect the target categories
print("\nTRAINING LABEL DISTRIBUTION")
print(train_df["category"].value_counts())

print("\nNumber of training categories:", train_df["category"].nunique())
print("Number of test categories:", test_df["category"].nunique())

print("\nMissing values in training data:")
print(train_df.isnull().sum())

print("\nMissing values in test data:")
print(test_df.isnull().sum())