import pandas as pd

# Load local dataset files
train_df = pd.read_csv("data/raw/train.csv")
test_df = pd.read_csv("data/raw/test.csv")

# Calculate text lengths
train_df["text_length"] = train_df["text"].str.len()
test_df["text_length"] = test_df["text"].str.len()

print("DATASET SHAPES")
print("Train:", train_df.shape)
print("Test:", test_df.shape)

print("\nCLASS DISTRIBUTION")
print(train_df["category"].value_counts().describe())

print("\nTEXT LENGTH STATISTICS")
print("Train:")
print(train_df["text_length"].describe())

print("\nTest:")
print(test_df["text_length"].describe())

print("\nDUPLICATE TEXT CHECK")
print("Train duplicate texts:", train_df["text"].duplicated().sum())
print("Test duplicate texts:", test_df["text"].duplicated().sum())

train_texts = set(train_df["text"].dropna())
test_texts = set(test_df["text"].dropna())

overlap = train_texts.intersection(test_texts)

print("\nTRAIN-TEST TEXT OVERLAP")
print("Identical messages in both splits:", len(overlap))