
from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"

df = pd.read_csv(DATA_PATH)
X = df["text"].astype(str)
y = df["category"]

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

features = FeatureUnion([
    (
        "word_tfidf",
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        ),
    ),
    (
        "char_tfidf",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True,
            max_features=100_000,
        ),
    ),
])

model = Pipeline([
    ("features", features),
    ("classifier", LogisticRegression(
        C=8.0,
        max_iter=2000,
        random_state=RANDOM_STATE,
    )),
])

print("Training evaluation model...")
model.fit(X_train, y_train)

predictions = model.predict(X_val)
probabilities = model.predict_proba(X_val)
confidence = probabilities.max(axis=1)

print("\nOVERALL VALIDATION METRICS")
print(f"Validation examples: {len(y_val)}")
print(f"Accuracy: {accuracy_score(y_val, predictions):.4f}")
print(
    "Macro-F1: "
    f"{f1_score(y_val, predictions, average='macro'):.4f}"
)
print(
    "Weighted-F1: "
    f"{f1_score(y_val, predictions, average='weighted'):.4f}"
)

print("\nCONFIDENCE ANALYSIS")
correct = (predictions == y_val.to_numpy())

print(f"Mean confidence: {confidence.mean():.4f}")
print(f"Accuracy when confidence >= 0.50: "
      f"{correct[confidence >= 0.50].mean():.4f}")
print(f"Examples with confidence < 0.50: "
      f"{(confidence < 0.50).sum()}")

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_val,
    predictions,
    zero_division=0,
))
