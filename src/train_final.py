
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "ticket_classifier.joblib"

df = pd.read_csv(DATA_PATH)

X = df["text"].astype(str)
y = df["category"]

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
    (
        "classifier",
        LogisticRegression(
            C=8.0,
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
    ),
])

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Training on {len(X)} labeled messages...")
model.fit(X, y)

joblib.dump(model, MODEL_PATH)

print("\nFinal model trained successfully.")
print(f"Training examples: {len(X)}")
print(f"Intent categories: {y.nunique()}")
print(f"Saved model: {MODEL_PATH}")
print(f"Model file size: {MODEL_PATH.stat().st_size / 1024:.1f} KB")
