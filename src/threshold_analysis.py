
from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "raw" / "train.csv")

X = df["text"].astype(str)
y = df["category"]

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

features = FeatureUnion([
    ("word_tfidf", TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )),
    ("char_tfidf", TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        max_features=100_000,
    )),
])

model = Pipeline([
    ("features", features),
    ("classifier", LogisticRegression(
        C=8.0,
        max_iter=2000,
        random_state=42,
    )),
])

print("Training model on the training split...")
model.fit(X_train, y_train)

probabilities = model.predict_proba(X_val)
predictions = model.classes_[probabilities.argmax(axis=1)]
confidence = probabilities.max(axis=1)
correct = predictions == y_val.to_numpy()

print(f"\nTotal validation messages: {len(y_val)}")
print(f"Overall accuracy: {accuracy_score(y_val, predictions):.4f}")

print("\nTHRESHOLD ANALYSIS")
print("Threshold | Coverage | Auto-routed | Selective accuracy")

for threshold in [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
    accepted = confidence >= threshold
    count = int(accepted.sum())
    coverage = accepted.mean()

    if count:
        selective_accuracy = correct[accepted].mean()
        accuracy_text = f"{selective_accuracy:.4f}"
    else:
        accuracy_text = "N/A"

    print(
        f"{threshold:>9.2f} | "
        f"{coverage:>7.1%} | "
        f"{count:>11} | "
        f"{accuracy_text:>18}"
    )
