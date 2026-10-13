
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

RANDOM_STATE = 42

df = pd.read_csv("data/raw/train.csv")

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
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
    ),
])

model.fit(X_train, y_train)
predictions = model.predict(X_val)

print(f"Validation examples: {len(y_val)}")
print(f"Accuracy:   {accuracy_score(y_val, predictions):.4f}")
print(f"Macro-F1:   {f1_score(y_val, predictions, average='macro'):.4f}")
print(f"Weighted-F1: {f1_score(y_val, predictions, average='weighted'):.4f}")
