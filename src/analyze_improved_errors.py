
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
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
    ("classifier", LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    )),
])

model.fit(X_train, y_train)
predictions = model.predict(X_val)

report = classification_report(
    y_val,
    predictions,
    output_dict=True,
    zero_division=0,
)

class_metrics = pd.DataFrame(report).T
class_metrics = class_metrics.loc[y.unique().tolist()]

print("\n10 LOWEST-F1 INTENTS")
print(
    class_metrics.sort_values("f1-score")
    [["precision", "recall", "f1-score", "support"]]
    .head(10)
    .round(3)
    .to_string()
)

errors = pd.DataFrame({
    "text": X_val.to_numpy(),
    "actual": y_val.to_numpy(),
    "predicted": predictions,
})

errors = errors[errors["actual"] != errors["predicted"]]

print(f"\nTOTAL MISCLASSIFICATIONS: {len(errors)}")
print("\nFIRST 20 MISCLASSIFIED MESSAGES")

for _, row in errors.head(20).iterrows():
    print(f"\nText: {row['text']}")
    print(f"Actual: {row['actual']}")
    print(f"Predicted: {row['predicted']}")
