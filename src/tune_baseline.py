
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

def build_model(c_value):
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

    return Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(
            C=c_value,
            max_iter=2000,
            random_state=RANDOM_STATE,
        )),
    ])

results = []

for c_value in [1.0, 2.0, 4.0, 8.0]:
    model = build_model(c_value)
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)
    macro_f1 = f1_score(
        y_val, predictions, average="macro"
    )
    weighted_f1 = f1_score(
        y_val, predictions, average="weighted"
    )

    results.append({
        "C": c_value,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    })

    print(
        f"C={c_value:<4} | "
        f"Accuracy={accuracy:.4f} | "
        f"Macro-F1={macro_f1:.4f} | "
        f"Weighted-F1={weighted_f1:.4f}"
    )

results_df = pd.DataFrame(results)
best = results_df.loc[results_df["macro_f1"].idxmax()]

print("\nBEST CONFIGURATION")
print(best.to_string())
