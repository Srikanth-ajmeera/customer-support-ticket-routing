
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import FeatureUnion, Pipeline

RANDOM_STATE = 42

df = pd.read_csv("data/raw/train.csv")
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
    ("classifier", LogisticRegression(
        C=8.0,
        max_iter=2000,
        random_state=RANDOM_STATE,
    )),
])

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

scoring = {
    "accuracy": "accuracy",
    "macro_f1": make_scorer(
        f1_score,
        average="macro",
        zero_division=0,
    ),
    "weighted_f1": make_scorer(
        f1_score,
        average="weighted",
        zero_division=0,
    ),
}

print("Running 5-fold cross-validation...")

results = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring=scoring,
    n_jobs=1,
    return_train_score=False,
)

for metric in scoring:
    scores = results[f"test_{metric}"]
    print(
        f"{metric}: "
        f"{scores.mean():.4f} "
        f"+/- {scores.std():.4f}"
    )

print("\nIndividual fold results")
for i in range(5):
    print(
        f"Fold {i + 1}: "
        f"Accuracy={results['test_accuracy'][i]:.4f}, "
        f"Macro-F1={results['test_macro_f1'][i]:.4f}"
    )
