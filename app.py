
from pathlib import Path

import joblib
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "artifacts" / "ticket_classifier.joblib"

REVIEW_THRESHOLD = 0.70

st.set_page_config(
    page_title="Customer Support Ticket Router",
    page_icon="🎫",
    layout="centered",
)

st.title("Customer Support Ticket Router")
st.write(
    "Classify a customer message into one of 77 support intents "
    "using a trained machine-learning model."
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

st.caption("Model: TF-IDF word and character features + Logistic Regression")

message = st.text_area(
    "Customer message",
    placeholder="Example: I lost my debit card and need to block it",
    height=130,
)

if st.button("Classify message", type="primary"):
    if not message.strip():
        st.warning("Please enter a customer message first.")
    else:
        probabilities = model.predict_proba([message])[0]
        classes = model.classes_
        top_indices = probabilities.argsort()[::-1][:3]

        best_index = top_indices[0]
        predicted_intent = classes[best_index]
        confidence = float(probabilities[best_index])

        st.subheader("Prediction")
        st.write(f"**Predicted intent:** `{predicted_intent}`")
        st.metric("Model confidence", f"{confidence:.2%}")

        if confidence >= REVIEW_THRESHOLD:
            st.success("Eligible for automatic routing under the demo policy.")
        else:
            st.warning("Human review recommended: low model confidence.")

        st.subheader("Top 3 candidate intents")
        for index in top_indices:
            st.write(
                f"- `{classes[index]}` — {probabilities[index]:.2%}"
            )

        st.caption(
            "Confidence is the model's predicted class probability, "
            "not a guarantee of correctness. The 70% threshold is "
            "provisional and has not been validated for production."
        )
