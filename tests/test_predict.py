
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "artifacts" / "ticket_classifier.joblib"

model = joblib.load(MODEL_PATH)


def predict_intent(message: str) -> str:
    """Predict the support intent for a non-empty message."""
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Message must be a non-empty string.")

    return model.predict([message])[0]


def test_model_file_exists():
    assert MODEL_PATH.exists()


def test_model_has_77_intents():
    assert len(model.classes_) == 77


def test_lost_card_message():
    assert predict_intent(
        "I lost my debit card and need to block it"
    ) == "lost_or_stolen_card"


def test_change_pin_message():
    assert predict_intent(
        "I want to change my PIN"
    ) == "change_pin"


def test_refund_message():
    assert predict_intent(
        "How can I get a refund for a purchase?"
    ) == "request_refund"


def test_empty_message_is_rejected():
    try:
        predict_intent("   ")
    except ValueError:
        return

    raise AssertionError("Empty message should raise ValueError.")
