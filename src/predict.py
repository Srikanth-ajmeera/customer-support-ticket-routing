
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "artifacts" / "ticket_classifier.joblib"

print("Loading customer-support classifier...")
model = joblib.load(MODEL_PATH)
print("Model loaded successfully.")

while True:
    message = input("\nEnter a customer message (or type 'exit'): ").strip()

    if message.lower() == "exit":
        print("Exiting.")
        break

    if not message:
        print("Please enter a non-empty message.")
        continue

    predicted_intent = model.predict([message])[0]
    probabilities = model.predict_proba([message])[0]
    confidence = float(probabilities.max())

    print(f"Predicted intent: {predicted_intent}")
    print(f"Model confidence: {confidence:.2%}")

    if confidence < 0.50:
        print("Review recommended: low model confidence.")
