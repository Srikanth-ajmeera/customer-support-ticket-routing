# AI-Powered Customer Support Ticket Classification & Routing

## Overview

An independent machine learning project that classifies banking customer messages into one of 77 intent categories and recommends whether a message is suitable for automatic routing or should receive human review.

The system combines word-level and character-level TF-IDF features with Logistic Regression and provides an interactive Streamlit interface.

## Key Results

| Metric | Result |
|---|---:|
| Validation accuracy | 90.20% |
| Validation macro-F1 | 90.30% |
| Five-fold CV accuracy (mean) | 89.67% |
| Five-fold CV macro-F1 (mean) | 89.57% |
| Number of intent categories | 77 |

The hold-out results were measured on a stratified validation split. Cross-validation results are the mean across five stratified folds. These are development evaluation results, not an official benchmark or production performance guarantee.

## Dataset

This project uses the public [BANKING77 dataset](https://github.com/PolyAI-LDN/task-specific-datasets), which contains banking-related customer queries labelled with intent categories.

- Training dataset: 10,003 examples
- Test dataset: 3,080 examples
- Intent categories: 77

The public test set was not used for model selection. Downloaded dataset files are excluded from Git.

## Model

The classification pipeline uses:

- Word-level TF-IDF features with unigrams and bigrams
- Character-level TF-IDF features using character n-grams
- Logistic Regression with `C=8`
- A confidence-based review recommendation in the Streamlit application

The model was evaluated using accuracy, macro-F1, weighted-F1, stratified cross-validation, and example predictions.

## Features

- Classifies customer messages into 77 intent categories
- Displays the predicted intent and model confidence
- Shows the top three candidate intents
- Recommends human review when confidence is below the demo threshold
- Includes automated tests for prediction behaviour and model availability

## Project Structure

```text
.
├── app.py
├── data/
│   └── raw/
├── notebooks/
│   ├── 01_dataset_inspection.py
│   └── 02_eda.py
├── src/
│   ├── train_baseline.py
│   ├── tune_baseline.py
│   ├── cross_validate_model.py
│   ├── train_final.py
│   ├── predict.py
│   ├── evaluate_model.py
│   ├── threshold_analysis.py
│   ├── analyze_errors.py
│   └── analyze_improved_errors.py
├── tests/
│   └── test_predict.py
├── artifacts/
├── requirements.txt
└── README.md
```

## Setup

Use Python 3.12 or another compatible Python version.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Dataset Preparation

Create the raw-data directory:

```powershell
New-Item -ItemType Directory -Force .\data\raw
```

Download the official public dataset CSV files and save them as:

- `data/raw/train.csv`
- `data/raw/test.csv`

Source URLs:

- [Training data](https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv)
- [Test data](https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv)

## Train the Model

From the project root, run:

```powershell
python .\src\train_final.py
```

This trains the model on the labelled training dataset and saves the model artifact under `artifacts/`.

## Run the Web Application

After installing dependencies, preparing the dataset, and training the model, run:

```powershell
python -m streamlit run app.py
```

The application displays a local URL in the terminal. Open that URL in your browser to classify sample messages.

## Run Tests

```powershell
python -m pytest .\tests\test_predict.py -v
```

## Limitations and Responsible Use

- Confidence scores are model outputs and are not necessarily calibrated probabilities.
- The demo review threshold is provisional and has not been validated for a real customer-support operation.
- Low-confidence or ambiguous messages should be reviewed by a human.
- Sensitive or high-impact cases, such as lost or stolen cards, should follow appropriate safety and escalation rules regardless of model confidence.
- The model should be evaluated on representative, unseen data before real-world deployment.

## Technologies

Python, Pandas, scikit-learn, TF-IDF, Logistic Regression, Joblib, Streamlit, pytest, Git, and GitHub.
