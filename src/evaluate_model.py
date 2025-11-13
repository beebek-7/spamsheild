import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import pickle


DATA_PATH = Path("data/sms_spam.csv")
MODEL_PATH = Path("models/spam_nb_tfidf.pkl")


def load_pipeline(path: Path = MODEL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_data(path: Path = DATA_PATH):
    df = pd.read_csv(path)
    df = df[["label", "text"]].dropna()
    return df


def evaluate_model():
    df = load_data()
    X = df["text"]
    y = df["label"]

    # Split with same settings as training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = load_pipeline()

    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label="spam"),
        "recall": recall_score(y_test, y_pred, pos_label="spam"),
        "f1": f1_score(y_test, y_pred, pos_label="spam"),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "labels": ["ham", "spam"],
        "sample_counts": df["label"].value_counts().to_dict(),
    }

    return metrics
