from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle


# Paths
DATA_PATH = Path("data/sms_spam.csv")
MODEL_PATH = Path("models/spam_nb_tfidf.pkl")


def load_data(path: Path) -> pd.DataFrame:
    """
    Load the spam dataset from a CSV file.
    Expects columns: 'label' (spam/ham) and 'text' (message).
    """
    df = pd.read_csv(path)
    # Keep only the needed columns and drop any missing values
    df = df[["label", "text"]].dropna()
    return df


def build_pipeline() -> Pipeline:
    """
    Create an sklearn Pipeline: TF-IDF vectorizer + Multinomial Naive Bayes classifier.
    """
    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("clf", MultinomialNB(alpha=1.0)),
        ]
    )
    return pipeline


def train_and_evaluate(df: pd.DataFrame) -> Pipeline:
    """
    Train the spam classifier and print basic evaluation metrics.
    Returns the trained pipeline.
    """
    X = df["text"]
    y = df["label"]

    # Split into train/test sets (80/20) with stratification to preserve spam/ham ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Predict on test set
    y_pred = pipeline.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.3f}\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    return pipeline


def save_model(pipeline: Pipeline, path: Path) -> None:
    """
    Save the trained pipeline to disk using pickle.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\nModel saved to: {path}")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find dataset at {DATA_PATH}")

    print(f"Loading data from: {DATA_PATH}")
    df = load_data(DATA_PATH)

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    print("\nTraining model...\n")
    pipeline = train_and_evaluate(df)

    save_model(pipeline, MODEL_PATH)


if __name__ == "__main__":
    main()
