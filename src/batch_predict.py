from pathlib import Path
import pickle
import pandas as pd


MODEL_PATH = Path("models/spam_nb_tfidf.pkl")


def load_model(path: Path = MODEL_PATH):
    """
    Load the trained TF-IDF + Naive Bayes pipeline from disk.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at {path}. Train it first with: python src/train.py"
        )
    with open(path, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline


def predict_batch(
    pipeline,
    input_csv: Path,
    output_csv: Path,
    text_column: str = "text",
):
    """
    Load a CSV, classify each row, and save a new CSV with predictions
    and probabilities.

    Expects a column named `text_column` containing the message text.
    Empty or missing rows are dropped.
    """
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found at {input_csv}")

    # Read the CSV
    df = pd.read_csv(input_csv)

    if text_column not in df.columns:
        raise KeyError(
            f"Column '{text_column}' not found in {input_csv}. "
            f"Available columns: {list(df.columns)}"
        )

    # Keep only the text column
    df = df[[text_column]].copy()

    # 1) Drop true NaNs first (before converting to string)
    df = df.dropna(subset=[text_column])

    # 2) Convert to string + strip whitespace
    df[text_column] = df[text_column].astype(str).str.strip()

    # 3) Drop empty strings and literal "nan" (from weird cases)
    df = df[df[text_column] != ""]
    df = df[df[text_column].str.lower() != "nan"]

    # Reset index after cleaning
    df = df.reset_index(drop=True)

    # Now texts is clean and aligned
    texts = df[text_column]

    # Predict labels
    pred_labels = pipeline.predict(texts)

    # Predict probabilities (if the model supports it)
    if hasattr(pipeline, "predict_proba"):
        probas = pipeline.predict_proba(texts)
        classes = list(pipeline.classes_)  # e.g., ['ham', 'spam']

        proba_df = pd.DataFrame(probas, columns=[f"proba_{cls}" for cls in classes])
        df["pred_label"] = pred_labels
        df = pd.concat([df, proba_df], axis=1)
    else:
        df["pred_label"] = pred_labels

    # Ensure output directory exists
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"Saved predictions to: {output_csv}")



def main():
    input_path = Path("data/messages_to_check.csv")
    output_path = Path("data/messages_to_check_labeled.csv")

    print(f"Loading model from: {MODEL_PATH}")
    pipeline = load_model()

    print(f"Running batch prediction on: {input_path}")
    predict_batch(pipeline, input_path, output_path, text_column="text")


if __name__ == "__main__":
    main()
