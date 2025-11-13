from pathlib import Path
import pickle

MODEL_PATH = Path("models/spam_nb_tfidf.pkl")


def load_model(path: Path = MODEL_PATH):
    """
    Load the trained TF-IDF + Naive Bayes pipeline from disk.
    """
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path}. "
                                f"Train the model first with: python src/train.py")
    with open(path, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline


def predict_one(pipeline, text: str):
    """
    Predict the label (spam/ham) for a single message and return
    both the label and class probabilities (if available).
    """
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Empty message. Please enter some text.")

    # pipeline expects a list of texts
    preds = pipeline.predict([cleaned])
    label = preds[0]

    proba_dict = None
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba([cleaned])[0]
        classes = pipeline.classes_  # e.g., ['ham', 'spam']
        proba_dict = {cls: float(p) for cls, p in zip(classes, probs)}

    return label, proba_dict


def main():
    """
    Simple interactive loop:
    - Loads the trained model
    - Asks the user for a message
    - Prints predicted label + probabilities
    """
    print("Loading model...")
    pipeline = load_model()
    print("Model loaded successfully.")

    print("\nType a message to classify it as spam or ham.")
    print("Type 'quit', 'exit', or 'q' to stop.\n")

    while True:
        text = input("Message> ")

        if text.lower().strip() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        try:
            label, proba = predict_one(pipeline, text)
        except ValueError as e:
            print(f"Error: {e}")
            continue

        print(f"\nPredicted label: {label}")

        if proba is not None:
            print("Probabilities:")
            for cls, p in proba.items():
                print(f"  P({cls}) = {p:.3f}")
        print()  # blank line for readability


if __name__ == "__main__":
    main()
