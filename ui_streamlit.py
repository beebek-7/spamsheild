import io
from pathlib import Path

import pandas as pd
import streamlit as st

from app import load_model, predict_one  # reuse our existing logic


# -------------------------
#  Cached model loader
# -------------------------
@st.cache_resource
def get_pipeline():
    return load_model()


# -------------------------
#  Small helper functions
# -------------------------
def spam_badge(label: str) -> str:
    """Return an HTML badge for spam/ham label."""
    if label.lower() == "spam":
        color = "#ff4b4b"
        text = "SPAM"
        emoji = "🚨"
    else:
        color = "#1db954"
        text = "HAM (Not spam)"
        emoji = "✅"

    return f"""
<span style="
    background-color:{color};
    color:white;
    padding:0.2rem 0.6rem;
    border-radius:999px;
    font-size:0.9rem;
    font-weight:600;
    display:inline-flex;
    align-items:center;
    gap:0.35rem;">
    <span>{emoji}</span><span>{text}</span>
</span>
    """


def style_page():
    """Apply some custom CSS styling."""
    st.set_page_config(
        page_title="SpamShield - Spam Detection",
        page_icon="📩",
        layout="wide",
    )

    st.markdown(
        """
<style>
.main > div {
    max-width: 900px;
    margin: 0 auto;
}
.app-header {
    text-align: center;
    margin-bottom: 1rem;
}
.app-header h1 {
    font-size: 2.4rem;
    margin-bottom: 0.2rem;
}
.app-header p {
    font-size: 0.95rem;
    color: #aaaaaa;
}
.prob-card {
    padding: 0.9rem 1rem;
    border-radius: 0.8rem;
    border: 1px solid #444;
    background-color: #111827;
    color: #f9fafb;
    margin-top: 0.5rem;
}
.prob-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.35rem;
}
.prob-row:last-child {
    margin-bottom: 0;
}
.prob-label {
    width: 90px;
    font-size: 0.85rem;
    font-weight: 600;
}
.prob-bar-bg {
    flex: 1;
    height: 0.6rem;
    border-radius: 999px;
    background: #374151;
    overflow: hidden;
}
.prob-bar {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s ease-out;
}
.prob-bar.ham {
    background: linear-gradient(90deg, #34d399, #10b981);
}
.prob-bar.spam {
    background: linear-gradient(90deg, #fb7185, #ef4444);
}
.prob-value {
    width: 70px;
    text-align: right;
    font-variant-numeric: tabular-nums;
    font-size: 0.85rem;
}

/* Word importance card */
.word-card {
    padding: 0.9rem 1rem;
    border-radius: 0.8rem;
    border: 1px solid #444;
    background-color: #020617;
    color: #e5e7eb;
    margin-top: 0.75rem;
}
.word-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.35rem;
}
.word-row:last-child {
    margin-bottom: 0;
}
.word-token {
    width: 110px;
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: lowercase;
}
.word-score {
    width: 80px;
    text-align: right;
    font-variant-numeric: tabular-nums;
    font-size: 0.8rem;
    opacity: 0.85;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def compute_word_importance(pipeline, text: str, predicted_label: str, top_k: int = 8):
    """
    Compute simple word-importance scores for a given message and predicted label.
    """
    try:
        vec = pipeline.named_steps.get("tfidf")
        clf = pipeline.named_steps.get("clf")
    except Exception:
        return []

    if vec is None or clf is None:
        return []

    X = vec.transform([text])
    if X.nnz == 0:
        return []

    feature_names = vec.get_feature_names_out()
    classes = list(clf.classes_)
    if "spam" not in classes or "ham" not in classes:
        return []

    if predicted_label == "spam":
        pos_idx = classes.index("spam")
        neg_idx = classes.index("ham")
    else:
        pos_idx = classes.index("ham")
        neg_idx = classes.index("spam")

    log_pos = clf.feature_log_prob_[pos_idx]
    log_neg = clf.feature_log_prob_[neg_idx]
    delta = log_pos - log_neg  # positive = supports predicted label

    row = X.tocoo()
    items = []
    for idx, val in zip(row.col, row.data):
        word = feature_names[idx]
        score = float(val * delta[idx])
        items.append({"word": word, "score": score})

    items = [it for it in items if it["score"] > 0]
    items.sort(key=lambda d: d["score"], reverse=True)
    return items[:top_k]


# -------------------------
#  Single message tab
# -------------------------
def single_message_tab(pipeline):
    st.subheader("Single Message Classification")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        user_text = st.text_area(
            "Enter a message to classify:",
            height=140,
            placeholder="Example: WINNER!! You've been selected for a free prize. Click here now...",
        )

        classify_clicked = st.button("🔍 Classify message", use_container_width=True)

    with col_right:
        st.markdown("**Tips:**")
        st.markdown(
            "- Try pasting suspicious SMS or email snippets.\n"
            "- Compare neutral vs. spammy language."
        )

    if classify_clicked:
        if not user_text.strip():
            st.warning("Please enter a non-empty message.")
        else:
            with st.spinner("Running SpamShield model..."):
                label, proba = predict_one(pipeline, user_text)

            st.markdown("### Result")
            st.markdown(spam_badge(label), unsafe_allow_html=True)

            if proba is not None:
                st.markdown("#### Class probabilities")

                ham_p = float(proba.get("ham", 0.0))
                spam_p = float(proba.get("spam", 0.0))

                ham_pct = max(0.0, min(100.0, ham_p * 100.0))
                spam_pct = max(0.0, min(100.0, spam_p * 100.0))

                prob_html = f"""
<div class="prob-card">
  <div class="prob-row">
    <div class="prob-label">📊 P(HAM)</div>
    <div class="prob-bar-bg">
      <div class="prob-bar ham" style="width:{ham_pct:.1f}%"></div>
    </div>
    <div class="prob-value">{ham_p:.3f}</div>
  </div>
  <div class="prob-row">
    <div class="prob-label">📊 P(SPAM)</div>
    <div class="prob-bar-bg">
      <div class="prob-bar spam" style="width:{spam_pct:.1f}%"></div>
    </div>
    <div class="prob-value">{spam_p:.3f}</div>
  </div>
</div>
                """
                st.markdown(prob_html, unsafe_allow_html=True)

            # ---------- Word importance visualization ----------
            importance = compute_word_importance(pipeline, user_text, label, top_k=8)
            if importance:
                st.markdown("#### Top contributing words")

                max_score = max(it["score"] for it in importance) or 1.0
                bar_class = "spam" if label.lower() == "spam" else "ham"

                rows_html = ""
                for it in importance:
                    width_pct = (it["score"] / max_score) * 100.0
                    width_pct = max(8.0, min(100.0, width_pct))

                    rows_html += f"""
<div class="word-row">
  <div class="word-token">{it['word']}</div>
  <div class="prob-bar-bg">
    <div class="prob-bar {bar_class}" style="width:{width_pct:.1f}%"></div>
  </div>
  <div class="word-score">{it['score']:.3f}</div>
</div>
                    """

                word_html = f"""
<div class="word-card">
{rows_html}
</div>
                """
                st.markdown(word_html, unsafe_allow_html=True)
            else:
                st.caption("No strong word contributions could be extracted for this message.")


# -------------------------
#  Batch upload tab
# -------------------------
def batch_tab(pipeline):
    st.subheader("Batch CSV Classification")

    st.write(
        "Upload a CSV file containing a column named **`text`**. "
        "SpamShield will label each message and return a downloadable CSV."
    )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read CSV file: {e}")
            return

        if "text" not in df.columns:
            st.error(f"Expected a column named `text`, but got: {list(df.columns)}")
            return

        df = df[["text"]].copy()
        df["text"] = df["text"].astype(str).str.strip()
        df = df[df["text"] != ""].reset_index(drop=True)

        if df.empty:
            st.warning("No non-empty text rows found.")
            return

        st.write("Preview:")
        st.dataframe(df.head(), use_container_width=True)

        if st.button("🚀 Run batch prediction"):
            with st.spinner("Classifying messages..."):
                texts = df["text"]
                pred_labels = pipeline.predict(texts)

                if hasattr(pipeline, "predict_proba"):
                    probas = pipeline.predict_proba(texts)
                    classes = list(pipeline.classes_)
                    proba_df = pd.DataFrame(
                        probas,
                        columns=[f"proba_{cls}" for cls in classes],
                    )
                    result_df = pd.concat(
                        [df, pd.Series(pred_labels, name="pred_label"), proba_df],
                        axis=1,
                    )
                else:
                    result_df = df.copy()
                    result_df["pred_label"] = pred_labels

            st.success("Batch prediction complete!")
            st.balloons()

            st.write("Results:")
            st.dataframe(result_df, use_container_width=True)

            csv_buffer = io.StringIO()
            result_df.to_csv(csv_buffer, index=False)
            csv_bytes = csv_buffer.getvalue().encode("utf-8")

            st.download_button(
                "⬇️ Download labeled CSV",
                data=csv_bytes,
                file_name="spamshield_labeled.csv",
                mime="text/csv",
            )


# -------------------------
#  Metrics tab
# -------------------------
def metrics_tab():
    st.subheader("Model Performance Metrics")

    from src.evaluate_model import evaluate_model
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    with st.spinner("Evaluating model..."):
        metrics = evaluate_model()

    st.markdown("### 📈 Summary")
    st.write(
        f"""
**Accuracy:** {metrics['accuracy']:.3f}  
**Precision (spam):** {metrics['precision']:.3f}  
**Recall (spam):** {metrics['recall']:.3f}  
**F1 Score (spam):** {metrics['f1']:.3f}  
        """
    )

    st.markdown("### 📊 Confusion Matrix")

    cm_array = np.array(metrics["confusion_matrix"])
    labels = metrics["labels"]

    fig, ax = plt.subplots()
    ax.imshow(cm_array, cmap="Blues")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm_array[i, j]), ha="center", va="center", color="black")

    st.pyplot(fig)

    st.markdown("### 📦 Dataset Label Distribution")

    counts = metrics["sample_counts"]
    counts_series = pd.Series(counts)
    st.bar_chart(counts_series)


# -------------------------
#  Main app
# -------------------------
def main():
    style_page()

    st.markdown(
        """
<div class="app-header">
  <h1>📩 SpamShield</h1>
  <p>AI-based spam detection using TF-IDF and Naive Bayes · CSCE 4201 Project</p>
</div>
        """,
        unsafe_allow_html=True,
    )

    pipeline = get_pipeline()

    tabs = st.tabs(
        ["💬 Single Message", "📂 Batch CSV Upload", "📊 Metrics Dashboard", "ℹ️ About"]
    )

    with tabs[0]:
        single_message_tab(pipeline)

    with tabs[1]:
        batch_tab(pipeline)

    with tabs[2]:
        metrics_tab()

    with tabs[3]:
        st.subheader("About this project")
        st.write(
            """
**SpamShield** is an AI-based spam detection system built for  
**CSCE 4201 – Introduction to Artificial Intelligence** at the  
**University of North Texas**.

### 👨‍💻 Developed By
- **Bibek Pandey**
- **Ojaswi Subedi**
- **Prasuna Khadka**

### 🧠 Features
- TF-IDF + Naive Bayes  
- Single-message prediction  
- Batch CSV labeling  
- Metrics dashboard with confusion matrix  
- Animated probability bars  
- Word-importance visualization  
- Modern Streamlit UI  

### 💡 Future Enhancements
- Logistic Regression, SVM  
- ROC curves and PR curves  
- Interactive threshold tuning  
- More detailed explainability
            """
        )


if __name__ == "__main__":
    main()
