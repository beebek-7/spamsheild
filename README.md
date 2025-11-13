\# 📩 SpamShield – AI-Based Spam Detection System



SpamShield is a lightweight, explainable AI system for SMS spam classification.

It uses \*\*TF-IDF text vectorization\*\*, \*\*Multinomial Naive Bayes\*\*, and a beautiful \*\*Streamlit UI\*\* to classify messages as \*\*Spam\*\* or \*\*Ham\*\*.

It also supports \*\*batch CSV labeling\*\*, \*\*model metrics\*\*, and \*\*word-importance visualization\*\* to show why the model made a decision.



This project was developed as part of \*\*CSCE 4201 – Introduction to Artificial Intelligence\*\*, University of North Texas.



---



\## 🧑‍💻 Developed By



\* \*\*Bibek Pandey\*\*



---



\# 🚀 Features



\### 🔍 Single Message Classification



\* Real-time spam/ham detection

\* Animated probability bars

\* Clean, modern UI



\### 📂 Batch CSV Classification



\* Upload a CSV containing a `text` column

\* Labeled CSV output with probabilities



\### 📊 Metrics Dashboard



\* Accuracy, Precision, Recall, F1

\* Confusion Matrix heatmap

\* Dataset label distribution



\### 🧠 Explainable AI (XAI)



\* Highlights the most influential words

\* Shows how each word contributed

\* Green gradients for HAM, red for SPAM



\### 🖥 Modern Streamlit UI



\* Dynamic animations

\* Dark-theme compatible

\* Fully interactive



---



\# 📁 Project Structure



```

spamsheild/

├── app.py

├── ui\_streamlit.py

├── make\_batch\_csv.py

├── README.md

├── requirements.txt

├── .gitignore

│

├── src/

│   ├── train.py

│   ├── batch\_predict.py

│   └── evaluate\_model.py

│

├── data/

│   ├── sms\_spam.csv

│   └── spamshield\_batch\_small.csv

│

└── models/

&nbsp;   └── spam\_nb\_tfidf.pkl

```



---



\# 🛠 Installation



\### 1️⃣ Create and activate virtual environment



```bash

python -m venv .venv

.\\.venv\\Scripts\\Activate

```



\### 2️⃣ Install dependencies



```bash

pip install -r requirements.txt

```



---



\# 🌐 Run Streamlit UI



```bash

streamlit run ui\_streamlit.py

```



Streamlit will open in your browser at:



```

http://localhost:8501

```



---



\# 🔍 Run CLI Version



```bash

python app.py

```



Example:



```

Message> You have won a FREE prize!

Predicted: spam

P(ham)=0.041  P(spam)=0.959

```



---



\# 🧪 Evaluate Model



```bash

python src/evaluate\_model.py

```



Outputs:



\* Accuracy

\* Precision

\* Recall

\* F1-score

\* Confusion matrix data



All metrics also appear in the Streamlit “📊 Metrics Dashboard”.



---



\# 🧠 Explainability



SpamShield uses TF-IDF feature weights and class log-probabilities to compute word-importance.



Explainability shows:



| Word  | Contribution |

| ----- | ------------ |

| free  | ████████████ |

| prize | ████████     |

| click | ███          |



This makes the model transparent, interpretable, and professor-friendly.



---



\# 🔮 Future Improvements



\* Add Logistic Regression, SVM, or BERT

\* Add ROC/PR curve visualizations

\* Add dataset exploration (word cloud)

\* Deploy on Streamlit Cloud / HuggingFace Spaces

\* Add persistent database for message storage



---



\# 📝 License



MIT License

Feel free to modify or extend this project.



---



