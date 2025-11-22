SpamShield — AI-Based Spam Detection System
CSCE 4201 — Introduction to Artificial Intelligence
University of North Texas
SpamShield is an AI-powered SMS spam detection system that uses
TF-IDF text vectorization, Multinomial Naive Bayes classification,
and a Streamlit-based user interface.
The system supports single-message classification, batch CSV processing,
a model evaluation dashboard, and explainability through word-importance analysis.
________________


Authors
* Bibek Pandey

* Ojaswi Subedi

* Prasuna Khadka

________________


1. Project Overview
SpamShield demonstrates the application of classical machine learning
techniques for text classification, focusing on:
   * TF-IDF feature extraction

   * Multinomial Naive Bayes classification

   * Web-based interactive inference

   * Explainable model outputs

   * Batch processing and evaluation

This project fulfills the requirements for Milestones of
CSCE 4201 – Introduction to Artificial Intelligence.
________________


2. Features
      * Single-message spam detection

      * Batch CSV file classification

      * Class probability visualization

      * Word-importance explainability

      * Model metrics dashboard (accuracy, precision, recall, F1-score, confusion matrix)

      * Modern Streamlit user interface

________________


3. Project Structure
spamsheild/
├── app.py                     # Command-line spam classifier
├── ui_streamlit.py            # Main interactive Streamlit UI
│
├── src/
│   ├── train.py               # Training script (TF-IDF + Naive Bayes)
│   ├── batch_predict.py       # Batch CSV prediction
│   └── evaluate_model.py      # Computes model metrics & confusion matrix
│
├── data/
│   ├── sms_spam.csv           # Training dataset
│   └── example_batch.csv      # Sample input for batch mode
│
├── models/
│   └── spam_nb_tfidf.pkl      # Trained model file
│
├── requirements.txt
├── .gitignore
└── README.md


________________


4. Installation Guide
Step 1 — Clone the repository
git clone https://github.com/beebek-7/spamsheild.git
cd spamsheild


Step 2 — Create a virtual environment
python -m venv .venv


Step 3 — Activate the virtual environment
Windows PowerShell:
.\.venv\Scripts\Activate


Step 4 — Install all required packages
pip install -r requirements.txt


________________


5. Quick Run Guide
A. Run the Streamlit Web Application
streamlit run ui_streamlit.py


Streamlit will launch at:
http://localhost:8501


Available tabs:
         1. Single Message

         2. Batch CSV Upload

         3. Metrics Dashboard

         4. About

________________


B. Use the Command-Line Classifier
python app.py


Example:
Message> Congratulations! You have won a free prize.
Prediction: spam
P(ham)=0.021 
P(spam)=0.979


________________


C. Re-train the Model (Optional)
If you want to regenerate the TF-IDF + Naive Bayes model:
python src/train.py


This creates/updates:
models/spam_nb_tfidf.pkl


________________


6. Evaluation
To compute and display model evaluation metrics:
python src/evaluate_model.py


Metrics include:
            * Accuracy

            * Precision

            * Recall

            * F1 Score

            * Confusion matrix

            * Dataset class distribution

These metrics also appear in the Streamlit Metrics Dashboard.
________________


7. Explainability
SpamShield includes a simple explainability module:
               * Extracts top TF-IDF weighted terms from the input message

               * Computes each word’s contribution toward spam or ham classification

               * Displays normalized contribution scores with visual bars

               * Helps interpret model behavior for transparency and debugging

________________


8. Requirements
All dependencies are listed in requirements.txt and include:
                  * Streamlit

                  * scikit-learn

                  * pandas

                  * numpy

                  * matplotlib

To install everything:
pip install -r requirements.txt


________________


9. Additional Notes for Instructors
To evaluate the project:
                     1. Install dependencies using the instructions above

                     2. Run the Streamlit application

                     3. Test single-message and batch functionality

                     4. Review model metrics and explainability outputs

                     5. (Optional) Re-run training for reproducibility

No extra configuration is required; all scripts are self-contained.
________________


10. License
MIT License.
This project is open for educational and non-commercial use.