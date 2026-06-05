# Multi-Label Emotion Detection (NLP)

Multi-label emotion classification on the GoEmotions dataset (Google)
using classical ML models and BERT fine-tuning with HuggingFace Transformers.


# Technologies
Python | PyTorch | HuggingFace Transformers | BERT | Scikit-learn | TF-IDF | NLTK | Pandas | Seaborn


# Models Used
| Model | Approach |
|---|---|
| Logistic Regression | TF-IDF + OneVsRest |
| SVM (LinearSVC) | TF-IDF + OneVsRest |
| Random Forest | TF-IDF + OneVsRest |
| Naive Bayes | TF-IDF + OneVsRest |
| KNN | TF-IDF + OneVsRest |
| **BERT** | Fine-tuned with HuggingFace |


# Dataset
- **GoEmotions** (Google Research) — 58,000+ Reddit comments
- **28 emotion labels** (multi-label)
- Split : Train / Validation / Test


# Pipeline
1. Text cleaning (lowercase, remove URLs, punctuation)
2. Stopwords removal (NLTK)
3. TF-IDF Vectorization (10,000 features)
4. Multi-hot encoding for multi-label targets
5. Classical ML models comparison
6. BERT fine-tuning with HuggingFace Transformers


# Evaluation Metric
- **F1-Score** (adapted for imbalanced multi-label classification)

---

# Structure
emotion-detection-nlp/
│── emotion_project.ipynb
│── README.md
