# 📧 Email Spam Detector

A machine learning-powered web application that classifies emails as **Spam** or **Ham (Not Spam)** using NLP techniques and deep learning. Built with an end-to-end ML pipeline and deployed via Streamlit.

---

## 📌 Overview

Email Spam Detector processes raw email text through NLP preprocessing and feeds it into a trained classification model to determine whether a message is spam. The app provides an intuitive interface for real-time predictions, making it a practical tool for understanding text classification workflows.

---
## 🌐 Live Demo

👉 **Try the app here:** https://spamwall.streamlit.app/

---

## 🚀 Features

- **Real-Time Spam Detection** — Paste any email text and instantly classify it as Spam or Ham
- **NLP-Powered Pipeline** — Text cleaning, tokenization, and vectorization handled under the hood
- **Interactive Streamlit UI** — Clean and simple interface for quick predictions
- **Trained Deep Learning Model** — Uses `tf-keras` for the classification backbone
- **End-to-End Architecture** — Modular `src/` pipeline for training, preprocessing, and inference

---

## 🗂️ Project Structure

```
email-spam-detector/
│
├── app.py                  # Streamlit application entry point
├── requirements.txt        # Python dependencies
├── setup.py                # Package setup
│
├── src/                    # Core ML pipeline modules
│   ├── pipeline/           # Training and prediction pipelines
│   └── components/         # Data ingestion, transformation, model training
│
├── data/                   # Raw dataset (CSV)
├── artifacts/              # Saved model and vectorizer
├── venv/                   # Virtual environment (not committed ideally)
└── .github/workflows/      # CI/CD workflows
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.x

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/AtharvaVSawant/email-spam-detector.git
cd email-spam-detector

# 2. (Recommended) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
```

---

## 📦 Tech Stack

| Category | Tools |
|---|---|
| Frontend | Streamlit |
| Deep Learning | TensorFlow / tf-keras |
| ML / NLP | Scikit-learn, NLTK |
| Data Processing | NumPy, Pandas |
| Visualization | Plotly |

---

## 🧠 Model Pipeline

1. **Data Ingestion** — Load the SMS/email spam dataset
2. **Text Preprocessing** — Lowercasing, punctuation removal, stopword removal, stemming (NLTK)
3. **Feature Extraction** — TF-IDF or similar vectorization
4. **Model Training** — Neural network built with tf-keras
5. **Artifact Saving** — Model and vectorizer serialized for inference
6. **Prediction** — Pipeline loads artifacts and classifies new email input

---

## 📊 Input / Output

**Input:** Raw email or SMS text (plain string)

**Output:** 
- ✅ `Ham` — Legitimate message
- 🚫 `Spam` — Unwanted / junk message

---

## 💡 Key Concepts

- **Tokenization** — Breaking text into individual words/tokens
- **Stopword Removal** — Filtering out common words (the, is, and...) that carry little meaning
- **Stemming** — Reducing words to their root form (running → run)
- **TF-IDF** — Weighting terms by frequency and uniqueness across documents
- **Binary Classification** — Model outputs spam (1) or ham (0)

---

## 👤 Author

**Atharva Sawant**
- 📧 atharvasawant3183@gmail.com
- 📞 +91 9653320569

---

## 📄 License

This project is open-source and available for portfolio and educational purposes.
