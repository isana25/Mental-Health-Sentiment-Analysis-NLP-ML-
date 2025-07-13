# 🧠 Mental Health Sentiment Analysis

This project analyzes social media statements to predict individuals’ mental health status using advanced Natural Language Processing (NLP) and machine learning techniques. By classifying text into categories such as Normal, Depression, Suicidal, Anxiety, Stress, Bi-Polar, and Personality Disorder, the model aims to provide actionable insights that can support early intervention and mental health care.

## 📂 Dataset

- **Source:** [Kaggle - Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)
- Contains labeled social media statements with seven mental health categories.

## 🚀 Features

- Loads and explores the dataset for class distribution and text characteristics
- Cleans and preprocesses text (lowercasing, punctuation removal, etc.)
- Converts text into numerical features using NLP methods
- Trains multiple ML models (Logistic Regression, Naive Bayes, etc.) for classification
- Evaluates models with accuracy, precision, recall, F1-score, and confusion matrix
- Provides real-time sentiment prediction for new user input

## 🛠️ Tech Stack

- **Language:** Python (pandas, numpy)
- **NLP:** NLTK, scikit-learn
- **Machine Learning:** Logistic Regression, Naive Bayes, (optionally BERT/transformers)
- **Visualization:** matplotlib, seaborn
- **Platform:** Google Colab, Jupyter Notebook

## 📝 How to Run

1. **Clone this repo and open the notebook in Google Colab**
2. **Download the dataset using the Kaggle API:**
    - Upload your `kaggle.json` credentials in the notebook and run the download cell
3. **Install requirements:**
    ```python
    !pip install pandas numpy scikit-learn matplotlib seaborn nltk
    ```
4. **Run the notebook cells to reproduce analysis, model training, and predictions**

## 📊 Example Outputs

- Model accuracy, precision, recall, and confusion matrix for each category
- Real-time prediction for user-provided text

## 📈 Project Demo

You can try the notebook interactively here:  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eBdFPgmFukb28vlJFPg_IwxG6pk_RnpS?usp=sharing)
