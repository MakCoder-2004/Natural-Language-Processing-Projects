# Fake News Detection

## Overview
This project implements a machine learning model to detect fake news articles. The model is trained to classify news articles as either "real" or "fake" based on their content. The project uses Natural Language Processing (NLP) techniques to preprocess and analyze the text data, and implements a Linear Support Vector Classifier (LinearSVC) for the classification task.

## Features
- Text preprocessing with spaCy (stopword removal, lemmatization, punctuation removal)
- TF-IDF vectorization for feature extraction
- LinearSVC model for classification
- Model evaluation with classification report and confusion matrix
- Handles both title and article text for better prediction accuracy

## Dataset
The model is trained on the `fake_or_real_news.csv` dataset, which contains labeled news articles classified as either "FAKE" or "REAL".

## Requirements
- Python 3.8+
- pandas
- scikit-learn
- spaCy
- en_core_web_lg (spaCy's large English language model)

## Installation
* Clone this repository

* Download the spaCy English language model:
   ```bash
   python -m spacy download en_core_web_lg
   ```

## Usage
1. Run the Jupyter notebook `main.ipynb`
2. The notebook includes all necessary code for:
   - Data loading and exploration
   - Text preprocessing
   - Model training
   - Model evaluation

## Model Performance
The model's performance is evaluated using standard classification metrics including precision, recall, and F1-score. The confusion matrix provides additional insights into the model's classification behavior.

## Future Improvements
- Experiment with different machine learning models (e.g., LSTM, BERT)
- Implement hyperparameter tuning
- Add a web interface for real-time predictions
- Include more sophisticated text preprocessing techniques
- Handle class imbalance if present in the dataset
