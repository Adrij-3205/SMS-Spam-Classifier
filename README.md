# SMS Spam Checker

This project deploys a Flask-based SMS Spam Checker using a Bernoulli Naive Bayes model. The app preprocesses SMS text using NLTK, vectorizes it with TF-IDF, and then predicts whether a message is spam or not.

## Files Included

- `app.py`: The Flask application.
- `tfidf.pkl`: Saved TF-IDF vectorizer.
- `bnb_model.pkl`: Saved Bernoulli Naive Bayes model.
- `requirements.txt`: Python dependencies.
- `Procfile`: Startup command for Render.
- `analysis.ipynb`: Jupyter Notebook with data analysis and model training.
- `README.md`: This file.

## How to Run Locally

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2. Run the Flask app:
    ```bash
    python app.py
    ```
3. Open your browser at [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Deployment

This app is deployed on Render. 
