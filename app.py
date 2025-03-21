# app.py

from flask import Flask, request, render_template_string
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize the PorterStemmer
ps = PorterStemmer()

def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)

# Load the pre-trained TF-IDF vectorizer and BernoulliNB model from pickle files
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('bnb_model.pkl', 'rb') as f:
    bnb = pickle.load(f)

# Initialize the Flask app
app = Flask(__name__)

# HTML template (using render_template_string for simplicity)
HTML_TEMPLATE = '''
<!doctype html>
<html>
    <head>
        <title>SMS Spam Checker</title>
    </head>
    <body>
        <h1>SMS Spam Checker</h1>
        <form method="post">
            <textarea name="sms_text" rows="4" cols="50" placeholder="Enter SMS message here"></textarea><br><br>
            <input type="submit" value="Check">
        </form>
        {% if prediction %}
            <h2>Prediction: {{ prediction }}</h2>
        {% endif %}
    </body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        sms_text = request.form.get('sms_text', '')
        # Preprocess the text using transform_text function
        transformed_text = transform_text(sms_text)
        # Vectorize the transformed text (returns a sparse matrix)
        vectorized_text = tfidf.transform([transformed_text])
        # Predict using the loaded BernoulliNB model
        pred = bnb.predict(vectorized_text)
        # Assuming 1 is Spam and 0 is Not Spam
        prediction = 'Spam' if pred[0] == 1 else 'Not Spam'
    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == '__main__':
    # Run the Flask development server
    app.run(debug=True)
