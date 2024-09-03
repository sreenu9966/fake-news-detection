from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import itertools
import nltk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc

# load the dataset
df = pd.read_csv('C:\\news.csv')
# clean the dataset
df.drop_duplicates(inplace=True)
# preprocess the textual column
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    preprocessed_text = ' '.join(lemmatized_tokens)
    return preprocessed_text
df['text'] = df['text'].apply(preprocess_text)

# split the dataset
x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=7)

# initialize a TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
# fit and transform train set, transform test set
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

# initialize a PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('articles.html')

@app.route('/results', methods=['POST'])
def results():
    title = request.form['title']
    text = request.form['text']

    # preprocess input data
    input_data = preprocess_text(title + " " + text)
    # vectorize input data
    input_vector = tfidf_vectorizer.transform([input_data])
    # predict on the input data
    prediction = pac.predict(input_vector)

    if prediction[0] == 'FAKE':
        result = "The News is Fake."
    else:
        result = "The News is Real."

    # render the results template with the result data
    return render_template('results.html', result=result, title=title, text=text)

if __name__ == '__main__':
    app.run(debug=True)
