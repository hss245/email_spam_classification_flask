from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from flask import Flask, redirect, url_for, request, render_template
import pickle

app = Flask(__name__ , template_folder='templates')
model = pickle.load(open('email_spam.pkl', 'rb'))
data = pd.read_csv('spam.csv',encoding = 'ISO-8859-1')
data['Class'] = data['v1']
data['Message'] = data['v2']
vectorizer = TfidfVectorizer(analyzer='word',stop_words='english')
mails_tfidf = vectorizer.fit_transform(data['Message'].values.tolist())

@app.route('/')
def home():
	return render_template('main.html')

@app.route('/spamm',methods=['POST'])
def spamm():
		string = request.form['keyword']
		string = vectorizer.transform([string.lower()]).toarray()
		output = model.predict(string)
		if output[0] == 0:
			return render_template('index.html', prediction_text='Ham')
		else:
			return render_template('index.html', prediction_text='Spam')

if __name__ == '__main__':
	app.run(debug=True)
