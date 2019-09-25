from spam import spam_identification
from flask import Flask, redirect, url_for, request, render_template

app = Flask(__name__ , template_folder='templates')

@app.route('/')
def home():
	return render_template('main.html')

@app.route('/spamm',methods=['POST'])
def spamm():
		string = request.form['keyword']
		return render_template('main.html', prediction_text=spam_identification(string))

if __name__ == '__main__':
	app.run(debug=True)