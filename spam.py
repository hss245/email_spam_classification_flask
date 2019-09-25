import pandas as pd
import numpy as np
from sklearn import naive_bayes
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv('spam.csv',encoding = 'ISO-8859-1')
data['Class'] = data['v1']
data['Message'] = data['v2']

del data['Unnamed: 2']
del data['Unnamed: 3']
del data['Unnamed: 4']
del data['v1']
del data['v2']

vectorizer = TfidfVectorizer(analyzer='word',stop_words='english')
mails_tfidf = vectorizer.fit_transform(data['Message'].values.tolist())

data['Class'] = data['Class'].replace('spam',1)
data['Class'] = data['Class'].replace('ham',0)

train_x, test_x, train_y, test_y = train_test_split(mails_tfidf.toarray(),data['Class'].values,test_size=0.20,random_state = 255)

model = naive_bayes.GaussianNB()
model.fit(train_x,train_y)
model.score(train_x,train_y)

pediction = model.predict(test_x)

def spam_identification(name):
    matrix = vectorizer.transform([name.lower()]).toarray()
    if model.predict(matrix) == 0:
        return 'HAM'
    else:
        return 'SPAM'   


