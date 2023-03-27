from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

#Naive Bayes algorithm estimates the probability of a class given an input by considering each feature (or attribute) independently. 
#The algorithm calculates the probability of each class and the conditional probability of each feature given each class, 
#and then it classifies the input to the class that has the highest probability.


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	df= pd.read_csv("spam.csv", encoding="latin-1")
	df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
	# Features and Labels
	df['label'] = df['v1'].map({'ham': 0, 'spam': 1})
	X = df['v2']
	y = df['label']
	
	# Extract Feature With CountVectorizer
	cv = CountVectorizer()  #convert a collection of text documents to a matrix of token counts 
	X = cv.fit_transform(X) # Fit the Data
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)  #splitting the data into training and testing sets
 
	#Naive Bayes Classifier
	from sklearn.naive_bayes import MultinomialNB

	clf = MultinomialNB()   #method from the sklearn library used to create a Naive Bayes classifier for text data. 
	clf.fit(X_train,y_train)  # evaluate the performance of the classifier
	clf.score(X_test,y_test)  
 
	#Alternative Usage of Saved Model
	# joblib.dump(clf, 'NB_spam_model.pkl')
	# NB_spam_model = open('NB_spam_model.pkl','rb')
	# clf = joblib.load(NB_spam_model)

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)