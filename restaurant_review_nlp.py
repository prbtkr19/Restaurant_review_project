#Natural language processing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#importing the dataset
dataset = pd.read_csv("F:\\Natural_Language_Processing\\Restaurant_Reviews.tsv" ,delimiter = '\t' , quoting =3)
#quoting=3 means ignoring double quotes
#cleaning the text
#import library re to clean text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1000):
   review = re.sub('[^a-zA-Z]',' ', dataset['Review'][i])
#converting all review into lower case
   review = review.lower()
#remove non significat word like articless,preposition ,this....
#split the review into different word
   review=review.split()
   ps=PorterStemmer()
   review =[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
   review =' '.join(review)
   corpus.append(review)
#set function used in if article is large like books to faster go to each words 
#stemming ia about taking root of word 
   #creating the bag of word model
   #matrix containing lots of zeros or matrix of independent features is called sparse matrix
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)#max_features gives most relevant words
X=cv.fit_transform(corpus).toarray()
#x is matrix of features
y =dataset.iloc[:,1].values

#splitting tha data into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test ,y_train ,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

 #feature scaling
from sklearn.preprocessing import StandardScalar
sc=StandardScalar()
X_train = sc.fit_transform(X_train)
X_test =sc.transform(X_test)
 
 #fitting Naivebayes to training set
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)
 
 #predicting the test set results
y_pred = classifier.predict(X_test)
 
 
 #making confusin matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
(67+113)/200
   

