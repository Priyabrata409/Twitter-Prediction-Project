## Feature Engineering  and Model building
## Importing Libraries
import pandas as pd
import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

data=pd.read_csv("Twitter_data.csv")
## I can see there are lot of similar values in home_location and tweet_location like "Mumbai India" and "Mumbai India" the only difference is space betwwen them
def format_loaction(x):
    words=x.split()
    words=[word.strip() for word in words]
    return " ".join(words)
data["home_location"]=data["home_location"].apply(lambda x: format_loaction(x))
data["tweet_location"]=data["tweet_location"].apply(lambda x: format_loaction(x))

# Determining the locations which are starting from Chennai
chennai_val=[val for val in data['home_location'].unique() if (val.split(" ")[0]=="Chennai") | (val.split(" ")[0]=="chennai")|(val.split(" ")[0]=="CHENNAI")]
# Determining the locations which are starting from Hyderabad
hyderabad_val=[val for val in data['home_location'].unique() if (val.split(" ")[0]=="Hyderabad") | (val.split(" ")[0]=="hyderabad")|(val.split(" ")[0]=="HYDERABAD")]
# Determining the locations which are starting from Mumbai
mumbai_val=[val for val in data['home_location'].unique() if (val.split(" ")[0]=="Mumbai") | (val.split(" ")[0]=="mumbai")|(val.split(" ")[0]=="MUMBAI")]

data["home_location"].replace(mumbai_val,"Mumbai India",inplace=True)
data["home_location"].replace(hyderabad_val,"Hyderabad India",inplace=True)
data["home_location"].replace(chennai_val,"Chennai India",inplace=True)
data["tweet_location"].replace(mumbai_val,"Mumbai India",inplace=True)
data["tweet_location"].replace(hyderabad_val,"Hyderabad India",inplace=True)
data["tweet_location"].replace(chennai_val,"Chennai India",inplace=True)
## Preparing Target Column
data["mentioned_location"].replace({"Hyderabad":1,"Mumbai":2,"Chennai":3},inplace=True)
sample=data["text"]+" "+data["home_location"]+" "+data["tweet_location"]

#Cleaning Stopwords from text
sentences=[]
for sen in sample:
    sen=sen.split()
    sen=[word for word in sen if word not in set(stopwords.words("english"))]
    sen=" ".join(sen)
    sentences.append(sen)

###Applying Bag of Words Techniques
print("Applying Bag of Words Techniques\n")

vectorizer=CountVectorizer(max_features=2000)
X=vectorizer.fit_transform(sentences).toarray()
y=data["mentioned_location"]
from sklearn.model_selection import train_test_split
train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.25,random_state=100)

# Applying GaussianNB model
gauss_model=GaussianNB()
gauss_model.fit(train_X,train_y)
pred=gauss_model.predict(test_X)
s1=accuracy_score(test_y,pred)
print(f"Accuracy Score using GaussianNB model   is {s1*100}%")


# Applying MultinomialNB model
multi_gauss_model=MultinomialNB()
multi_gauss_model.fit(train_X,train_y)
pred=multi_gauss_model.predict(test_X)
s2=accuracy_score(test_y,pred)
print(f"Accuracy Score using MultinomialNB   is {s2*100}%")

# Applying SVM model
svc=SVC()
svc.fit(train_X,train_y)
pred=svc.predict(test_X)
s3=accuracy_score(test_y,pred)
print(f"Accuracy Score using Support Vector Machine is {s3*100}%")

# Applying Decision Tree model
dt=DecisionTreeClassifier()
dt.fit(train_X,train_y)
pred=dt.predict(test_X)
s4=accuracy_score(test_y,pred)
print(f"Accuracy Score using Decision Tree is {s4*100}%\n")

models=["GaussianNB","MultinomialNB","SVC","DecisionTreeClassifier"]
scores=[s1,s2,s3,s4]
plt.bar(x=models,height=scores,color=["red","blue","yellow","green"])
plt.title("Using Bag of Words Model")
plt.ylabel("Accuracies")
plt.show()

### Applying TF-IDF(Term frequency and Inverse document frequency) Techniques which gives semantic meaning to words
print("Applying TF-IDF(Term frequency and Inverse document frequency) Techniques\n")

tf_vectorizer=TfidfVectorizer()
X=tf_vectorizer.fit_transform(sentences).toarray()

from sklearn.model_selection import train_test_split
train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.25,random_state=100)

# Applying GaussianNB model
gauss_model=GaussianNB()
gauss_model.fit(train_X,train_y)
pred=gauss_model.predict(test_X)
s1=accuracy_score(test_y,pred)
print(f"Accuracy Score using GaussianNB model   is {s1*100}%")
# Applying MultinomialNB model
multi_gauss_model=MultinomialNB()
multi_gauss_model.fit(train_X,train_y)
pred=multi_gauss_model.predict(test_X)
s2=accuracy_score(test_y,pred)
print(f"Accuracy Score using MultinomialNB   is {s2*100}%")

# Applying SVM model
svc=SVC()
svc.fit(train_X,train_y)
pred=svc.predict(test_X)
s3=accuracy_score(test_y,pred)
print(f"Accuracy Score using Support Vector Machine is {s3*100}%")

# Applying Decision Tree model
dt=DecisionTreeClassifier()
dt.fit(train_X,train_y)
pred=dt.predict(test_X)
s4=accuracy_score(test_y,pred)
print(f"Accuracy Score using Decision Tree is {s4*100}%")
## Plotting models performances

models=["GaussianNB","MultinomialNB","SVC","DecisionTreeClassifier"]
scores=[s1,s2,s3,s4]
plt.bar(x=models,height=scores,color=["red","blue","yellow","green"])
plt.title("Using TF-IDF Techniques")
plt.ylabel("Accuracies")
plt.show()