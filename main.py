#streamlit formatting
import streamlit as st
st.title('Fake News Detector')
st.write('# Find out if your news article is fake or not',)

author=st.text_input("News article author's name",key='author')
news_title=st.text_input("News article author's name",key='news_title')
news_content=st.text_input("Content of the news article",key='news_content')



import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ClassifierMixin
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')

train=pd.read_csv("./train.csv")
# 1=Fake news
# 0=Real news
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(fill_value='',strategy='constant')
train=pd.DataFrame(imputer.fit_transform(train),columns=train.columns)

#merging columns
train['content']=train['title']+''+train['author']+''+train['text']

port_stem = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

train['content']=train['content'].apply(stemming)

X=train['content'].values
Y=train['label'].values
Y = Y.astype(int)

# vectorization
vectorizer=TfidfVectorizer()
X=vectorizer.fit_transform(X)

model=LogisticRegression()
model.fit(X,Y)


st.button('Check Authenticity of the news article')

user_input=author+' '+news_title+' '+news_content
user_input=user_input.apply(stemming)
output=model.predict(user_input)
st.write(output)  
