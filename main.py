



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
stop_words=set(stopwords.words('english'))
train=pd.read_csv("./train.csv")
# 1=Fake news
# 0=Real news
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(fill_value='',strategy='constant')
train=pd.DataFrame(imputer.fit_transform(train),columns=train.columns)

#merging columns
train['content']=train['title']+''+train['author']

port_stem = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stop_words]
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


#streamlit formatting
import streamlit as st
st.title('Fake News Detector')
st.write('# Find out if your news article is fake or not',)

author=st.text_input("News article author's name",key='author')
news_title=st.text_input("News article title",key='news_title')
news_content=st.text_input("Content of the news article",key='news_content')

if st.button('Check Authenticity of the news article'):
  # Preprocess user input
    user_input = stemming(f"{author} {news_title}")
    user_input = vectorizer.transform([user_input])    
    prediction = model.predict(user_input)
    if prediction[0] == 1:
        st.write('### The news article is likely fake.')
    else:
        st.write('### The news article is likely real.')
