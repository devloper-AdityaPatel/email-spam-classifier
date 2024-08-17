import pickle
import nltk
import string
import streamlit as st
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = list()
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    ps = PorterStemmer()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/Sms Spam Classifier")
st.subheader("Designed and Deployed by Aditya Patel")
input_sms =st.text_area("Enter the message")
if st.button("predict"):
    # preprocess
    transformed = transform_text(input_sms)

    # vectorize

    vector_input = tfidf.transform([transformed])
    # predict

    result = model.predict(vector_input)[0]
    # display
    if(result):
        st.header("Spam")
    else:
        st.header("Not Spam")
    
