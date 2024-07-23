import numpy as np
import pickle
import streamlit as st
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

#Loading the save model
preprocess = pickle.load(open('C:/Users/prana/Downloads/vectorizer.sav', 'rb'))
loaded_model = pickle.load(open('C:/Users/prana/Downloads/trained_model.sav', 'rb'))

# Initialize the lemmatizer and stopwords
lemma = WordNetLemmatizer()
Stopwords = set(stopwords.words('english'))

def preprocess_text(text):
    string = ""

    text=text.lower()

    text=re.sub(r"i'm","i am",text)
    text=re.sub(r"he's","he is",text)
    text=re.sub(r"she's","she is",text)
    text=re.sub(r"that's","that is",text)
    text=re.sub(r"what's","what is",text)
    text=re.sub(r"where's","where is",text)
    text=re.sub(r"\'ll"," will",text)
    text=re.sub(r"\'ve"
                ," have",text)
    text=re.sub(r"\'re"," are",text)
    text=re.sub(r"\'d"," would",text)
    text=re.sub(r"won't","will not",text)
    text=re.sub(r"can't","cannot",text)

    text=re.sub(r"[-()\"#!@$%^&*{}?.,:]"," ",text)
    text=re.sub(r"\s+"," ",text)
    text=re.sub('[^A-Za-z0-9]+',' ', text)

    for word in text.split():
        if word not in Stopwords:
            string += lemma.lemmatize(word) + " "

    return string.strip()

def predict_news(text):
    
    # Preprocess the input text
    pro_text = preprocess_text(text)
    
    # Transform the text using the vectorizer
    text_vectorized = preprocess.transform([pro_text])
    
    # Convert sparse matrix to dense array
    text_vectorized = text_vectorized.toarray()
    
    # Make a prediction using the trained model
    prediction = loaded_model.predict(text_vectorized)
    
    # Return the prediction result
    if prediction[0] == 1:
        return "Fake News"
    else:
        return "Real News"
    
def main():
    #giving a title
    st.title('Fake News Detection')
    
    #Getting the input data from the user
    input = st.text_input("Write the News : ")
    
    result = ''
    if st.button('Check the News'):
        result = predict_news(input)
        
    st.success(result)

if __name__ == '__main__' :
    main()