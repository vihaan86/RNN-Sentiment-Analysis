import numpy as np
import pandas as pd
import tensorflow as tf
from keras.datasets import imdb
from keras.utils import pad_sequences
from keras.models import load_model
from keras.layers import Embedding,SimpleRNN,Dense

#Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index= {value: key for key,value in word_index.items()}

#Load the pre-trained model with relu activation
model = load_model("simple_rnn_imdb.h5")

#Step2:Helper function
#Function to decode reviews

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,"?") for i in encoded_review])

#Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2)+ 3 for word in words]
    padded_review = pad_sequences([encoded_review],maxlen=500)
    return padded_review


##Design streamlit app
import streamlit as st

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as positive or negative")

user_input=st.text_area("Movie review")

if st.button("Classify"):
    preprocessed_input=preprocess_text(user_input)

    prediction=model.predict(preprocessed_input)
    sentiment="Positive" if prediction[0][0] > 0.5 else "Negative"
    
    st.write(f"Sentiment:{sentiment}")
    st.write(f"Prediction Score:{prediction[0][0]}")

else:
    st.write("Please enter a movie review")




