import streamlit as st
import numpy as np
import pickle
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = load_model("model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load config
with open("config.json", "r") as f:
    config = json.load(f)

max_sequence_len = config["max_sequence_len"]

# Reverse word index
index_word = {v: k for k, v in tokenizer.word_index.items()}

# Prediction function
def predict_next_words(seed_text, next_words=1):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted)

        next_word = index_word.get(predicted_word_index, "")
        seed_text += " " + next_word

    return seed_text

# UI
st.title("Next Word Prediction with Deep Learning")

input_text = st.text_input("Enter a Starting Text:")

num_words = st.slider("Number of words to generate", 1, 20, 5)

if st.button("Generate"):
    if input_text:
        output = predict_next_words(input_text, num_words)
        st.write(output)
    else:
        st.warning("Please enter some text")