import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

st.title("📝 Live Next-Word Prediction with LSTM")

st.sidebar.header("🔧 Settings")
model_path = st.sidebar.text_input("Enter Model Path:", r"E:\Experiments\NLP\Next_Word_Prediction\models\next_word_lstm.h5")
tokenizer_path = st.sidebar.text_input("Enter Tokenizer Path:", r"E:\Experiments\NLP\Next_Word_Prediction\models\tokenizer.pickle")

@st.cache_resource
def load_model(model_path, tokenizer_path):
    if os.path.exists(model_path) and os.path.exists(tokenizer_path):
        model = tf.keras.models.load_model(model_path)
        with open(tokenizer_path, "rb") as handle:
            tokenizer = pickle.load(handle)
        return model, tokenizer
    else:
        st.error("❌ Invalid Model/Tokenizer Path!")
        return None, None

model, tokenizer = load_model(model_path, tokenizer_path)

if model and tokenizer:
    max_seq_length = model.input_shape[1] + 1

    def predict_next_word(seed_text):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_length - 1, padding="pre")
        predicted = np.argmax(model.predict(token_list), axis=-1)

        for word, index in tokenizer.word_index.items():
            if index == predicted:
                return word
        return "..."

    user_input = st.text_input("Type a sentence:", "")

    if user_input.strip():
        next_word = predict_next_word(user_input)
        st.markdown(f"**Predicted Next Word:** `{next_word}`")
