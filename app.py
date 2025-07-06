import numpy as np
import streamlit as st
import pickle
from keras.src.utils import pad_sequences


# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    if not text.strip():
        return "(Empty input)"
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

from tensorflow.keras.models import load_model
model = load_model("next_word_lstm.h5")

DEFAULT_TEXT = "to be or not to"

# Define a function to clear input/output when the button is clicked
def clear_all():
    st.session_state.input_text = ""
    st.session_state.output_text = ""

if "input_text" not in st.session_state:
    st.session_state.input_text = "to be or not to"
if "output_text" not in st.session_state:
    st.session_state.output_text = ""
if "clear_flag" not in st.session_state:
    st.session_state.clear_flag = False

st.title('Welcome to Next Word Predictor using LSTM RNN')

# Input field tied to session state
input_text = st.text_input(
    'Enter the sequence of words',
    value=st.session_state.get("input_text", DEFAULT_TEXT),
    key="input_text"
)

# Predict button
if st.button('Predict'):
    if input_text.strip() == "":
        st.warning("Please enter a valid input.")
    else:
        max_sequence_len = model.input_shape[1] + 1
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        st.session_state.output_text = f"Next Word: {next_word}"

# Clear button — uses a callback function
st.button('Clear', on_click=clear_all)

# Display prediction output
if "output_text" in st.session_state and st.session_state.output_text:
    st.write(st.session_state.output_text)


