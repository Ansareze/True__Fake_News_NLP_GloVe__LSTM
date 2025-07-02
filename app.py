import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

@st.cache_resource
def load_model_and_tokenizer():
    model = load_model('fake_news_lstm.h5')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()
maxlen = 300

st.title("Fake News Detector")

user_input = st.text_area("Enter news article text:")

if st.button("Predict"):
    seq = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(seq, maxlen=maxlen)
    pred = model.predict(padded)[0][0]
    label = "Fake" if pred < 0.5 else "Not Fake"
    st.write(f"Prediction: **{label}** (probability: {pred:.4f})")