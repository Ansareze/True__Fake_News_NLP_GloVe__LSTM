from tensorflow.keras.models import load_model
import pickle

print("Loading model...")
model = load_model('fake_news_lstm.h5')
print("Model loaded.")
print("Loading tokenizer...")
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
print("Tokenizer loaded.")