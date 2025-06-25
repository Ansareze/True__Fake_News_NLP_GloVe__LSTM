# True/Fake News Detection with GloVe & LSTM

## Requirements
- Python 3.x
- TensorFlow, Keras, Pandas, NumPy, NLTK, Seaborn, Matplotlib, scikit-learn, BeautifulSoup4, wordcloud

## Dataset
- Place `True.csv` and `Fake.csv` in the project directory.

## GloVe Embeddings
This project requires the GloVe Twitter 27B 100d embeddings.

1. Download from [Stanford NLP GloVe page](https://nlp.stanford.edu/data/glove.twitter.27B.zip).
2. Unzip the file.
3. Place `glove.twitter.27B.100d.txt` in the project directory (same folder as the notebook).

**Note:** The GloVe file is large (~350MB) and is not included in this repository. It is listed in `.gitignore`.

## Running the Notebook
Open `true__fake_news_NLP_GloVe__LSTM.ipynb` and follow the steps in order.
