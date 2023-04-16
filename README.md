# SentenceRepresentations_NLI

1. Download GloVe embeddings
cd dataset
wget -P /dataset/ -q https://nlp.stanford.edu/data/glove.840B.300d.zip
cd ..

2. NLTK install
Run in python terminal:
import nltk
nltk.download('punkt')

3. Change lines in senteval utils.py
