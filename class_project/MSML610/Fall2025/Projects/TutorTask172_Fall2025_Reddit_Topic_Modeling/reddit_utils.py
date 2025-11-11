import re
import nltk
from nltk.corpus import stopwords
import numpy as np

nltk.download('stopwords', quiet=True)

def clean_text(text):
    """Clean Reddit comment text."""
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    tokens = text.lower().split()
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    return ' '.join(tokens)

def average_vector(model, text):
    """Compute average fastText embedding for a text."""
    words = text.split()
    word_vecs = [model.get_word_vector(w) for w in words if w in model.words]
    return np.mean(word_vecs, axis=0) if word_vecs else np.zeros(model.get_dimension())
