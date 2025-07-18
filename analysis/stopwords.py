# analysis/stopwords.py
import unidecode
from config.settings import DEFAULT_STOPWORDS

def clean_text(text):
    text = str(text).lower().strip()
    text = unidecode.unidecode(text)
    return text

def remove_stopwords(words, custom_stopwords=None):
    stopwords = set(DEFAULT_STOPWORDS)
    if custom_stopwords:
        stopwords = stopwords.union(set(custom_stopwords))
    return [w for w in words if w not in stopwords and len(w) > 1]
