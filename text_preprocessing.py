# Import libraries
import os
import numpy as np
import re
import unicodedata
import nltk
# from nltk.corpus import stopwords
import itertools
from tqdm import tqdm
from nltk.corpus import stopwords
nltk.download('stopwords')
# nltk.download('stopwords')
stopword = stopwords.words('english')

# Define text preprocessing functions


def unicode_to_ascii(sentence):
    """ 
    Input: a string (one sentence) in Unicode character encoding
    Output: a string (one sentence) in ASCII character encoding
    """
    return ''.join(character for character in unicodedata.normalize('NFD', sentence) if unicodedata.category(character) != 'Mn')


def clean_stopwords_shortwords(sentence):
    """ 
    Input: a string (one sentence)
    Output: a string (one sentence) without stop words and words shorter than 2 characters
    """
    stopwords_list = stopwords.words('english')
    words = sentence.split()
    clean_words = [word for word in words if (
        word not in stopwords_list) and len(word) > 2]
    return " ".join(clean_words)


def preprocess_sentence(sentence):
    """
    Input: a raw sentence
    Output: a clean sentence ready to be passed to a tokenizer
    """
    sentence = unicode_to_ascii(sentence.lower().strip())
    sentence = re.sub(r"([?.!,¿])", r" ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence)
    sentence = clean_stopwords_shortwords(sentence)
    sentence = re.sub(r'@\w+', '', sentence)
    if 'http' in sentence:
        sentence = sentence.split(' http')[0]
    return sentence
