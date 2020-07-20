import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from unidecode import unidecode
import string
import numpy as np


def pre_process(corpus):
    '''
    Function removes basic stop words, punctuation signs
    :param corpus: text string to be pre-processed
    :return: pre-processed string
    '''
    corpus = corpus.lower()
    punkts = string.punctuation
    punkts = punkts.replace('#', '') # assume we have C#, C++ programmer
    punkts = punkts.replace('+', '')
    stop_tokens = stopwords.words('english') + list(punkts)
    corpus = " ".join([w for w in word_tokenize(corpus) if w not in stop_tokens])
    corpus = unidecode(corpus)
    return corpus


def naive_metric(text1, text2):
    '''
    Normalizes words and counts cooccurance of words between two texts
    :param text1:
    :param text2:
    :return: ratio of matching words  / by total words
    '''
    text1 = text1.lower()
    text2 = text2.lower()
    punkts = string.punctuation
    punkts = punkts.replace('#', '') # assume we have C#, C++ programmer
    punkts = punkts.replace('+', '')
    stop_tokens = stopwords.words('english') + list(punkts)
    lemmatizer = WordNetLemmatizer()
    corpus1 = [lemmatizer.lemmatize(w) for w in word_tokenize(text1) if w not in stop_tokens]
    corpus2 = [lemmatizer.lemmatize(w) for w in word_tokenize(text2) if w not in stop_tokens]
    N_word_matches = 0
    for w in corpus1:
        if w in corpus2:
            N_word_matches += 1
    return N_word_matches/len(corpus1)




def sentence_vectorizer(sent, model):
    e = 0.0000001
    N = 300
    sent = pre_process(sent)
    sent_vec = np.zeros(N)
    embeded_w_count = 0
    for w in sent:
        try:
            v = model[w][:N]
            sent_vec = np.add(sent_vec, v)
            embeded_w_count += 1
        except:
            pass
        return sent_vec / (np.sqrt(sent_vec.dot(sent_vec)) + e)

