data = [
    "Computer programmer",
    "Computer programmer intern",
    "Software developer",
    "starting Software developer",
    "IT specialist",
    "C++ Software Engineer",
    "Front-end developer",
    "Full stack engineer",
    "Software team lead",
    "Controls Software Engineer",
    "Fire Protection Journeyman",
    "Accountant",
    "Casino dealer",
    "Air package handler",
    "Owner",
    "Executive Chef",
    "Clinical Pharmacist",
    "Therapist Surgeon",
    "Radiology Information Systems"
]
from Preprocessing import sentence_vectorizer, pre_process, naive_metric
import gensim
from math import sqrt
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, paired_distances
import numpy as np
import pandas as pd
from BertEncoding import BertEncoder
import textdistance
from collections import Counter

bb = BertEncoder()


def cosine(vec1, vec2):
    '''
    Cosine similarity for two vectors
    :return: (0, 1) value
    '''
    e = 0.0000001
    c = 0
    # cosine formula
    for i in range(len(vec1)):
        c += vec1[i] * vec2[i]
    return c / float(sqrt(sum([e**2 for e in vec1])*sum([e**2 for e in vec2]))+e)


def best_match(title, names_pool):
    '''
    Best match implmentation using word2vec encoder.
    Requires GoogleNews-vectors-negative300.bin.gz model to be downloaded
    :param title: searched occupation name
    :param names_pool: where we searching
    :return: best match occupation name from pool
    '''
    scores1 = []
    scores2 = []
    model = gensim.models.KeyedVectors.load_word2vec_format(
        '/home/sula/Downloads/GoogleNews-vectors-negative300.bin.gz', binary=True)
    l1 = sentence_vectorizer(title, model)
    for name in names_pool:
        l2 = sentence_vectorizer(name, model)
        l1 = np.array(l1).reshape(-1, 1)
        l2 = np.array(l2).reshape(-1, 1)
        scores1.append(np.sum(euclidean_distances(l1, l2)))
        scores2.append(np.sum(paired_distances(l1, l2)))
    return names_pool[np.argmax(scores1)],  names_pool[np.argmax(scores2)]


def best_match_bert(title, names_pool):
    '''
    Best matching using bert, some simple text distance measures
    :param title: searched occupation name
    :param names_pool: where are we searching
    :return: best match occupation name from pool
    '''
    scores1 = []
    scores2 = []
    scores3 = []
    scores4 = []
    scores5 = []
    scores6 = []
    scores7 = []
    # 1,2, 8 - scores > 1
    title = pre_process(title)
    l1 = bb.bert_encoder(title)
    for name in names_pool:
        name = pre_process(name)
        l2 = bb.bert_encoder(name)
        l1 = np.array(l1).reshape(-1, 1)
        l2 = np.array(l2).reshape(-1, 1)
        scores1.append(np.sum(euclidean_distances(l1, l2)))
        scores2.append(np.sum(paired_distances(l1, l2)))
        scores3.append(np.sum(cosine(l1, l2)))
        scores4.append(np.sum(naive_metric(title, name)))
        scores5.append(np.sum(textdistance.jaccard(title, name)))
        scores6.append(np.sum(textdistance.sorensen_dice(title, name)))
        scores7.append(np.sum(textdistance.damerau_levenshtein(title, name)))

    chosen_idx = [names_pool[np.argmin(scores1)], names_pool[np.argmin(scores2)], names_pool[np.argmax(scores3)],
                  names_pool[np.argmax(scores4)], names_pool[np.argmax(scores5)],
                  names_pool[np.argmax(scores6)], names_pool[np.argmin(scores7)]]
    print("Chosen names \n", chosen_idx)
    print("------------------------------")
    c = Counter(chosen_idx)
    frequency = c.most_common(1)[0][1]
    if frequency < 3:
        return names_pool[np.argmax(scores3)]
    return c.most_common(1)[0][0]

test_titles_df = pd.read_csv('test_list.csv')
occup_pool_df = pd.read_csv('Occupation Data.csv', names=['title', 'description'], delimiter=';', skiprows=1, header=None)

#print(best_match_bert('computer scientist', data))
#print(best_match_bert('C++ Software Engineer', occup_pool_df['title']))


def test(test_file, occupation_pool):
    '''
    Test some tricky occupation names from test file
    :param test_file: has test pairs
    :param occupation_pool: where are we searching
    :return: number of correctly identified out of total
    '''
    test_df = pd.read_csv(test_file, delimiter=';')
    correct_matches = 0
    print("entered title    |    target title  |    predicted")
    for idx, row in test_df.iterrows():
        pred = best_match_bert(row['title'], occupation_pool)
        print(row['title']+" | ", row['target']+" | ", pred)
        if pred == row['target']:
            correct_matches += 1
    print(f"Correct matches {correct_matches}/{len(test_df['title'])}")

test('test_occupations.csv', occup_pool_df['title'])