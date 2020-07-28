from Preprocessing import sentence_vectorizer, pre_process, naive_metric
import gensim
from math import sqrt
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, paired_distances
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
import numpy as np
import pandas as pd
from BertEncoding import BertEncoder
import textdistance
from collections import Counter
from multiprocessing import Process, Manager, JoinableQueue
import time
import ray




def timit(fnc):
    def wrapper(*args, **kwargs):
        s = time.time()
        res = fnc(*args, **kwargs)
        e = time.time()
        print(f"Time elapsed: {(e - s) * 1000:0.1f} ms")
        return res

    return wrapper


def calc_vectors(text_df: pd.DataFrame):
    bb = BertEncoder()
    result = []
    for _, v in text_df['Alternate Title'].items():
        v = pre_process(v)
        r = np.squeeze(bb.bert_encoder(v))
        result.append(r)
    result = np.array(result)
    np.savetxt('alter_bert_vectors.csv', result, delimiter=',')


df_alter = pd.read_excel('data/Alternate Titles.xlsx')


# calc_vectors(df_alter)


def cosine(vec1: np.ndarray, vec2: np.ndarray):
    '''
    Cosine similarity for two vectors
    :return: (0, 1) value; 1 meaning exactly the same
    '''
    e = 0.0000001
    c = np.dot(vec1, vec2)
    return c / float(sqrt(sum([e ** 2 for e in vec1]) * sum([e ** 2 for e in vec2])) + e)


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
    return names_pool[np.argmax(scores1)], names_pool[np.argmax(scores2)]


def best_match_bert(orig_title, names_pool):
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
    title = pre_process(orig_title)
    bb = BertEncoder()
    l1 = bb.bert_encoder(title)
    for name in names_pool:
        proc_name = pre_process(name)
        l2 = bb.bert_encoder(proc_name)
        l1 = np.array(l1).reshape(-1, 1)
        l2 = np.array(l2).reshape(-1, 1)
        scores1.append(np.sum(euclidean_distances(l1, l2)))
        scores2.append(np.sum(paired_distances(l1, l2)))
        scores3.append(np.sum(cosine(l1, l2)))
        scores4.append(np.sum(naive_metric(title, proc_name)))
        scores5.append(np.sum(textdistance.jaccard(title, proc_name)))
        scores6.append(np.sum(textdistance.sorensen_dice(title, proc_name)))
        scores7.append(np.sum(textdistance.damerau_levenshtein(title, proc_name)))
        textdistance.j

    chosen_idx = [names_pool[np.argmin(scores1)], names_pool[np.argmin(scores2)], names_pool[np.argmax(scores3)],
                  names_pool[np.argmax(scores4)], names_pool[np.argmax(scores5)],
                  names_pool[np.argmax(scores6)], names_pool[np.argmin(scores7)]]
    print("-----------------------------------------------------------------------------------------")
    print("All title similarity candidates \n", chosen_idx)
    c = Counter(chosen_idx)
    frequency = c.most_common(1)[0][1]
    if frequency < 3:
        return names_pool[np.argmax(scores3)]
    return c.most_common(1)[0][0]


# test_titles_df = pd.read_table('all_test_samples', names=['title'])[:25]
occup_pool_df = pd.read_csv('Occupation Data.csv', names=['title', 'description'], delimiter=';', skiprows=1,
                            header=None)


# print(best_match_bert('computer scientist', data))
# print(best_match_bert('C++ Software Engineer', occup_pool_df['title']))
# print(best_match('Sr. IT Recruiter', occup_pool_df['titles']))

def test(test_file, occupation_pool):
    '''
    Test some tricky occupation names from test file
    :param test_file: has test pairs
    :param occupation_pool: where are we searching
    :return: number of correctly identified out of total
    '''
    test_df = pd.read_csv(test_file, delimiter=';', names=['title'])[:25]
    correct_matches = 0
    print("Entered title                 |                Predicted")
    for idx, row in test_df.iterrows():
        pred = best_match_bert(row['title'], occupation_pool)
        print("| " + row['title'] + " | ", pred + " | ")



@ray.remote
def best_match_v3(input_title, precalc_vectors: np.ndarray, title_names_pool):
    input_title = pre_process(input_title)
    bb = BertEncoder()
    vec = bb.bert_encoder(input_title)

    trim_n = 300 #precalc_vectors.shape[1]
    scores = []
    wass_scores = []
    jsh_scores = []
    for i in range(precalc_vectors.shape[0]):
        s_d = cosine(vec[:trim_n], precalc_vectors[i][:trim_n])
        w_d = wasserstein_distance(vec[:trim_n], precalc_vectors[i][:trim_n])
        jsh_d = jensenshannon(np.abs(vec[:trim_n]), np.abs(precalc_vectors[i][:trim_n]))
        scores.append(s_d)
        wass_scores.append(w_d)
        jsh_scores.append(jsh_d)
    # take top N elements
    top_n = 3
    idxs = np.argpartition(scores, -top_n)[-top_n:]
    idxs2 = np.argpartition(wass_scores, top_n)[:top_n]
    idxs3 = np.argpartition(jsh_scores, top_n)[:top_n]
    title_names_pool = np.array(title_names_pool)
    scores = np.array(scores)
    wass_scores = np.array(wass_scores)
    jsh_scores = np.array(jsh_scores)
    res = [x for x in zip(title_names_pool[idxs], scores[idxs], title_names_pool[idxs2], wass_scores[idxs2],
                          title_names_pool[idxs3], jsh_scores[idxs3]
                          )]
    print(res)
    return res


vectors = np.loadtxt(open("alter_bert_vectors.csv", "rb"), delimiter=",")


# brute_paral(look4titles, vectors, df_alter['Alternate Title'])
#best_match_v3('IT Data & Workstation Analyst', vectors, df_alter['Alternate Title'])
# test('all_test_samples.csv', occup_pool_df['title']) Time elapsed: 115275.7 ms

@timit
def run_preds(look4titles, cand_pool):
    #ray.init(num_cpus=6)
    results = []
    step = 3
    for i in range(0, len(look4titles), step):
        result_ids = []
        ray.init(num_cpus=6, ignore_reinit_error=True)
        for j in range(i, i+step if (i+step)< len(look4titles) else len(look4titles)):
            # Start 4 tasks in parallel.
            result_ids.append(best_match_v3.remote(look4titles[j], vectors, cand_pool))
            # Wait for the task1s to complete and retrieve the results.
            # With at least 4 cores, this will take 1 second.
        results += list(ray.get(result_ids))
        ray.shutdown()
    print(results)
    res = pd.DataFrame([list(look4titles), results], index=['input_titles', 'predicted_title']).T
    res.to_csv('data/predictions.csv', sep=',')



cand_pool = df_alter['Alternate Title']
look4titles = ['IT Data & Workstation Analyst', 'Senior Software Engineer', 'Facilities Design Manager']

all_test_samples = pd.read_excel('data/missmatches.xlsx', names=['title', 'pred'])['title']
run_preds(list(all_test_samples), cand_pool)
