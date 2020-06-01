#-*- coding: utf-8 -*-

import sys
import pickle
import glob
import numpy as np
import pandas as pd
import itertools
import math
import statistics
import copy
import time
import datetime

from gensim.models.ldamulticore import LdaMulticore
from multiprocessing import Process, Queue, Manager
import logging
import logging.handlers


### IO
def LDA_IO(lda_model_name):
    with open('dictionary.pkl', 'rb') as f:
        dictionary = pickle.load(f)

    with open('doc2idx.pkl', 'rb') as f:
        doc2idx = pickle.load(f)

    lda_model = LdaMulticore.load(lda_model_name)
    token2id = dictionary.token2id  
    return doc2idx, lda_model, token2id

def load_token2id_old():
    with open('token2id_old.pkl', 'rb') as f:
        token2id_old = pickle.load(f)
    return token2id_old

def load_csv(file):
    data = pd.read_csv(file)
    return data


### preprocess
def get_term_topic_df(lda_model):
    n_topics = [i+1 for i in range(len(lda_model.get_topics()))]
    term_topic_arr = lda_model.get_topics().T
    term_topic_df = pd.DataFrame(data=term_topic_arr, columns=n_topics, dtype=float)
    return term_topic_df

def get_old2new(token2id_old, token2id):
    old_tokens = list(token2id_old.keys())
    for token in old_tokens:
        try:
            tmp = token2id[token]
        except:
            del token2id_old[token]
    old_ids = list(token2id_old.values())
    id2token_old = {y:x for x,y in token2id_old.items()}
    old2new = dict()
    for i in old_ids:
        old2new[i] = token2id[id2token_old[i]]
    return old_ids, old2new

def update_term_topic_df(term_topic_df, old_ids, old2new):
    term_topic_df = term_topic_df[term_topic_df.index.isin(old_ids)]

    old_idx = list(term_topic_df.index)
    new_idx = []
    for idx in old_idx:
        tmp = old2new[idx]
        new_idx.append(tmp)

    term_topic_df.index = new_idx
    return term_topic_df

def update_word_stats(word_stats, old_ids, old2new):
    word_stats = word_stats[word_stats.id.isin(old_ids)]

    old_idx = list(word_stats.id)
    new_idx = []
    for idx in old_idx:
        tmp = old2new[idx]
        new_idx.append(tmp)

    word_stats.id = new_idx
    return word_stats

def scale_word_stats(word_stats, theta, delta):
    word_stats_idx = word_stats.id
    # entropy
    word_entropy = word_stats.entropy_n
    word_entropy = np.interp(word_entropy, (word_entropy.min(), word_entropy.max()), (0, +1))
    word_entropy = pd.Series(word_entropy, index=word_stats_idx, name='entropy_n').sort_index()
    word_entropy = np.log(2+word_entropy)
    word_entropy = theta * word_entropy
    
    # DF
    docu_f = word_stats.df
    docu_f = np.interp(docu_f, (docu_f.min(), docu_f.max()), (0, +1))
    docu_f = pd.Series(docu_f, index=word_stats_idx, name='df').sort_index()
    docu_f = np.log(2+docu_f)
    docu_f = (1-theta) * docu_f

    # distribution stat
    dist_score = (word_entropy + docu_f) ** delta

    return dist_score


# not use now
def split_term_topic_df(term_topic_df, split_interval):
    split_interval = split_interval
    iter_length = math.ceil(len(term_topic_df) / split_interval)

    term_topic_df_li = []
    idxfrom = 0
    for i in range(1,iter_length+1):
        tmp = term_topic_df.iloc[idxfrom:split_interval*i]
        term_topic_df_li.append(tmp)
        idxfrom = split_interval*i
    return term_topic_df_li


def split_input_data(n_core, doc2idx):
    n_core = n_core
    split_interval = math.ceil(len(doc2idx)/n_core)
    doc2idx_li = []
    idxfrom = 0
    for i in range(1,n_core+1):
        doc2idx_li.append(doc2idx[idxfrom:split_interval*i])
        idxfrom = split_interval*i
    return doc2idx_li


def make_a_small_set_for_test(doc2idx_li, test_data_len):
    for i in range(len(doc2idx_li)):
        doc2idx_li[i] = doc2idx_li[i][:test_data_len]
    return doc2idx_li


def get_total_length(doc2idx_li):
    total_length = sum([len(doc2idx_li[i]) for i in range(len(doc2idx_li))])
    return total_length
    

### main process
# diverse keyword extraction for each document
# append w where max(h(w,S)) until n(S) reaches k
def diverse_keyword_extraction(document, term_topic_df, k, lambda_score, dist_score):
    topics = list(term_topic_df.columns)
    docu_set = list(set(document))
    rest_set = copy.deepcopy(docu_set)
    subset = []
    
    docu_term_topic_df = term_topic_df.loc[document]
    set_term_topic_df = term_topic_df.loc[docu_set]

    while (len(subset) < k) and (len(subset) < len(docu_set)):
        extracted_keyword = get_keyword_idx_tuned(document, subset, rest_set, docu_term_topic_df, set_term_topic_df, lambda_score, dist_score)
        subset.append(extracted_keyword)
        rest_set.remove(extracted_keyword)
    return subset

# not using now
def get_keyword_idx(document, subset, rest_set, docu_term_topic_df, set_term_topic_df, lambda_score):
    beta_score_df = docu_term_topic_df.sum(axis=0) / len(document)
    updated_reward_score_vector = (set_term_topic_df.loc[rest_set] + set_term_topic_df.loc[subset].sum(axis=0)) ** lambda_score
    keyword_score_df = beta_score_df * updated_reward_score_vector
    keyword_idx = keyword_score_df.sum(axis=1).idxmax()
    return keyword_idx

def get_keyword_idx_tuned(document, subset, rest_set, docu_term_topic_df, set_term_topic_df, lambda_score, dist_score):
    beta_score_df = docu_term_topic_df.sum(axis=0) / len(document)
    updated_reward_score_vector = (set_term_topic_df.loc[rest_set] + set_term_topic_df.loc[subset].sum(axis=0)) ** lambda_score
    keyword_score_df = beta_score_df * updated_reward_score_vector
    keyword_score = keyword_score_df.sum(axis=1)
    idx = keyword_score.index

    keyword_score = np.interp(keyword_score, (keyword_score.min(), keyword_score.max()), (0, +1))
    keyword_score = pd.Series(keyword_score, index=idx, name='keyword_score').sort_index()
    keyword_score = np.log(1+keyword_score)
    
    dist_score = dist_score[dist_score.index.isin(list(idx))].sort_index()

    final_score = keyword_score / dist_score
    final_score = pd.DataFrame(final_score, index=idx, columns=['final_score'])

    keyword_idx = int(final_score.idxmax())
    return keyword_idx


### multiprocessing
def run_model(qData, queue_li_part, qTerm, n_keyword, lambda_score, dist_score):
    data_part = qData.get()
    term_topic_df = qTerm.get()
    k = n_keyword
    lambda_score = lambda_score
    dist_score = dist_score

    for document in data_part:
        keyword_li = diverse_keyword_extraction(document, term_topic_df, k, lambda_score, dist_score)
        queue_li_part.put(keyword_li)
    time.sleep(60.5)


def counting(queue_li, total_length):
    start_d_time_total = datetime.datetime.now()
    start_time_total = time.time()
    print("counting started")

    count = 0
    start_time = time.time()
    while count <= total_length-1:
        if time.time() - start_time > 60:
            count = sum([queue.qsize() for queue in queue_li])

            print("%s percent done. [%s / %s]" % ((round(count/total_length*100, 2), count, total_length)))
            sys.stdout.flush()
            print([queue.qsize() for queue in queue_li])
            sys.stdout.flush()
            print("from start: %s (%.3f seconds)." % (str((datetime.datetime.now() - start_d_time_total))[:7], 
                                                    (time.time() - start_time_total)))
            print('')
            sys.stdout.flush()
            if count >= total_length:
                print("counting finished")
                sys.stdout.flush()
                break
            else:
                start_time = time.time()


def merge_sets(queue_li):
    rec_keyword_li = []
    for queue in queue_li:
        while not queue.empty():
            item = queue.get()
            rec_keyword_li.append(item)
        print(len(rec_keyword_li))
    return rec_keyword_li


if __name__ == '__main__':
    logger = logging.getLogger("06.diverse_keyword_extraction")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler("06.diverse_keyword_extraction.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info('========================================')
    logger.info('Processes started.')


    # parameters
    test = False
    if test:
        logger.info('*** testing ***')
    logger.info('*** processes started ***')
    start_d_time_total = datetime.datetime.now()
    start_time_total = time.time()

    lda_model_name = 'lda_30_abs'
    word_stats_name = 'word_analysis_0710.csv'
    n_keyword = 5
    lambda_score = 0.75
    theta = 0.75
    delta = 1

    n_core = 20
    test_data_len = 2000
    split_interval = 20000

    print('[parameters]\nmodel: %s\nn keyword: %s\nlambda score: %s\ntheta: %s\ndelta: %s\ncore used: %s\ntest data length: %s\nsplit interval: %s\n' % (lda_model_name, n_keyword, lambda_score, theta, delta, n_core, test_data_len, split_interval))


    # I/O
    logger.info('*** I/O started ***')
    start_d_time = datetime.datetime.now()
    start_time = time.time()

    doc2idx, lda_model, token2id = LDA_IO(lda_model_name)
    token2id_old = load_token2id_old()
    word_stats = load_csv(word_stats_name)

    logger.info('*** I/O finished: took %s (%.3f seconds). ***' % (str((datetime.datetime.now() - start_d_time))[:7], (time.time() - start_time)))


    # preprocess
    logger.info('*** preprocess started ***')
    start_d_time = datetime.datetime.now()
    start_time = time.time()

    term_topic_df = get_term_topic_df(lda_model)
    old_ids, old2new = get_old2new(token2id_old, token2id)
    term_topic_df = update_term_topic_df(term_topic_df, old_ids, old2new)
    word_stats = update_word_stats(word_stats, old_ids, old2new)
    dist_score = scale_word_stats(word_stats, theta, delta)
    doc2idx_li = split_input_data(n_core, doc2idx)
    if test:
        doc2idx_li = make_a_small_set_for_test(doc2idx_li, test_data_len)
    total_length = get_total_length(doc2idx_li)

    logger.info('*** preprocess finished: took %s (%.3f seconds). ***' % (str((datetime.datetime.now() - start_d_time))[:7], (time.time() - start_time)))    


    # clean memory
    del lda_model
    del token2id
    del doc2idx


    ### multiprocessing

    # stack
    logger.info('*** multiprocessing-stack started ***')
    start_d_time = datetime.datetime.now()
    start_time = time.time()

    queue_li = [Manager().Queue() for i in range(n_core)]
    data = doc2idx_li
    
    procs = []
    for idx in range(n_core):
        qData = Queue()
        qData.put(data[idx])
        
        qTerm = Queue()
        qTerm.put(term_topic_df)
    
        proc = Process(target=run_model, kwargs={'qData':qData, 'queue_li_part': queue_li[idx], 'qTerm': qTerm, 'n_keyword': n_keyword, 'lambda_score': lambda_score, 'dist_score': dist_score})
        procs.append(proc)
        proc.start()

    logger.info('total document count: %s' % (total_length))

    proc_count = Process(target=counting, kwargs={'queue_li':queue_li, 'total_length':total_length})
    proc_count.start()
    proc_count.join()

    logger.info('*** multiprocessing-stack finished: took %s (%.3f seconds). ***' % (str((datetime.datetime.now() - start_d_time))[:7], (time.time() - start_time)))


    # merge
    while sum([queue.qsize() for queue in queue_li]) != total_length:
        continue
    print([queue.qsize() for queue in queue_li])

    logger.info('*** merge started ***')
    start_d_time = datetime.datetime.now()
    start_time = time.time()

    rec_keyword_li = merge_sets(queue_li)
    print(len(rec_keyword_li))
    print([queue.qsize() for queue in queue_li])

    for proc in procs:
        proc.join()

    print('input length: %s' % (total_length))
    print('length of keyword list: %s' % len(rec_keyword_li))


    logger.info('*** merge finished: took %s (%.3f seconds). ***' % (str((datetime.datetime.now() - start_d_time))[:7], (time.time() - start_time)))
    
    # save result
    logger.info('*** save started ***')
    start_d_time = datetime.datetime.now()
    start_time = time.time()

    if test:
        with open('rec_keyword_li_test.pkl', 'wb') as f:
            pickle.dump(rec_keyword_li, f)
    else:
        with open('rec_keyword_li_%s_tuned.pkl' % (lambda_score), 'wb') as f:
            pickle.dump(rec_keyword_li, f)
    print('pickle saved: rec_keyword_li_%s_tuned' % (lambda_score))
    logger.info('*** save finished: took %s (%.3f seconds). ***' % (str((datetime.datetime.now() - start_d_time))[:7], (time.time() - start_time)))

    logger.info('*** all processes finished: took %s (%.3f seconds). ***' % (str((datetime.datetime.now() - start_d_time_total))[:7], (time.time() - start_time_total)))
