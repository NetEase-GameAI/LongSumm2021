# @Author: yingsenci
# @Time: 2021/03/30
# @Contact: scying@zju.edu.com,
# @Description: ensemble method for summarization


import re
import nltk
from summa import summarizer
from summa import keywords
import json
import numpy as np
from nltk.corpus import stopwords, wordnet
from model.text_rank import rank_scores, gen_embedding
from tqdm import tqdm

stop_words = stopwords.words('english')


def drop_sent(raw_s: str):
    """drop redundancy"""
    sentences = nltk.sent_tokenize(raw_s)
    d_sentences = []
    [d_sentences.append(it) for it in sentences if it not in d_sentences]

    return " ".join(d_sentences)


def add_summ(text):
    """auto summarize"""
    res = summarizer.summarize(text, ratio=0.8)
    return res


def sent_sim(s, t, mode='jaccard', use_stopwords=True):
    """sentences similarity """
    ws = nltk.word_tokenize(s)
    wt = nltk.word_tokenize(t)
    if use_stopwords:
        ws = [i for i in ws if i.lower() not in stop_words]
        wt = [i for i in wt if i.lower() not in stop_words]

    if mode == 'jaccard':
        ws = set(ws)
        wt = set(wt)
        return len(ws & wt) / len(ws | wt)


def self_clip(raw_str: str, r=0.8):
    """clip same sentences"""
    sents = nltk.sent_tokenize(raw_str)
    sents = np.array(sents)
    l = len(sents)
    mask = np.ones(l, dtype=bool)
    for i in range(l):
        if not mask[i]:
            continue
        for j in range(i+1, l):
            if sent_sim(sents[i],sents[j], mode='jaccard') >= r:
                mask[j] = False

    selected_sents = sents[mask]
    clip_str = " ".join(selected_sents)
    return clip_str


def summary_merge(file1, file2, out_file, r=0.5, method='recall'):
    """ensemble the output of abstractive model and the extractive model"""

    with open(file1, 'r', encoding='utf-8') as f:
        d1 = json.load(f)

    with open(file2, 'r', encoding='utf-8') as f:
        d2 = json.load(f)

    embedding_path = "/home/gitlib/pretrain_model/glove/glove.6B.200d.txt"
    emb = gen_embedding(embedding_path)
    res = {}
    for k in tqdm(d1.keys()):
        if method == 'raw':
            m_s = self_clip(" ".join([d1[k], d2[k]]), r)
        if method == 'recall':
            s = self_clip(" ".join([d1[k], d2[k]]), r)
            sents = nltk.sent_tokenize(s)

            scores = rank_scores(sents, emb)
            sents = np.array(sents)
            mask = (scores > 1)
            m_s = " ".join(sents[mask])

        if method == 'text_rank':
            m_s = add_summ(" ".join([d1[k], d2[k]]))

        res[k] = m_s
    with open(out_file, 'w') as f:
        json.dump(res, f)


def join_words(s: str):
    with open("../dataset/words.txt", 'r', encoding='utf-8') as f:
        txt = f.read()
    words_list = txt.split('\n')
    sents = nltk.sent_tokenize(s)
    words = []

    def is_word(w):
        return w in words_list
    for sent in sents:
        words.extend(nltk.word_tokenize(sent))
    res = ""
    for i in range(len(words)):
        if i != len(words)-1 and not is_word(words[i]) and not is_word(words[i+1]) and is_word(words[i]+words[i+1]):
            print("merge words: {}, {}".format(words[i], words[i+1]))
            res += " " + words[i] + words[i+1]
            i += 1
        else:
            res += " " + words[i]
    return res