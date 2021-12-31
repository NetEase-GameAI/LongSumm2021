# @Author: yingsenci
# @Time: 2021/03/30
# @Contact: scying@zju.edu.com,
# @Description: divide scientific documents into session piece dataset for training


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
import json
import torch
import torch.nn as nn
from nltk.corpus import stopwords
from tqdm import tqdm
import numpy as np
from utils.build_data import write_record
import random

stop_words = stopwords.words('english')

print("length of stopwords: ", len(stop_words))
print(stop_words[:20])

window_size = 800
buffer = 200
decode_max_len = 220
split = 30
print("window_size: ", window_size)
print("buffer: ", buffer)
print("decode_max_len: ", decode_max_len)


def load_model(path="/home/gitlib/pretrain_model/pytorch_pegasus"):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path)
    return tokenizer, model


def window_score_pegasus(single_window_data: str, gt_emb, tokenizer, model):
    sents = nltk.sent_tokenize(single_window_data)
    batch_sent = tokenizer(sents, return_tensors="pt", padding=True, truncation=True)
    output = model(batch_sent['input_ids'], batch_sent['attention_mask'], decoder_input_ids=batch_sent['input_ids'])
    encode_sent = output.encoder_last_hidden_state
    sent_emb = torch.sum(encode_sent, dim=1)

    cos = nn.CosineSimilarity()

    scores = [torch.mean(cos(sent_emb, g.reshape(1, -1))) for g in gt_emb]
    scores = torch.hstack(scores)
    return scores


def slide_window(raw_data: str, mode='window'):
    """
    split raw data into piece of session

    Args:
        raw_data: str, raw text of a paper
        mode: str, not used
    Returns:
    split_data: list, session data

    """
    spilt_data = []
    words = nltk.word_tokenize(raw_data)
    if len(words) - window_size - buffer < 0:
        spilt_data.append(raw_data)
    else:
        for i in range(0, len(words), window_size):
            spilt_data.append(" ".join(words[max(0, i - buffer):min(i + window_size, len(words))]))

    return spilt_data


def window_score(single_window_data: str, gt_rm_sw: list, metric='recall'):
    """compute all scores of a session with all ground truth """
    if metric == 'recall':
        win_rm_sw = word_token(single_window_data, "article")
        score = [len(set(win_rm_sw) & set(it)) / len(set(it)) for it in gt_rm_sw]

    if metric == 'precision':
        win_rm_sw = word_token(single_window_data, "article")
        score = [len(set(win_rm_sw) & set(it)) / len(set(win_rm_sw)) for it in gt_rm_sw]

    return score


def word_token(raw_str, str_type):
    """token replacement"""
    if str_type == "article":
        sents = nltk.sent_tokenize(raw_str)
        words = []
        for sent in sents:
            words.extend(nltk.word_tokenize(sent))
    if str_type == "sent":
        words = nltk.word_tokenize(raw_str)
    if str_type == "no_stop":
        return nltk.word_tokenize(raw_str)

    return [w for w in words if w.lower() not in stop_words]


def save_record(d, s, f):
    """save tf record type data"""
    feat = {"document": d, "summary": s}
    feat_type = {'document': 'string',
                 'summary': 'string'}
    write_record(feat, feat_type, f)


def session_rank(document, summary, out_file, mode='more'):
    """
    split document into session piece and match each session with its ground truth

    Args:
        document: list, document list
        summary: list, ground truth list
        out_file: str, output file path
        mode: str, match method 'more' means matching as much labels as possible to each session

    Returns:
    tf record type dataset
    """
    split_doc = list(map(slide_window, document))
    split_summary = [nltk.sent_tokenize(it) for it in summary]

    features, labels = [], []
    for i in tqdm(range(len(split_doc))):
        gt_rm_sw = [word_token(s, "sent") for s in split_summary[i]]
        if mode == 'less':
            scores = np.array([window_score(it, gt_rm_sw, metric="recall") for it in split_doc[i]])

            ind = np.argmax(scores, axis=0)
            summ = [[] for it in split_doc[i]]
            for j in range(len(split_summary[i])):
                summ[ind[j]].append(split_summary[i][j])

            for j in range(len(split_doc[i])):
                if not summ[j]:
                    continue
                features.append(split_doc[i][j])
                labels.append(" ".join(summ[j]))

        if mode == 'more':
            np_s = np.array(split_summary[i])
            scores = np.array([window_score(it, gt_rm_sw, metric="recall") for it in split_doc[i]])
            d1, d2 = scores.shape
            for j in range(d1):
                s_len = 0
                for _ in range(d2):
                    if s_len > decode_max_len:
                        break
                    r = np.argmax(scores[j])
                    s_len += len(word_token(split_summary[i][r], 'no_stop'))
                    scores[j][r] = -1
            mask = (scores < 0)

            for j in range(d1):
                summ = " ".join(np_s[mask[j]])
                features.append(split_doc[i][j])
                labels.append(summ)

    save_record(features, labels, out_file)
    print("num examples: ", len(features))

    print("write into %s" % out_file)


def test_unit():
    a = "this is my dog. do you like cs? machine learning is good, i think"
    b = ["my cat is good", "bad machine is you"]
    gt_rm_sw = [word_token(s, "sent") for s in b]
    print(gt_rm_sw)
    score = window_score(a, gt_rm_sw)
    print(score)


def input_fn(file_path):
    d = []
    with open(file_path, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            d.append(json.loads(line))
            line = f.readline()

    random.shuffle(d)
    document = []
    summary = []
    for it in d:
        document.append(it['artical'])
        summary.append(it['summary'])
    return document, summary


def main():
    in_file = '../dataset/json_data/acl_ss.json'
    document, summary = input_fn(in_file)
    out_file = '../dataset/acl_ss_small/train/train.tfrecord'
    val_file = '../dataset/acl_ss_small/eval/eval.tfrecord'
    pred_json = '../dataset/acl_ss_small/pred/pred.json'
    pred_tfd = '../dataset/acl_ss_small/pred/pred.tfrecord'
    with open ('../dataset/acl_ss_small/param.json', 'w') as f:
        json.dump({'window_size': window_size, 'decode_max_len': decode_max_len, 'buffer': buffer}, f)
    train_x, train_y = document[split:], summary[split:]
    val_x, val_y = document[:split], summary[:split]

    session_rank(train_x, train_y, out_file)
    session_rank(val_x, val_y, val_file)
    pred = []
    for i in range(split):
        pred.append({'document':val_x[i], "summary": val_y[i]})
    with open(pred_json, 'w') as f:
        json.dump(pred, f)
    save_record(val_x, val_y, pred_tfd)


