import json
import os
# from xml.dom.minidom import parse
from xml.etree.ElementTree import parse
from lxml import etree
from xml.etree.ElementTree import iterparse
import pandas as pd
import re
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as tft
from tqdm import tqdm
import numpy as np

import tensorflow as tf


def gen_data_dict():
    def gen_cont(path='./train_json'):
        _, _, files = next(os.walk(path))
        json_list = []
        for file in files:
            with open(path + '/' + file, 'r', encoding='utf-8') as fp:
                raw_d = json.load(fp)
                json_list.append(raw_d)
        return json_list

    def gen_info(path='./abstractive_summaries/by_clusters'):
        info_list = []
        for subdir, dirs, files in os.walk(path):
            for file in files:
                filepath = subdir + os.sep + file

                if filepath.endswith(".json"):
                    with open(filepath, 'r') as in_f:
                        summ_info = json.load(in_f)
                        info_list.append(summ_info)

        return info_list

    p1, p2, dt = gen_cont(), gen_info(), []
    for pp1 in p1:
        tmp = {}
        tmp['body'] = ""
        for pp2 in p2:
            name = str(pp2['id']) + '.pdf'
            if pp1['name'] == name:
                tmp['id'] = pp2['id']
                tmp['summary'] = pp2['summary']
                tmp['target'] = " ".join(pp2['summary'])
                tmp['sections'] = pp1['metadata']['sections']
                if tmp['sections']:
                    text = ""
                    for s in tmp['sections']:
                        text += s['text']
                    tmp['body'] = text.replace('\n', ' ')
        if not tmp['body'] or not tmp['target']:
            continue
        dt.append(tmp)

    with open('./dataset/raw.json', 'w') as f:
        json.dump(dt, f)
    return dt


def gen_tf_training_data():
    def _bytes_feature(value):
        """Returns a bytes_list from a string/byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        if isinstance(value, str):
            value = value.encode('unicode-escape')
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def get_tfrecords_example(inputs, targets):
        tfrecords_features = {}
        tfrecords_features['inputs'] = _bytes_feature(inputs)
        tfrecords_features['targets'] = _bytes_feature(targets)

        return tf.train.Example(features=tf.train.Features(feature=tfrecords_features))

    dt = gen_data_dict()

    tfrecord_wrt = tf.python_io.TFRecordWriter('./dataset/train.tfrecord')
    cnt = 0
    for d in dt:
        if 'body' in d.keys():
            example = get_tfrecords_example(d['body'], d['target'])
            tfrecord_wrt.write(example.SerializeToString())
            cnt += 1
    tfrecord_wrt.close()
    print("total training examples: ", cnt)


def gen_training_data():
    dt = gen_data_dict()
    inp, tar, sec, summ = [], [], [], []
    for d in dt:
        if 'body' in d.keys():
            inp.append(d['body']), tar.append(d['target']), sec.append(d['sections']), summ.append(d['summary'])
    return inp, tar, sec, summ


def xml2json(xml_path='../dataset/xml'):
    def ns_tag(*args):
        ns = "/{http://www.tei-c.org/ns/1.0}"

        return './%s%s' % (ns, ns.join(args))
    ds = []
    for d, _, files in os.walk(xml_path):
        item = {}
        for file in files:
            xml_file = xml_path + os.sep + file

            dom = etree.parse(xml_file)
            root = dom.getroot()

            try:
                abstract = str(root.find(ns_tag('abstract', 'p')).xpath('text()')[0])
            except:
                print("no abstract found in ", xml_file)
                exit(1)
            item['abstract'] = abstract

            body = root.findall(ns_tag('body', 'div'))
            section_name, section_content = [], []

            for div in body:
                head = div.find(ns_tag('head'))
                ps = div.findall(ns_tag('p'))
                if not ps:
                    print('no content found in ')
                    continue
                try:
                    section_name.append(etree.tostring(head, encoding='utf-8', method='text').decode('utf-8').lower())
                except:
                    print("no head found in ", xml_file)
                    exit(1)
                section_content.append(" ".join([etree.tostring(p, encoding='utf-8', method='text').
                                                decode('utf-8') for p in ps]))
            item['section_name'] = section_name
            item['section_content'] = section_content
            ds.append(item)
    with open("../dataset/xml/gtest.json", 'w') as f:
        json.dump(ds, f)


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, str):
        value = value.encode('utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_record(features: dict, feat_type: dict, file_path):
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode('utf-8', 'ignore')]))

    feat_name = list(features.keys())
    n_example = len(features[feat_name[0]])
    writer = tf.io.TFRecordWriter(file_path)
    for it in range(n_example):
        feature = {}
        for name in feat_name:
            if feat_type[name] == 'string':
                tf_feat = _bytes_feature(features[name][it])
            feature[name] = tf_feat
        tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(tf_example.SerializeToString())

    writer.close()


def _decode_record(record):
    """Decodes a record to a TensorFlow example."""
    name_to_features = {
        "document": tf.io.FixedLenFeature([], tf.string),
        "summary": tf.io.FixedLenFeature([], tf.string),

    }
    example = tf.io.parse_single_example(record, name_to_features)
    return example["document"], example["summary"]


def read_tf_record(file_path):  # "need '/' at the end of the path eg: /home/dataset/ "
    f = tf.io.gfile.walk(file_path)
    f = next(f)
    print("find files: ", f)
    files = [file_path + str(it) for it in f[2]]

    ds = tf.data.TFRecordDataset(files)
    print("total num: ", len(list(ds.as_numpy_iterator())))
    ds = ds.map(_decode_record)

    # try this to print an example
    for i in ds.take(10):
        print("Feature:\n{}\n\nLabel:\n{}\n\n".format(i[0].numpy().decode('utf-8'), i[1].numpy().decode('utf-8')))

    return ds


file_path = '../dataset/gen_data/train.tfrecord-'


def gen_acl_ss_data():

    with open("../dataset/json_data/acl515.json", 'r') as f:
        d = json.load(f)
    tot_data = []
    cnt = 0
    for it in d:
        if "shortscience" in it['source_website']:
            continue
        doc = it['abstract'] + " ".join(it['section_content'])
        summ = " ".join(it['summary'])

        if len(summ.split()) <= 100:
            cnt += 1
            continue

        tot_data.append({"document": doc,
                         "summary": summ})
    print("acl drop num: ", cnt)

    union_data = []
    with open('../dataset/json_data/union_add.json', 'r') as f:
        line = f.readline()
        while line:
            union_data.append(json.loads(line))
            line = f.readline()
    cnt = 0

    def drop_url(s):
        return re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%|\-|\#)*\b', '[url]', s, flags=re.MULTILINE)
    for it in union_data:
        doc = it['page']['text']
        summ = drop_url(it['summary'])
        if len(doc.split()) < 1000 or len(summ.split()) < 50:
            cnt += 1
            continue
        tot_data.append({"document": doc,
                         "summary": summ})

    print("short science drop num: ", cnt)
    print("=="*32)
    print("total data: ", len(tot_data))
    with open("../dataset/json_data/acl_ss.json", 'w') as f:
        json.dump(tot_data, f)
    print("finish")


# xml2json()