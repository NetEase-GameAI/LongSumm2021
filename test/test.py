import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as tft
import torch
import json
import re
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"


def test_dataset():
    max_encoder_length = 3072
    max_decoder_length, substitute_newline, batch_size = 256, "<n>", 2
    data_dir = "/home/pycharm_work_space/LongSumm/dataset/arxiv/"
    data_dir = "/home/tensorflow_datasets/scientific_papers/arxiv/1.1.1/"
    def _decode_record(record):
        """Decodes a record to a TensorFlow example."""
        name_to_features = {
            "document": tf.io.FixedLenFeature([], tf.string),
            "summary": tf.io.FixedLenFeature([], tf.string),
        }
        name_to_features = {
            "article": tf.io.FixedLenFeature([], tf.string),
            "abstract": tf.io.FixedLenFeature([], tf.string),
            "section_names": tf.io.FixedLenFeature([], tf.string),
        }
        example = tf.io.parse_single_example(record, name_to_features)
        return example["article"], example["abstract"]

    def _tokenize_example(document, summary):
        tokenizer = tft.SentencepieceTokenizer(
            model=tf.io.gfile.GFile("/home/pycharm_work_space/LongSumm/bigbird/vocab/pegasus.model", "rb").read())
        if substitute_newline:
            document = tf.strings.regex_replace(document, "\n", substitute_newline)
        # Remove space before special tokens.
        document = tf.strings.regex_replace(document, r" ([<\[]\S+[>\]])", b"\\1")
        document_ids = tokenizer.tokenize(document)
        if isinstance(document_ids, tf.RaggedTensor):
            document_ids = document_ids.to_tensor(0)
        document_ids = document_ids[:max_encoder_length]

        # Remove newline optionally
        if substitute_newline:
            summary = tf.strings.regex_replace(summary, "\n", substitute_newline)
        # Remove space before special tokens.
        summary = tf.strings.regex_replace(summary, r" ([<\[]\S+[>\]])", b"\\1")
        summary_ids = tokenizer.tokenize(summary)
        # Add [EOS] (1) special tokens.
        suffix = tf.constant([1])
        summary_ids = tf.concat([summary_ids, suffix], axis=0)
        if isinstance(summary_ids, tf.RaggedTensor):
            summary_ids = summary_ids.to_tensor(0)
        summary_ids = summary_ids[:max_decoder_length]

        return document_ids, summary_ids

    d1 = tfds.load('scientific_papers/arxiv', split="train", data_dir='/home/tensorflow_datasets',
                      shuffle_files=False, as_supervised=True)

    file_dir = tf.io.gfile.walk(data_dir)
    file_dir = next(file_dir)

    files = [ data_dir + "train.tfrecord-%d" % i for i in range(1)]
    files = [data_dir+ "scientific_papers-test.tfrecord-00001-of-00002"]
    print("filenames: ", files)

    d2 = tf.data.TFRecordDataset(files)

    d2 = d2.map(_decode_record,
              num_parallel_calls=tf.data.experimental.AUTOTUNE)

    d1 = d1.map(_tokenize_example,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

    for i in d2.take(1):
        print(i[0])
        print(i[1])
    d2 = d2.map(_tokenize_example,
              num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # d = d.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
    # d = d.repeat()
    d1 = d1.padded_batch(batch_size, ([max_encoder_length], [max_decoder_length]),
                         drop_remainder=True)
    d2 = d2.padded_batch(batch_size, ([max_encoder_length], [max_decoder_length]),
                       drop_remainder=True)
    t1, t2 = [],[]
    for i in d2.take(1):
        print(i)


def gen_test_data():
    d = []
    res = []
    with open("../dataset/json_data/raw_test.json", 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            d.append(json.loads(line))
            line = f.readline()
    for it in d:
        res.append({"document": it['text'], "id": it['id'], "summary": " "})
    with open("../dataset/json_data/test.json", 'w') as f:
        json.dump(res, f)


def test_re():
    d = "human ab We use the SimpleQuestions (Bordes et al., 2015) and WebQSP (Yih et al., 2016) datasets."
    p = re.compile('\(.*?et al.*?\)')
    print(re.sub(p, '', d))


def test_test():
    d = []
    with open("../output/test/aic.json", 'r', encoding='utf-8') as f:
        l = f.readline()
        while l:
            d.append(json.loads(l))
            l = f.readline()
    res = {}
    for i in d:
        res[i['id']] = i['text']
    with open("../output/test/raic.json", 'w') as f:
        json.dump(res, f)


test_test()