import json
import re


def drop_str(summary_text):
    text = summary_text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = re.split(r"\s+", text)
    tokens = [x for x in tokens if re.match(r"^[a-z0-9]+$", x)]
    # tokens = tokens[0:min(max_tokens, len(tokens))]
    return " ".join(tokens)


def gen_train_data(path='train.json'):
    with open(path, 'r') as f:
        raw_data = json.load(f)
    train_v1 = []
    for d in raw_data:
        article = " ".join(d['section_content'])
        summary = " ".join(d['summary'])
        article_words = len(article.split())
        summary_words = len(summary.split())
        train_v1.append({'id': d['id'], 'article': article, 'summary': summary,
                         'article_words': article_words, 'summary_words': summary_words})
        print("id: %d, article words: %d, summary words: %d" % (d['id'], article_words, summary_words))
    with open('train_v1.json', 'w') as f:
        json.dump(train_v1, f)
    return


def gen_train_data_v2(path='train.json'):
    with open(path, 'r') as f:
        raw_data = json.load(f)
    train_v2 = []
    cnt = 0
    f = 0
    for d in raw_data:
        article = d['abstract']
        summary = " ".join(d['summary'])
        end = 0
        for i, sec_name in enumerate(d['section_name']):
            article += " " + d['section_content'][i]
            if "conclusion" in sec_name:
                break
        article_words = len(article.split())
        summary_words = len(summary.split())

        train_v2.append({'id': d['id'], 'article': article, 'summary': summary,
                         'article_words': article_words, 'summary_words': summary_words})
        print("id: %d, article words: %d, summary words: %d" % (d['id'], article_words, summary_words))

    with open('train_v2.json', 'w') as f:
        json.dump(train_v2, f)


def gen_train_data_v3(path='train.json'):
    with open(path, 'r') as f:
        raw_data = json.load(f)
    train_v3 = []
    cnt = 0
    f = 0
    for d in raw_data:
        abstract = d['abstract']
        summary = " ".join(d['summary'])
        introduction = d['section_content'][0]
        conclusion = d['section_content'][-1]
        for i, sec_name in enumerate(d['section_name']):
            if "conclusion" in sec_name:
                conclusion = d['section_content'][i]
        summary_words = len(summary.split())
        train_v3.append({'id': d['id'], 'article': [abstract, introduction, conclusion], 'summary': summary,
                         'summary_words': summary_words})
        # print("id: %d, article words: %d, summary words: %d" % (d['id'], article_words, summary_words))

    with open('train_v3.json', 'w') as f:
        json.dump(train_v3, f)


gen_train_data_v3()
