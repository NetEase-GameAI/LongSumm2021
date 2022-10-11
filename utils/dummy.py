import os
import json
import re
import requests
from tqdm import tqdm


def gen_pdf_info():
    urls_dict = {}

    path = '../abstractive_summaries/by_clusters'

    pdf_list = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            # print os.path.join(subdir, file)
            filepath = subdir + os.sep + file

            if filepath.endswith(".json"):
                with open(filepath, 'r') as in_f:
                    summ_info = json.load(in_f)
                    pdf_list.append(summ_info)
                    urls_dict[summ_info['id']] = summ_info['pdf_url']
            else:
                print(filepath)
                continue
    with open("../dataset/pdf_info.json", 'w') as f:
        json.dump(pdf_list, f)


def gen_broken_pdf_info():
    path = '../broken_pdf'
    pdf_id = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            pdf_id.append(file.replace(".pdf", ''))
    with open('../dataset/result.txt', 'r') as f:
        line = f.readline()
        while line:
            p = re.compile('[0-9]+')
            num = p.findall(line)
            pdf_id.append(num[0])
            line = f.readline()
    print("broken pdf num: ", len(pdf_id))
    with open('../dataset/pdf_info.json', 'r') as f:
        ds = json.load(f)
    res = []
    for d in ds:
        if str(d['id']) in pdf_id:
            res.append(d)
    print("check broken num: ", len(res))
    with open('../dataset/broken.json', 'w') as f:
        json.dump(res, f)


def download_broken():
    with open('../dataset/broken.json', 'r') as f:
        ds = json.load(f)
        for d in tqdm(ds):
            try:
                r = requests.get(d['pdf_url'])
                with open('../dataset/%d.pdf' % (d['id']), 'wb') as f:
                    f.write(r.content)
            except:
                print("error id : ", d['id'])


def print_broken_pdf():
    with open("../dataset/broken.json", 'r') as f:
        ds = json.load(f)
        for d in ds:
            print(d['id'])
            print(d['pdf_url'])


def gen_abstract_data():
    with open('../dataset/pdf_info.json', 'r') as f:
        tot_pdf = json.load(f)
    pdf_id = []
    for subdir, dirs, files in os.walk('../abstractive_papers'):
        for file in files:
            pdf_id.append(file.replace(".pdf", ''))
    if '.DS_Store' in pdf_id:
        pdf_id.remove('.DS_Store')
    print(len(pdf_id))
    c_pdf, b_pdf = [], []
    for pdf in tot_pdf:
        if str(pdf['id']) in pdf_id:
            c_pdf.append(pdf)
        else:
            b_pdf.append(pdf)
    print("num: good pdf %d, broken pdf %d" % (len(c_pdf), len(b_pdf)))
    with open("../dataset/abstract_info.json", 'w') as f:
        json.dump(c_pdf, f)
    with open("../dataset/broken_info.json", 'w') as f:
        json.dump(b_pdf, f)
    print("finish")


# gen_broken_pdf_info()
# download_broken()
# print_broken_pdf()
gen_abstract_data()
