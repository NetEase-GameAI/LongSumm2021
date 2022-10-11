import json
import nltk
from model.text_rank import rank_scores, gen_embedding
import numpy as np
import random
from tqdm import tqdm

with open("../dataset/json_data/sections_test.json",'r') as f:
    d = json.load(f)

for it in d:
    docs = it['document']
    it['document'] = " ".join([docs[0], docs[1], docs[-1]])

with open("../dataset/json_data/test_header.json", 'w') as f:
    json.dump(d, f)
