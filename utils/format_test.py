from utils.format import *
import json
from tqdm import tqdm


def trans_file():
    with open('../output/test/pred.json', 'r') as f:
        d = json.load(f)

    res = {}
    with open("../output/test/test.json", 'w') as f:
        for it in d:
            summ = drop_sent(it['pred'])
            res[it['id']] = summ
        f.write(json.dumps(res))


def summ_test(pred_in, pred_out):
    with open(pred_in, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    res = {}
    with open(pred_out, 'w') as f:
        for k in dataset.keys():
            doc = dataset[k]
            s = add_summ(doc)
            # print(s), print("--" * 64), print(add_keywords(doc)), print("**" * 64)
            res[k] = s
        f.write(json.dumps(res))


def test_merge():
    out_dir = "../output/test/"
    file = 'test.json'
    with open(out_dir+file, 'r', encoding='utf-8') as f:
        d = json.load(f)
    clip_d = {}
    for k in tqdm(d.keys()):
        clip_d[k] = self_clip(d[k], r=0.8)
    with open(out_dir+"clip_"+file, 'w') as f:
        json.dump(clip_d, f)


def merge_files():
    f1 = "../output/merge/1003_1021_replace_stay.json"
    f2 = "../output/merge/ex_best.json"
    out = "../output/merge/merge.json"

    summary_merge(f1, f2, out, r=0.3, method='recall')


# trans_file()
# test_merge()
# merge_files()

print(join_words("abs is a good : , fue, cla ss, i am a boy"))