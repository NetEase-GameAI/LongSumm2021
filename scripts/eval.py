import sys
import re
import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer


def impose_max_length(summary_text, max_tokens=600):
    text = summary_text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = re.split(r"\s+", text)
    tokens = [x for x in tokens if re.match(r"^[a-z0-9]+$", x)]
    tokens = tokens[0:min(max_tokens, len(tokens))]
    return " ".join(tokens)


def rouge_metric(pred: list, gt: list):
    metrics = ['rouge1', 'rouge2', 'rougeL']
    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
    results = {"rouge1_f": [], "rouge1_r": [], "rouge2_f": [], "rouge2_r": [], "rougeL_f": [], "rougeL_r": []}
    results_avg = {}
    batch = len(pred)
    for i in range(batch):
        clip_pred, clip_gt = impose_max_length(pred[i]), impose_max_length(gt[i])
        scores = scorer.score(clip_gt.strip(), clip_pred.strip())
        for metric in metrics:
            results[metric + "_f"].append(scores[metric].fmeasure)
            results[metric + "_r"].append(scores[metric].recall)

        for rouge_metric, rouge_scores in results.items():
            results_avg[rouge_metric] = np.average(rouge_scores)

    return results_avg
