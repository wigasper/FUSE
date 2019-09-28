#!/usr/bin/env python3

import sys
import json
import logging
import traceback

from tqdm import tqdm
import numpy as np 

global logger

def evaluate(preds, solution, uids):
    true_pos = 0
    false_pos = 0
    false_neg = 0

    for pmid in preds:
        true_pos += len([pred for pred in preds[pmid] if pred in solution[pmid]])
        false_pos += len([pred for pred in preds[pmid] if pred not in solution[pmid]])
        false_neg += len([sol for sol in solution[pmid] if sol not in preds[pmid]])

    if true_pos == 0:
        mi_precision = 0
        mi_recall = 0
        mi_f1 = 0
    else:
        mi_precision = true_pos / (true_pos + false_pos)
        mi_recall = true_pos / (true_pos + false_neg)
        mi_f1 = (2 * mi_precision * mi_recall) / (mi_precision + mi_recall)

    logger.info(f"Micro-averaged F1 from test set: {mi_f1}")
    logger.info(f"Micro-averaged precision from test set: {mi_precision}")
    logger.info(f"Micro-averaged recall from test set: {mi_recall}\n")

    eb_ps = []
    eb_rs = []
    eb_f1s = []

    for pmid in preds:
        true_pos = len([pred for pred in preds[pmid] if pred in solution[pmid]])
        false_pos = len([pred for pred in preds[pmid] if pred not in solution[pmid]])
        false_neg = len([sol for sol in solution[pmid] if sol not in preds[pmid]])

        if true_pos == 0:
            eb_precision = 0
            eb_recall = 0
            eb_f1 = 0
        else:
            eb_precision = true_pos / (true_pos + false_pos)
            eb_recall = true_pos / (true_pos + false_neg)
            eb_f1 = (2 * eb_precision * eb_recall) / (eb_precision + eb_recall)

        eb_ps.append(eb_precision)
        eb_rs.append(eb_recall)
        eb_f1s.append(eb_f1)

    eb_f1 = sum(eb_f1s) / len(eb_f1s)
    eb_recall = sum(eb_rs) / len(eb_rs)
    eb_precision = sum(eb_ps) / len(eb_ps)

    logger.info(f"Example-based F1 from test set: {eb_f1}")
    logger.info(f"Example-based precision from test set: {eb_precision}")
    logger.info(f"Example-based recall from test set: {eb_recall}\n")

    ma_ps = []
    ma_rs = []
    ma_f1s = []

    for uid in uids:
        true_pos = 0
        false_pos = 0
        false_neg = 0

        for pmid in preds:
            if uid in preds[pmid] and uid in solution[pmid]:
                true_pos += 1
            if uid in preds[pmid] and uid not in solution[pmid]:
                false_pos += 1
            if uid in solution[pmid] and uid not in preds[pmid]:
                false_neg += 1

        if true_pos == 0:
            ma_precision = 0
            ma_recall = 0
            ma_f1 = 0
        else:
            ma_precision = true_pos / (true_pos + false_pos)
            ma_recall = true_pos / (true_pos + false_neg)
            ma_f1 = (2 * ma_precision * ma_recall) / (ma_precision + ma_recall)

        if true_pos + false_pos + false_neg > 0:
            ma_ps.append(ma_precision)
            ma_rs.append(ma_recall)
            ma_f1s.append(ma_f1)

    ma_f1 = sum(ma_f1s) / len(ma_f1s)
    ma_recall = sum(ma_rs) / len(ma_rs)
    ma_precision = sum(ma_ps) / len(ma_ps)

    logger.info(f"Macro-averaged F1 from test set: {ma_f1}")
    logger.info(f"Macro-averaged precision from test set: {ma_precision}")
    logger.info(f"Macro-averaged recall from test set: {ma_recall}\n")

def get_f1(thresh, term_freqs, solution):
    # Predict
    predictions = {}
    for doc in term_freqs.keys():
        predictions[doc] = [key for key, val in term_freqs[doc].items() if val > thresh]

    # Get evaluation metrics
    true_pos = 0
    false_pos = 0
    false_neg = 0

    for pmid in predictions:
        true_pos += len([pred for pred in predictions[pmid] if pred in solution[pmid]])
        false_pos += len([pred for pred in predictions[pmid] if pred not in solution[pmid]])
        false_neg += len([sol for sol in solution[pmid] if sol not in predictions[pmid]])

    if true_pos == 0:
        precision = 0
        recall = 0
        f1 = 0
    else:
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        f1 = (2 * precision * recall) / (precision + recall)

    return f1

def train(term_freqs, solution):
    curr_thresh = 0.0
    step_val = 0.01
    f1s = []

    f1s.append(get_f1(curr_thresh, term_freqs, solution))
    f1s.append(get_f1(curr_thresh + step_val, term_freqs, solution))

    curr_thresh += step_val
    next_thresh_f1 = get_f1(curr_thresh + step_val, term_freqs, solution)

    while not (next_thresh_f1 < f1s[-1] and next_thresh_f1 < f1s[-2] and f1s[-1] < f1s[-2]):
        curr_thresh += step_val
        f1s.append(get_f1(curr_thresh, term_freqs, solution))
        logger.info(f"curr thresh: {curr_thresh}, f1: {f1s[-1]}")
        next_thresh_f1 = get_f1(curr_thresh + step_val, term_freqs, solution)

    return curr_thresh - step_val

def predict(test_freqs, thresh):
    # Test it out
    predictions = {}

    # Predict
    for doc in test_freqs.keys():
        predictions[doc] = [key for key, val in test_freqs[doc].items() if val > thresh]

    return predictions

def array_builder(filepath, term_idxs):
    array_out = np.zeros((len(term_idxs), len(term_idxs)))
    with open(filepath, "r") as handle:
        for line in handle:
            line = line.strip("\n").split(",")
            if line[0] in term_idxs and line[1] in term_idxs:
                array_out[term_idxs[line[0]], term_idxs[line[1]]] = float(line[2])
                array_out[term_idxs[line[1]], term_idxs[line[0]]] = float(line[2])
    return array_out

if __name__ == "__main__":   
    # Set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler("../logs/simple_informed_model.log")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # load in UIDs
    uids = []
    with open("../data/mesh_data.tab", "r") as handle:
        for line in handle:
            line = line.strip("\n").split("\t")
            if line[0] in subset:
                uids.append(line[0])

    # Load in term frequencies
    with open("../data/term_freqs_rev_3_all_terms.json", "r") as handle:
        temp = json.load(handle)
    
    train_docs = docs_list[0:partition]
    test_docs = docs_list[partition:]

    docs_list = list(temp.keys())
    
    term_freqs = temp

    # Load in solution values
    solution = {}
    docs_list = set(docs_list)
    with open("../data/pm_doc_term_counts.csv", "r") as handle:
        for line in handle:
            line = line.strip("\n").split(",")
            if line[0] in docs_list:
                # Only use samples indexed with MeSH terms
                terms = [term for term in line[1:] if term]
                if terms:
                    solution[line[0]] = terms    
    
    # normalize
    for pmid in term_freqs:
        freqs = term_freqs[pmid]
        mean_freq = sum(freqs.values()) / len(freqs.values())
        min_freq = min(freqs.values())
        max_freq = max(freqs.values())
        if max_freq - min_freq > 0:
            for freq in freqs:
                freqs[freq] = (freqs[freq] - mean_freq) / (max_freq - min_freq)
        term_freqs[pmid] = freqs
    
    # Load term subset to count for
    term_subset = []
    with open("../data/subset_terms_list", "r") as handle:
        for line in handle:
            term_subset.append(line.strip("\n"))
    
    # Dict for array assembly and lookup
    term_idxs = {term_subset[idx]: idx for idx in range(len(term_subset))}
    term_idxs_reverse = {idx: term_subset[idx] for idx in range(len(term_subset))}
    
    #sem_sims = array_builder("../data/semantic_similarities_rev1.csv", term_idxs)
                
    coocc_log_ratios = array_builder("../data/term_co-occ_log_likelihoods.csv", term_idxs)
    max_ratio = np.max(coocc_log_ratios)
    
    # testing tuning value to improve model
    tuning_val = 1
    
    logger.info("Beginning co-occurrence incorporation")
    for doc in tqdm(term_freqs.keys()):
        try:
            # add semantically similar terms to each pool and weight by similarity
            coocc_terms = {}
            for term in term_freqs[doc].keys():
                if term in term_idxs.keys():
                    row = term_idxs[term]
                    # coocc_log_ratios must have same dims here, may need to do something about this
                    for col in range(coocc_log_ratios.shape[0]):
                        if coocc_log_ratios[row, col] > 3: #or coocc_log_ratios[row, col] < -3:
                            coocc_terms[term_idxs_reverse[col]] = ((coocc_log_ratios[row, col] * tuning_val) / max_ratio) * term_freqs[doc][term]
    
            for term in coocc_terms.keys():
                if term in term_freqs[doc].keys():
                    term_freqs[doc][term] += coocc_terms[term]
#                else:
#                    term_freqs[doc][term] = coocc_terms[term]
        except Exception as e:
            trace = traceback.format_exc()
            logger.error(repr(e))
            logger.critical(trace)
    logger.info("Co-occurrence incorporation complete")
    
    del(coocc_log_ratios)

    # Train and test
    train_freqs = {}
    for doc in train_docs:
        if doc in solution.keys():
            train_freqs[doc] = term_freqs[doc]

    test_freqs = {}
    for doc in test_docs:
        if doc in solution.keys():
            test_freqs[doc] = term_freqs[doc]  

    thresh = train(train_freqs, solution)
    
    preds = predict(test_freqs, thresh)

    evaluate(preds, solution, uids)
