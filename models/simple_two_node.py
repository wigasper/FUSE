#!/usr/bin/env python3
import json
import logging
import argparse

from tqdm import tqdm

# Default threshold function
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

def learn_default_threshold(term_freqs, solution):
    curr_thresh = 0.0
    step_val = 0.001
    ##
    f1s = []
    
    f1s.append(get_f1(curr_thresh, term_freqs, solution))
    f1s.append(get_f1(curr_thresh + step_val, term_freqs, solution))
    
    curr_thresh += step_val
    next_thresh_f1 = get_f1(curr_thresh + step_val, term_freqs, solution)
    
    while not (next_thresh_f1 < f1s[-1] and next_thresh_f1 < f1s[-2] and f1s[-1] < f1s[-2]):
        curr_thresh += step_val
        f1s.append(get_f1(curr_thresh, term_freqs, solution))
        next_thresh_f1 = get_f1(curr_thresh + step_val, term_freqs, solution)
    
    print(curr_thresh - step_val)
    return curr_thresh - step_val

def predict(test_freqs, top_500_thresh, rest_thresh, top_500_set, solution, uids):

    thresholds = {}
    for uid in uids:
        if uid in top_500_set:
            thresholds[uid] = top_500_thresh
        else:
            thresholds[uid] = rest_thresh
    # Test it out
    predictions = {}
            
    # Predict
    for doc in test_freqs.keys():
        predictions[doc] = [key for key, val in test_freqs[doc].items() if val > thresholds[key]]
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

    return predictions, f1

def train(train_freqs, top_500_set, solution):
    top_500_freqs = {}
    for doc in train_freqs.keys():
        top_500_freqs[doc] = {term: freq for term, freq in train_freqs[doc].items() if term in top_500_set}

    top_500_sol = {}
    for doc in solution.keys():
        top_500_sol[doc] = {term for term in solution[doc] if term in top_500_set}
    
    top_500_thresh = learn_default_threshold(top_500_freqs, top_500_sol)
    rest_thresh = learn_default_threshold(train_freqs, solution)

    return top_500_thresh, rest_thresh

def main():
    # Set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler("../two_node.log")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Load in term frequencies and partition
    with open("../data/term_freqs_rev_3_all_terms.json", "r") as handle:
        temp = json.load(handle)

    docs_list = list(temp.keys())
    partition = int(len(docs_list) * .8)

    train_docs = docs_list[0:partition]
    test_docs = docs_list[partition:]

    top_500 = []
    with open("../data/top_500_terms", "r") as handle:
        for line in handle:
            top_500.append(line.strip("\n"))
    top_500_set = set(top_500)

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

    uids = []
    with open("../data/mesh_data.tab", "r") as handle:
        for line in handle:
            line = line.strip("\n").split("\t")
            uids.append(line[0])

    # Build training/test data, ensure good solution data is available
    # Solution data is not always available because documents may not be
    # indexed - even though obviously some of their references have been indexed
    train_freqs = {}
    for doc in train_docs:
        if doc in solution.keys():
            train_freqs[doc] = temp[doc]

    test_freqs = {}
    for doc in test_docs:
        if doc in solution.keys():
            test_freqs[doc] = temp[doc]  

    import time
    start = time.perf_counter()
    top_500_thresh, rest_thresh = train(train_freqs, top_500_set, solution)
    print(f"elapsed: {time.perf_counter() - start}")
    logger.info(f"default: {rest_thresh}, top_500: {top_500_thresh}")

    start = time.perf_counter()
    preds, f1 = predict(test_freqs, top_500_thresh, rest_threst, top_500_set, solution, uids)
    print(f"elapsed: {time.perf_counter() - start}")
    logger.info(f"f1: {f1}")
if __name__ == "__main__":
    main()
