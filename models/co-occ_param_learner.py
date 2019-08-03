#!/usr/bin/env python3

import sys
import json
import logging
import traceback

from tqdm import tqdm
import numpy as np 
    
def array_builder(filepath, term_idxs):
    array_out = np.zeros((len(term_idxs), len(term_idxs)))
    with open(filepath, "r") as handle:
        for line in handle:
            line = line.strip("\n").split(",")
            if line[0] in term_idxs and line[1] in term_idxs:
                array_out[term_idxs[line[0]], term_idxs[line[1]]] = float(line[2])
                array_out[term_idxs[line[1]], term_idxs[line[0]]] = float(line[2])
    return array_out

def main():   
    # Set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler("../logs/co-occ_param_learner.log")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Load in term frequencies
    with open("../data/term_freqs_rev_2_all_terms.json", "r") as handle:
        temp = json.load(handle)
    
    docs_list = list(temp.keys())
    partition = int(len(docs_list) * .8)

    train_docs = docs_list[0:partition]

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

    term_freqs = {}
    for doc in train_docs:
        if doc in solution.keys():
            term_freqs[doc] = temp[doc]

    # Load term subset to count for
    term_subset = []
    with open("../data/subset_terms_list", "r") as handle:
        for line in handle:
            term_subset.append(line.strip("\n"))
    
    # Dict for array assembly and lookup
    term_idxs = {term_subset[idx]: idx for idx in range(len(term_subset))}
    term_idxs_reverse = {idx: term_subset[idx] for idx in range(len(term_subset))}
                
    coocc_log_ratios = array_builder("../data/term_co-occ_log_likelihoods.csv", term_idxs)
    max_ratio = np.max(coocc_log_ratios)
    
    # testing tuning value to improve model
    tuning_val = 1

    thresh = 0.016

    ratio_cutoffs = [6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6]

    for ratio_cutoff in ratio_cutoffs:
        logger.info(f"Beginning feature engineering with co-occurrence log likelihood ratios at cutoff {ratio_cutoff}")
        for doc in tqdm(term_freqs.keys()):
            try:
                coocc_terms = {}
                for term in term_freqs[doc].keys():
                    if term in term_idxs.keys():
                        row = term_idxs[term]
                        for col in range(coocc_log_ratios.shape[0]):
                            if coocc_log_ratios[row, col] > ratio_cutoff: 
                                coocc_terms[term_idxs_reverse[col]] = ((coocc_log_ratios[row, col] / max_ratio) * tuning_val) * term_freqs[doc][term]
        
                for term in coocc_terms.keys():
                    if term in term_freqs[doc].keys():
                        term_freqs[doc][term] += coocc_terms[term]
                    else:
                        term_freqs[doc][term] = coocc_terms[term]
            except Exception as e:
                trace = traceback.format_exc()
                logger.error(repr(e))
                logger.critical(trace)
        
        predictions = {}
        
        # Predict
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
        
        logger.info(f"Cutoff: {ratio_cutoff}, f1: {f1}")

if __name__ == "__main__":
	main()