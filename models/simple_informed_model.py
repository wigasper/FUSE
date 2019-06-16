#!/usr/bin/env python3

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
    handler = logging.FileHandler("../logs/simple_informed_model.log")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Load in term frequencies
    with open("../data/term_freqs.json", "r") as handle:
        temp = json.load(handle)
    # original file clearly needs to be rebuilt as dict
    term_freqs = {}
    for doc in temp:
        term_freqs[doc[0]] = doc[1]
    
    # Load in solution values
    with open("../data/baseline_solution.json", "r") as handle:
        solution = json.load(handle)
    
    # Load term subset to count for
    term_subset = []
    with open("../data/subset_terms_list", "r") as handle:
        for line in handle:
            term_subset.append(line.strip("\n"))
    
    # Dict for array assembly and lookup
    term_idxs = {term_subset[idx]: idx for idx in range(len(term_subset))}
    term_idxs_reverse = {idx: term_subset[idx] for idx in range(len(term_subset))}
    
    sem_sims = array_builder("../data/semantic_similarities_rev1.csv", term_idxs)
                
    coocc_log_ratios = array_builder("../data/term_co-occ_log_likelihoods.csv", term_idxs)
    max_ratio = np.max(coocc_log_ratios)
    
    logger.info("Beginning semantic similarity and co-occurrence incorporation")
    for doc in tqdm(term_freqs.keys()):
        try:
            # add semantically similar terms to each pool and weight by similarity
            similar_terms = {}
            coocc_terms = {}
            for term in term_freqs[doc].keys():
                if term in term_idxs.keys():
                    row = term_idxs[term]
                    # coocc_log_ratios must have same dims here, may need to do something about this
                    for col in range(sem_sims.shape[0]):
                        if sem_sims[row, col] > .5:
                            similar_terms[term_idxs_reverse[col]] = sem_sims[row,col] * term_freqs[doc][term]
                        if coocc_log_ratios[row, col] > 5:
                            coocc_terms[term_idxs_reverse[col]] = (coocc_log_ratios[row, col] / max_ratio) * term_freqs[doc][term]
            for term in similar_terms.keys():
                if term in term_freqs[doc].keys():
                    term_freqs[doc][term] += similar_terms[term]
                else:
                    term_freqs[doc][term] = similar_terms[term]
    
            for term in coocc_terms.keys():
                if term in term_freqs[doc].keys():
                    term_freqs[doc][term] += coocc_terms[term]
                else:
                    term_freqs[doc][term] = coocc_terms[term]
        except Exception as e:
            trace = traceback.format_exc()
            logger.error(repr(e))
            logger.critical(trace)
    logger.info("Semantic similarity and co-occurrence incorporation complete")
    
    logger.info("Writing output")
    with open("../data/term_freqs_w_semsim_termcoocc.json", "w") as out:
        json.dump(term_freqs, out)

"""

############################################
                ############################
    coocc_log_ratios = array_builder("../data/term_co-occ_log_likelihoods.csv", term_idxs)
    max_ratio = np.max(coocc_log_ratios)
    for doc in tqdm(term_freqs.keys()):
        coocc_terms = {}
        for term in term_freqs[doc].keys():
            if term in term_idxs.keys():
                row = term_idxs[term]
                for col in range(coocc_log_ratios.shape[0]):
                    if coocc_log_ratios[row, col] > 5:
                        coocc_terms[term_idxs_reverse[col]] = (coocc_log_ratios[row, col] / max_ratio) * term_freqs[doc][term]
        for term in coocc_terms.keys():
            if term in term_freqs[doc].keys():
                term_freqs[doc][term] += coocc_terms[term]
            else:
                term_freqs[doc][term] = similar_terms[term]
    
"""