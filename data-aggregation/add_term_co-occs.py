import sys
import json

import numpy as np
from tqdm import tqdm

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
    with open("../data/term_freqs_rev_3_all_terms.json", "r") as handle:
        term_freqs = json.load(handle)

    term_subset = []
    with open("../data/subset_terms_list", "r") as handle:
        for line in handle:
            term_subset.append(line.strip("\n"))

    # Dict for array assembly and lookup
    term_idxs = {term_subset[idx]: idx for idx in range(len(term_subset))}
    term_idxs_reverse = {idx: term_subset[idx] for idx in range(len(term_subset))}

    coocc_log_ratios = array_builder("../data/term_co-occ_log_likelihoods.csv", term_idxs)
    max_ratio = np.max(coocc_log_ratios)
    
    tuning_val = 1
    ratio_cutoff = 6.0

    for doc in tqdm(term_freqs.keys()):
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

    with open("../data/term_freqs_rev_3_w_cooccs.json", "w") as out:
        json.dump(term_freqs, out)

if __name__ == "__main__":
    main()
