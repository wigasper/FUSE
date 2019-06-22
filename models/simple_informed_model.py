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
        
    # Get max freq for docs with more than 10 terms applied to their citations
    max_freq = 0
    for doc in term_freqs:
        for term in term_freqs[doc].keys():
            if len(term_freqs[doc]) > 10 and term_freqs[doc][term] > max_freq:
                max_freq = doc[1][term]
    
    # Divide values by max
    for doc in term_freqs:
        for term in term_freqs[doc].keys():
            doc[1][term] = doc[1][term] / max_freq
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
    
    logger.info("Beginning semantic similarity and co-occurrence incorporation")
    for doc in tqdm(term_freqs.keys()):
        try:
            # add semantically similar terms to each pool and weight by similarity
#            similar_terms = {}
            coocc_terms = {}
            for term in term_freqs[doc].keys():
                if term in term_idxs.keys():
                    row = term_idxs[term]
                    # coocc_log_ratios must have same dims here, may need to do something about this
                    for col in range(coocc_log_ratios.shape[0]):
#                        if sem_sims[row, col] > .5:
#                            similar_terms[term_idxs_reverse[col]] = sem_sims[row,col] * term_freqs[doc][term]
                        if coocc_log_ratios[row, col] > 2 or coocc_log_ratios[row, col] < -2:
                            coocc_terms[term_idxs_reverse[col]] = (coocc_log_ratios[row, col] / max_ratio) * term_freqs[doc][term]
#            for term in similar_terms.keys():
#                if term in term_freqs[doc].keys():
#                    term_freqs[doc][term] += similar_terms[term]
#                else:
#                    term_freqs[doc][term] = similar_terms[term]
    
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

############################################################################
#    logger.info("Writing output")
#    with open("../data/term_freqs_w_semsim_termcoocc.json", "w") as out:
#        json.dump(term_freqs, out)
#
#    with open("../data/term_freqs_w_semsim_termcoocc.json", "r") as handle:
#        term_freqs = json.load(handle)
#        
##############################################################
        
    thresholds = [x * .005 for x in range(0,200)]
    
    predictions = {}
    precisions = []
    recalls = []
    f1s = []
    
    # Run the model for all thresholds
    for thresh in tqdm(thresholds):
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
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)   
            
    from sklearn.metrics import auc
    from matplotlib import pyplot
    
    # AUC
    AUC = auc(recalls, precisions)
    print("AUC: ", AUC)
    pyplot.plot([0, 1], [0.5, 0.5], linestyle="--")
    pyplot.plot(recalls, precisions, marker=".")
    #pyplot.savefig("../pr_curve.png")
    pyplot.show()
    
    from notify import notify
    msg = "".join(["all done, auc: ", str(AUC), "\nmax F1: ", str(max(f1s))])
    notify(msg)
    
    # Write evaluation metrics
    with open("../data/informed_eval_metrics_1.csv", "w") as out:
        for index in range(len(thresholds)):
            out.write("".join([str(thresholds[index]), ","]))
            out.write("".join([str(precisions[index]), ","]))
            out.write("".join([str(recalls[index]), ","]))
            out.write("".join([str(f1s[index]), "\n"]))      
if __name__ == "__main__":
	main()
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
