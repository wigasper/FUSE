#!/usr/bin/env python3

import json

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
    
    for doc in tqdm(term_freqs.keys()):
        # add semantically similar terms to each pool and weight by similarity
        similar_terms = {}
        coocc_terms = {}
        for term in term_freqs[doc].keys():
            if term in term_idxs.keys():
                row = term_idxs[term]
                # coocc_log_ratios must have same dims here, may need to do something about this
                for col in range(sem_sims.shape[0]):
                    if sem_sims[row, col] > .5 :
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
    
    # before:
    term_freqs["21502780"]
    {'D000063': 0.05405405405405406,
     'D000328': 0.08108108108108109,
     'D001334': 0.08108108108108109,
     'D040421': 0.08108108108108109,
     'D005260': 0.08108108108108109,
     'D006801': 0.08108108108108109,
     'D008297': 0.08108108108108109,
     'D008875': 0.08108108108108109,
     'D012307': 0.02702702702702703,
     'D014739': 0.05405405405405406,
     'D000293': 0.02702702702702703,
     'D000368': 0.02702702702702703,
     'D018592': 0.02702702702702703,
     'D004636': 0.02702702702702703,
     'D016016': 0.02702702702702703,
     'D018570': 0.02702702702702703,
     'D013997': 0.02702702702702703,
     'D014904': 0.02702702702702703,
     'D017677': 0.02702702702702703,
     'D012309': 0.02702702702702703,
     'D017678': 0.02702702702702703}
    
    
    
    
    
    
    
    
    # The baseline model. This model will predict a term if its frequency
    # is greater than the threshold
    thresholds = [x * .005 for x in range(0,200)]
    
    predictions = {}
    precisions = []
    recalls = []
    f1s = []
    
    # Run the model for all thresholds
    for thresh in thresholds:
        # Predict
        for doc in tqdm(term_freqs):
            term_pool = [t for term in doc[1].keys() if term in sem_sims.keys() for t in sem_sims[term]]
            term_pool.extend(doc[1].keys())
            term_pool = list(dict.fromkeys(term_pool))
            pred = []
            for term in term_pool:
                term_val = []
                if term in doc[1].keys():
                    term_val.append(doc[1][term])
                if term in sem_sims.keys():
                    term_val.extend([doc[1][t] for t in doc[1].keys() if t in sem_sims[term]])
                term_val = sum(term_val)
                if term_val > thresh:
                    pred.append(term)
            predictions[doc[0]] = pred
            
        # Get evaluation metrics
        true_pos = 0
        false_pos = 0
        false_neg = 0
        
        for pmid in predictions.keys():
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
    
        print("thresh: " + str(thresh))
        print("precision: " + str(precision))
        print("recall: " + str(recall))
        print("f1: " + str(f1))
    
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
            
    from sklearn.metrics import auc
    from matplotlib import pyplot
    with open("../data/baseline_sem_sim_corr_results_2", "w") as out:
        out.write("".join(["auc: ", str(auc(recalls, precisions))]))
        out.write("\n")
        out.write("f1s: ")
        out.write(",".join([str(f1) for f1 in f1s]))
        out.write("\n")
        out.write("precisions: ")
        out.write(",".join([str(prec) for prec in precisions]))
        out.write("\n")
        out.write("recalls: ")
        out.write(",".join([str(rec) for rec in recalls]))
    
    # AUC
    #print("AUC: ", auc(recalls, precisions))
    pyplot.plot([0, 1], [0.5, 0.5], linestyle="--")
    pyplot.plot(recalls, precisions, marker=".")
    pyplot.savefig("../pr_curve.png")
    pyplot.show()