##!/usr/bin/env python3
#import json
#
#import numpy as np
#
## Load in term frequencies
#with open("./data/term_freqs.json", "r") as handle:
#    term_freqs = json.load(handle)
#
## Load in solution values
#with open("./data/baseline_solution.json", "r") as handle:
#    solution = json.load(handle)
#    
## Load in semantic similarities
#sim_cutoff = .8
#sem_sims = {}
#with open("./data/semantic_similarities.csv", "r") as handle:
#    for line in handle:
#        line = line.strip("\n").split(",")
#        if float(line[2]) > sim_cutoff and not np.isnan(float(line[2])):
#            sem_sims[",".join([line[0], line[1]])] = float(line[2])
#
## The baseline model. This model will predict a term if its frequency
## is greater than the threshold
#thresholds = [x * .005 for x in range(0,200)]
#
#predictions = {}
#precisions = []
#recalls = []
#f1s = []
#
#partial_correct_val = 0
#
## Run the model for all thresholds
#for thresh in thresholds:
#    # Predict
#    for doc in term_freqs:
#        predictions[doc[0]] = [key for key, val in doc[1].items() if val > thresh]
#        
#    # Get evaluation metrics
#    true_pos = 0
#    false_pos = 0
#    false_neg = 0
#    
#    for pmid in predictions.keys():
#        true_pos += len([pred for pred in predictions[pmid] if pred in solution[pmid]])
#        for pred in predictions[pmid]:
#            if pred not in solution[pmid]:
#                found = False
#                for sol in solution[pmid]:
#                    if f"{pred},{sol}" in sem_sims.keys():
#                        true_pos += partial_correct_val
#                        false_pos += (1 - partial_correct_val)
#                        found = True
#                    if f"{sol},{pred}" in sem_sims.keys():
#                        true_pos += partial_correct_val
#                        false_pos += (1 - partial_correct_val)
#                        found = True
#                if not found:
#                    false_pos += 1
#        for sol in solution[pmid]:
#            if sol not in predictions[pmid]:
#                for pred in predictions[pmid]:
#                    found = False
#                    if f"{pred},{sol}" in sem_sims.keys():
#                        true_pos += partial_correct_val
#                        false_neg += (1 - partial_correct_val)
#                        found = True
#                    if f"{sol},{pred}" in sem_sims.keys():
#                        true_pos += partial_correct_val
#                        false_neg += (1 - partial_correct_val)
#                        found = True
#                if not found:
#                    false_neg += 1
#
#    if true_pos == 0:
#        precision = 0
#        recall = 0
#        f1 = 0
#    else:
#        precision = true_pos / (true_pos + false_pos)
#        recall = true_pos / (true_pos + false_neg)
#        f1 = (2 * precision * recall) / (precision + recall)
#    
#    precisions.append(precision)
#    recalls.append(recall)
#    f1s.append(f1)
# 
#
#from sklearn.metrics import auc
#from matplotlib import pyplot
#
## AUC
#print("AUC: ", auc(recalls, precisions))
#pyplot.plot([0, 1], [0.5, 0.5], linestyle="--")
#pyplot.plot(recalls, precisions, marker=".")
#pyplot.savefig("pr_curve.png")
#pyplot.show()
#
#########################################################
#
## Doing it this way pretty much improves recall across the board
## lower AUC and f1s though
## Load in semantic similarities
#sim_cutoff = .9
#sem_sims = {}
#
#with open("./data/semantic_similarities.csv", "r") as handle:
#    for line in handle:
#        line = line.strip("\n").split(",")
#        if float(line[2]) > sim_cutoff and not np.isnan(float(line[2])):
#            if line[0] in sem_sims.keys():
#                sem_sims[line[0]].append(line[1])
#            else:
#                sem_sims[line[0]] = [line[1]]
#            if line[1] in sem_sims.keys():
#                sem_sims[line[1]].append(line[0])
#            else:
#                sem_sims[line[1]] = [line[0]]
#
## The baseline model. This model will predict a term if its frequency
## is greater than the threshold
#thresholds = [x * .005 for x in range(0,200)]
#
#predictions = {}
#precisions = []
#recalls = []
#f1s = []
#
#
## Run the model for all thresholds
#for thresh in thresholds:
#    # Predict
#    for doc in term_freqs:
#        predictions[doc[0]] = [key for key, val in doc[1].items() if val > thresh]
#        sims = []
#        for pred in predictions[doc[0]]:
#            if pred in sem_sims.keys():
#                sims.extend(sem_sims[pred])
#        predictions[doc[0]].extend(sims)
#
#    # Get evaluation metrics
#    true_pos = 0
#    false_pos = 0
#    false_neg = 0
#    
#    for pmid in predictions.keys():
#        true_pos += len([pred for pred in predictions[pmid] if pred in solution[pmid]])
#        false_pos += len([pred for pred in predictions[pmid] if pred not in solution[pmid]])
#        false_neg += len([sol for sol in solution[pmid] if sol not in predictions[pmid]])
#
#    if true_pos == 0:
#        precision = 0
#        recall = 0
#        f1 = 0
#    else:
#        precision = true_pos / (true_pos + false_pos)
#        recall = true_pos / (true_pos + false_neg)
#        f1 = (2 * precision * recall) / (precision + recall)
#    
#    precisions.append(precision)
#    recalls.append(recall)
#    f1s.append(f1)
#
#from sklearn.metrics import auc
#from matplotlib import pyplot
#
## AUC
#print("AUC: ", auc(recalls, precisions))
#pyplot.plot([0, 1], [0.5, 0.5], linestyle="--")
#pyplot.plot(recalls, precisions, marker=".")
#pyplot.savefig("pr_curve.png")
#pyplot.show()

#################################################3

"""
import json

from tqdm import tqdm
import numpy as np 

# Load in term frequencies
with open("./data/term_freqs.json", "r") as handle:
    term_freqs = json.load(handle)

# Load in solution values
with open("./data/baseline_solution.json", "r") as handle:
    solution = json.load(handle)

sim_cutoff = .8
sem_sims = {}

with open("./data/semantic_similarities_rev0.csv", "r") as handle:
    for line in handle:
        line = line.strip("\n").split(",")
        if float(line[2]) > sim_cutoff and not np.isnan(float(line[2])):
            if line[0] in sem_sims.keys():
                sem_sims[line[0]].append(line[1])
            else:
                sem_sims[line[0]] = [line[1]]
            if line[1] in sem_sims.keys():
                sem_sims[line[1]].append(line[0])
            else:
                sem_sims[line[1]] = [line[0]]

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
        pred = []
        for term in doc[1]:
            term_val = [doc[1][term]]
            sim_term_freqs = []
            if term in sem_sims.keys():
                term_val.extend([doc[1][t] for t in doc[1].keys() if t in sem_sims[term]])
            term_val = sum(term_val)
            if term_val > thresh:
                pred.append(term)
        predictions[doc[0]] = pred
#            for key in doc[1].keys():
#                if key in sem_sims.keys() and key in sem_sims[term]:
#            sim_term_freqs = [doc[1][k] for k in doc[1].keys() if term in sem_sims.keys() and k in sem_sims[term]]
        #predictions[doc[0]] = [key for key, val in doc[1].items() if val > thresh]
        
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
    
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
        
from sklearn.metrics import auc
from matplotlib import pyplot
with open("./data/baseline_sem_sim_corr_results_1", "w") as out:
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
pyplot.savefig("pr_curve.png")
pyplot.show()
"""
#########################################

import json

from tqdm import tqdm
import numpy as np 

# Load in term frequencies
with open("./data/term_freqs.json", "r") as handle:
    term_freqs = json.load(handle)

# Load in solution values
with open("./data/baseline_solution.json", "r") as handle:
    solution = json.load(handle)

sim_cutoff = .8
sem_sims = {}

with open("./data/semantic_similarities_rev0.csv", "r") as handle:
    for line in handle:
        line = line.strip("\n").split(",")
        if float(line[2]) > sim_cutoff and not np.isnan(float(line[2])):
            if line[0] in sem_sims.keys():
                sem_sims[line[0]].append(line[1])
            else:
                sem_sims[line[0]] = [line[1]]
            if line[1] in sem_sims.keys():
                sem_sims[line[1]].append(line[0])
            else:
                sem_sims[line[1]] = [line[0]]

# The baseline model. This model will predict a term if its frequency
# is greater than the threshold
thresholds = [x * .005 for x in range(0,40)]

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
    
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
        
from sklearn.metrics import auc
from matplotlib import pyplot
with open("./data/baseline_sem_sim_corr_results_1", "w") as out:
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
pyplot.savefig("pr_curve.png")
pyplot.show()
