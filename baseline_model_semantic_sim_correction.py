#!/usr/bin/env python3
import json

import numpy as np

# Load in term frequencies
with open("./data/term_freqs.json", "r") as handle:
    term_freqs = json.load(handle)

# Load in solution values
with open("./data/baseline_solution.json", "r") as handle:
    solution = json.load(handle)

# Load in semantic similarities
sim_cutoff = .8
sem_sims = {}
with open("./data/semantic_similarities.csv", "r") as handle:
    for line in handle:
        line = line.strip("\n").split(",")
        if float(line[2]) > sim_cutoff and not np.isnan(float(line[2])):
            sem_sims[",".join([line[0], line[1]])] = float(line[2])

# The baseline model. This model will predict a term if its frequency
# is greater than the threshold
thresholds = [x * .005 for x in range(0,200)]

predictions = {}
precisions = []
recalls = []
f1s = []

partial_correct_val = .75

# Run the model for all thresholds
for thresh in thresholds:
    # Predict
    for doc in term_freqs:
        predictions[doc[0]] = [key for key, val in doc[1].items() if val > thresh]
        
    # Get evaluation metrics
    true_pos = 0
    false_pos = 0
    false_neg = 0
    
    for pmid in predictions.keys():
        true_pos += len([pred for pred in predictions[pmid] if pred in solution[pmid]])
        for pred in predictions[pmid]:
            if pred not in solution[pmid]:
                found = False
                for sol in solution[pmid]:
                    if f"{pred},{sol}" in sem_sims.keys():
                        true_pos += partial_correct_val
                        false_pos += (1 - partial_correct_val)
                        found = True
                    if f"{sol},{pred}" in sem_sims.keys():
                        true_pos += partial_correct_val
                        false_pos += (1 - partial_correct_val)
                        found = True
                if not found:
                    false_pos += 1
        for sol in solution[pmid]:
            if sol not in predictions[pmid]:
                for pred in predictions[pmid]:
                    found = False
                    if f"{pred},{sol}" in sem_sims.keys():
                        true_pos += partial_correct_val
                        false_neg += (1 - partial_correct_val)
                        found = True
                    if f"{sol},{pred}" in sem_sims.keys():
                        true_pos += partial_correct_val
                        false_neg += (1 - partial_correct_val)
                        found = True
                if not found:
                    false_neg += 1

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
print("AUC: ", auc(recalls, precisions))
pyplot.plot([0, 1], [0.5, 0.5], linestyle="--")
pyplot.plot(recalls, precisions, marker=".")
pyplot.savefig("pr_curve.png")
pyplot.show()

########################################################


# Load in semantic similarities
sim_cutoff = .8
sem_sims = {}
with open("./data/semantic_similarities.csv", "r") as handle:
    for line in handle:
        line = line.strip("\n").split(",")
        if float(line[2]) > sim_cutoff and not np.isnan(float(line[2])):
            sem_sims[",".join([line[0], line[1]])] = float(line[2])

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
    for doc in term_freqs:
        predictions[doc[0]] = [key for key, val in doc[1].items() if val > thresh]
        for pred in predictions[doc[0]]:
            sims = [tup for tup in sem_sims.keys() if pred in tup]
#            for sim in sims:
#                for tup in sim.split(","):
#                    if tup != pred:
#                        pass
            for sim in sims:
                predictions[doc[0]].extend([tup for tup in sim.split(",") if tup != pred])
#            sims = [tup for tup in sim for sim in sims if tup != pred]
#            sims = [tup for tup in sim.split(",") for sim in sims if tup != pred]
            #sims = [tup for tup in sims.split(",") if tup != pred]
            #predictions[doc[0]].append(sims)

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

# AUC
print("AUC: ", auc(recalls, precisions))
pyplot.plot([0, 1], [0.5, 0.5], linestyle="--")
pyplot.plot(recalls, precisions, marker=".")
pyplot.savefig("pr_curve.png")
pyplot.show()

