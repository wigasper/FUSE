#!/usr/bin/env python3

import json

from sklearn.metrics import auc
from matplotlib import pyplot

# Load in term frequencies
with open("./data/term_freqs.json", "r") as handle:
    term_freqs = json.load(handle)

# Load in solution values
with open("./data/baseline_solution.json", "r") as handle:
    solution = json.load(handle)

ui_depths = {}
with open("./data/mesh_data.tab", "r") as handle:
    for line in handle:
        line = line.strip("\n").split("\t")
        ui_depths[line[0]] = int(line[2])
        # may also be able to use distinct_tree_posits here, from line[3]

#depths = [v for k,v in ui_depths.items()]
def weight(ui, weight_dict=ui_depths):
    return weight_dict[ui] / 4
# Run the baseline model. This model will predict a term if its frequency
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
        predictions[doc[0]] = [key for key, val in doc[1].items() if (weight(key) * val) > thresh]
        
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
        
# AUC
print("AUC: ", auc(recalls, precisions))
pyplot.plot([0, 1], [0.5, 0.5], linestyle="--")
pyplot.plot(recalls, precisions, marker=".")
pyplot.show()

