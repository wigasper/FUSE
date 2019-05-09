#!/usr/bin/env python3

import json

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
        ui_depths[line[0]] = line[2]
        # may also be able to use distinct_tree_posits here, from line[3]

depths = [v for k,v in ui_depths.items()]
def weight(ui, weight_dict=ui_depths):
    
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
        predictions[doc[0]] = [key for key, val in doc[1].items() if val > thresh]
        
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

# Write evaluation metrics
with open("./data/baseline_eval_metrics.csv", "w") as out:
    for index in range(len(thresholds)):
        out.write("".join([str(thresholds[index]), ","]))
        out.write("".join([str(precisions[index]), ","]))
        out.write("".join([str(recalls[index]), ","]))
        out.write("".join([str(f1s[index]), "\n"]))

# This code is for reading in evaluation metrics for later examination
precisions = []
recalls = []
accuracies = []
f1s = []
tprs = []
fprs = []

with open("./data/baseline_eval_metrics.csv", "r") as handle:
    for line in handle:
        line = line.strip("\n").split(",")
        precisions.append(float(line[1]))
        recalls.append(float(line[2]))
        accuracies.append(float(line[3]))
        f1s.append(float(line[4]))
        tprs.append(float(line[5]))
        fprs.append(float(line[6]))
        
        
from sklearn.metrics import auc
from matplotlib import pyplot

# AUC
print("AUC: ", auc(recalls, precisions))
pyplot.plot([0, 1], [0.5, 0.5], linestyle="--")
pyplot.plot(recalls, precisions, marker=".")
pyplot.show()



