#!/usr/bin/env python3
import json

# Load in term frequencies
with open("../data/term_freqs.json", "r") as handle:
    term_freqs = json.load(handle)

# Load in solution values
with open("../data/baseline_solution.json", "r") as handle:
    solution = json.load(handle)

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

# Write evaluation metrics
with open("../data/baseline_eval_metrics.csv", "w") as out:
    for index in range(len(thresholds)):
        out.write("".join([str(thresholds[index]), ","]))
        out.write("".join([str(precisions[index]), ","]))
        out.write("".join([str(recalls[index]), ","]))
        out.write("".join([str(f1s[index]), "\n"]))      
        
from sklearn.metrics import auc
from matplotlib import pyplot

# AUC
print("AUC: ", auc(recalls, precisions))
pyplot.plot([0, 1], [0.5, 0.5], linestyle="--")
pyplot.plot(recalls, precisions, marker=".")
pyplot.savefig("../pr_curve.png")
pyplot.show()
