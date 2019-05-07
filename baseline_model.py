#!/usr/bin/env python3
import json

# Load in term frequencies
############################################
############# change path after rebuild ####
############################################
with open("./data/train_term_counts.json", "r") as handle:
    term_counts = json.load(handle)

# Load in solution values
with open("./data/baseline_solution.json", "r") as handle:
    solution = json.load(handle)

#thresh = .01

thresholds = [x * .005 for x in range(0,200)]

predictions = {}
precision_avgs = []
recall_avgs = []

# Results as a list of tuples: (FPR, TPR)
for thresh in thresholds:
    for doc in term_counts:
        predictions[doc[0]] = [key for key, val in doc[1].items() if val > thresh]
        
    # Calculate precision and recall
    precision_vals = []
    recall_vals = []

    for pmid in predictions.keys():
        true_pos = len([pred for pred in predictions[pmid] if pred in solution[pmid]])
        false_pos = len([pred for pred in predictions[pmid] if pred not in solution[pmid]])
        false_neg = len([sol for sol in solution[pmid] if sol not in predictions[pmid]])
        
        if true_pos == 0:
            precision_vals.append(0)
            recall_vals.append(0)
        else:
            precision_vals.append(true_pos / (true_pos + false_pos))
            recall_vals.append(true_pos / (true_pos + false_neg))
    
    precision_avgs.append(sum(precision_vals) / len(precision_vals))
    recall_avgs.append(sum(recall_vals) / len(recall_vals))
    
from sklearn.metrics import auc
from matplotlib import pyplot

print("AUC: ", auc(recall_avgs, precision_avgs))
pyplot.plot([0, 1], [0.5, 0.5], linestyle="--")
pyplot.plot(recall_avgs, precision_avgs, marker=".")
pyplot.show()    
    
    
# note: there can be positive solution values that are not in any of the 
# citations so this skews the false_neg values - totally unknown to 
# classifier
    

test = [val for key, val in doc[1].items() for doc in term_counts]
import seaborn as sns
sns.distplot(test)