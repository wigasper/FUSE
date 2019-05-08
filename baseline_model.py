#!/usr/bin/env python3
import json
from tqdm import tqdm

# Load in term frequencies
############################################
############# change path after rebuild ####
############################################
with open("./data/train_term_counts.json", "r") as handle:
    term_counts = json.load(handle)

# Load in solution values
with open("./data/baseline_solution.json", "r") as handle:
    solution = json.load(handle)
        
#########################################################
# this calculates TP, TN, FP, FN for every sample
    
descriptors = []

for pmid in solution.keys():
    for descriptor in solution[pmid]:
        descriptors.append(descriptor)

for sample in term_counts:
    for descriptor in sample[1].keys():
        descriptors.append(descriptor)

descriptors = list(dict.fromkeys(descriptors))


thresholds = [x * .005 for x in range(0,200)]

predictions = {}
precisions = []
recalls = []
accuracies = []
f1s = []
tprs = []
fprs = []

for thresh in thresholds:
    for doc in term_counts:
        predictions[doc[0]] = [key for key, val in doc[1].items() if val > thresh]
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    
    for pmid in tqdm(predictions.keys()):
        true_pos += len([pred for pred in predictions[pmid] if pred in solution[pmid]])
        true_neg += len([desc for desc in descriptors if desc not in solution[pmid] and desc not in predictions[pmid]])
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
        
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    
    precisions.append(precision)
    recalls.append(recall)
    accuracies.append(accuracy)
    f1s.append(f1)
    tprs.append(true_pos / (true_pos + false_neg))
    fprs.append(false_pos / (true_neg + false_pos))
    
with open("./data/baseline_eval_metrics.csv", "w") as out:
    for index in range(len(thresholds)):
        out.write("".join([str(thresholds[index]), ","]))
        out.write("".join([str(precisions[index]), ","]))
        out.write("".join([str(recalls[index]), ","]))
        out.write("".join([str(accuracies[index]), ","]))
        out.write("".join([str(f1s[index]), ","]))
        out.write("".join([str(tprs[index]), ","]))
        out.write("".join([str(fprs[index]), "\n"]))

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