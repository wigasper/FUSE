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

#thresh = .01

#thresholds = [x * .005 for x in range(0,200)]
#
#predictions = {}
#precision_avgs = []
#recall_avgs = []
#f1_avgs = []
#
## Results as a list of tuples: (FPR, TPR)
#for thresh in thresholds:
#    for doc in term_counts:
#        predictions[doc[0]] = [key for key, val in doc[1].items() if val > thresh]
#        
#    # Calculate precision and recall
#    precision_vals = []
#    recall_vals = []
#    f1_vals = []
#
#    for pmid in predictions.keys():
#        true_pos = len([pred for pred in predictions[pmid] if pred in solution[pmid]])
#        false_pos = len([pred for pred in predictions[pmid] if pred not in solution[pmid]])
#        false_neg = len([sol for sol in solution[pmid] if sol not in predictions[pmid]])
#        
#        if true_pos == 0:
#            precision = 0
#            recall = 0
#            f1 = 0
#            #precision_vals.append(0)
#            #recall_vals.append(0)
#        else:
#            precision = true_pos / (true_pos + false_pos)
#            recall = true_pos / (true_pos + false_neg)
#            f1 = (2 * precision * recall) / (precision + recall)
#            #precision_vals.append(true_pos / (true_pos + false_pos))
#            #recall_vals.append(true_pos / (true_pos + false_neg))
#        precision_vals.append(precision)
#        recall_vals.append(recall)
#        f1_vals.append(f1)
#    
#    # need to weight average precision here?
#    # see scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
#    precision_avgs.append(sum(precision_vals) / len(precision_vals))
#    recall_avgs.append(sum(recall_vals) / len(recall_vals))
#    f1_avgs.append(sum(f1_vals) / len(f1_vals))
    
#from sklearn.metrics import auc
#from matplotlib import pyplot
#
#print("AUC: ", auc(recall_avgs, precision_avgs))
#pyplot.plot([0, 1], [0.5, 0.5], linestyle="--")
#pyplot.plot(recall_avgs, precision_avgs, marker=".")
#pyplot.show()    
    
    
# note: there can be positive solution values that are not in any of the 
# citations so this skews the false_neg values - totally unknown to 
# classifier
    
# best F1 at threshold .015 ....
#test = [val for key, val in doc[1].items() for doc in term_counts]
#import seaborn as sns
#sns.distplot(test)

######################################################################
#######################################################################
######### try microaveraging here
# from towardatascience.com/journey-to-the-center-of-multi-label-classification0384c40229bff
# get a list of all possible classifications
#descriptors = [desc for desc in sample[1].keys() for sample in term_counts]
# This calculates TP, TN, FP, FN for every class

descriptors = []

for pmid in solution.keys():
    for descriptor in solution[pmid]:
        descriptors.append(descriptor)

for sample in term_counts:
    for descriptor in sample[1].keys():
        descriptors.append(descriptor)

descriptors = list(dict.fromkeys(descriptors))


thresholds = [x * .01 for x in range(0,50)]

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
    for desc in tqdm(descriptors):
        for pmid in predictions.keys():
            if desc in predictions[pmid] and desc in solution[pmid]:
                true_pos += 1
            if desc in predictions[pmid] and desc not in solution[pmid]:
                false_pos += 1
            if desc not in predictions[pmid] and desc in solution[pmid]:
                false_neg += 1
            if desc not in predictions[pmid] and desc not in solution[pmid]:
                true_neg += 1
    
    if true_pos == 0:
        precision = 0
        recall = 0
    else:
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)

    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    f1 = (2 * precision * recall) / (precision + recall)
    
    
    precisions.append(precision)
    recalls.append(recall)
    accuracies.append(accuracy)
    f1s.append(f1)
    tprs.append(true_pos / (true_pos + false_neg))
    fprs.append(false_pos / (true_neg + false_pos))

with open("./data/metrics_by_class.csv", "w") as out:
    for index in range(len(thresholds)):
        out.write("".join([thresholds[index], ","]))
        out.write("".join([precisions[index], ","]))
        out.write("".join([recalls[index], ","]))
        out.write("".join([accuracies[index], ","]))
        out.write("".join([f1s[index], ","]))
        out.write("".join([tprs[index], ","]))
        out.write("".join([fprs[index], "\n"]))
        
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


thresholds = [x * .01 for x in range(0,50)]

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
    else:
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    f1 = (2 * precision * recall) / (precision + recall)
    
    precisions.append(precision)
    recalls.append(recall)
    accuracies.append(accuracy)
    f1s.append(f1)
    tprs.append(true_pos / (true_pos + false_neg))
    fprs.append(false_pos / (true_neg + false_pos))
    
with open("./data/metrics_by_sample.csv", "w") as out:
    for index in range(len(thresholds)):
        out.write("".join([thresholds[index], ","]))
        out.write("".join([precisions[index], ","]))
        out.write("".join([recalls[index], ","]))
        out.write("".join([accuracies[index], ","]))
        out.write("".join([f1s[index], ","]))
        out.write("".join([tprs[index], ","]))
        out.write("".join([fprs[index], "\n"]))