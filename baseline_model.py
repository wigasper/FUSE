#!/usr/bin/env python3
import json

with open("./data/train_term_counts.json", "r") as handle:
    term_counts = json.load(handle)

threshold = .01

thresholds = [x * .005 for x in range(0,200)]

predictions = {}

# Results as a list of tuples: (FPR, TPR)
for thresh in thresholds:
    for doc in term_counts:
        predictions[doc[0]] = [key for (key, val) in doc[1].items() if val > thresh]
        
    # Calculate TP, TN, FP, FN for all predictions here
    
    # Append to results