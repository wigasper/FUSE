#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 13:28:08 2019

@author: wkg
"""

import numpy as np

co_matrix = np.load("./data/co-occurrence-matrix.npy")


term_subset = []
with open("./data/subset_terms_list", "r") as handle:
    for line in handle:
        term_subset.append(line.strip("\n"))
#term_subset = set(term_subset)

test2 = []
for term in term_subset:
    test2.append(term)

row = 0
col = 0
max_val = 0

# for getting the max
for r in test:
    for c in r:
        if row != col and test[row,col] > max_val:
            max_loc = (row, col)
            max_val = test[row,col]
            
        col += 1
    col = 0    
    row += 1
    

row = 0
col = 0
# get all paris
pairs = []
for r in test:
    for c in r:
        if row != col and test[row,col] > 0:
            pairs.append([term_subset[row],term_subset[col],test[row,col]])
            
        col += 1
    col = 0    
    row += 1

sorted_pairs = sorted(pairs, key=lambda l:l[2], reverse=True)


# Load term subset to count for
term_subset = []
with open("./data/subset_terms_list", "r") as handle:
    for line in handle:
        term_subset.append(line.strip("\n"))

doc_count = 1900000
    # Compute probabilities to compare against
term_counts = {}

with open("./data/mesh_data.tab", "r") as handle:
    for line in handle:
        line = line.strip("\n").split("\t")
        term_counts[line[0]] = 0

with open("./data/pm_bulk_doc_term_counts.csv", "r") as handle:
    for _ in range(doc_count):
        line = handle.readline()
        line = line.strip("\n").split(",")
        terms = line[1:]
        terms = [term for term in terms if term]
        for term in terms:
            term_counts[term] += 1
            
# Get probability of each term for the document set
total_terms = sum(term_counts.values())

for term in term_counts:
    term_counts[term] = term_counts[term] / total_terms

# Create the expected probability matrix
expected = np.zeros((len(term_subset), len(term_subset)))

for row in range(expected.shape[0]):
    for col in range(expected.shape[1]):
        expected[row, col] = term_counts[term_subset[row]] * term_counts[term_subset[col]]

expected[expected == 0] = np.NaN
# Get the total number of co-occurrences
total_cooccurrs = 0
for row in range(co_matrix.shape[0]):
    for col in range(co_matrix.shape[1]):
        if row != col:
            total_cooccurrs += co_matrix[row, col]
total_cooccurrs = total_cooccurrs / 2

temp_total_array = np.full((len(co_matrix), len(co_matrix)), total_cooccurrs)

co_matrix = np.divide(co_matrix, temp_total_array)

differential = np.divide(co_matrix, expected)

differential[differential == 0] = np.NaN

differential = np.log(differential)

term_pair_scores = []

for row in range(differential.shape[0]):
    for col in range(row + 1, differential.shape[1]):
        if not np.isnan(differential[row,col]):
            term_pair_scores.append([term_subset[row], term_subset[col], differential[row,col]])
            
sorted_pairs = sorted(term_pair_scores, key=lambda l:l[2], reverse=True)
