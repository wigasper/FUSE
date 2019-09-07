#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 14:27:24 2019

@author: wkg
"""

import seaborn as sns
import numpy as np

top_500 = []
with open("./data/top_500_terms", "r") as handle:
    for line in handle:
        top_500.append(line.strip("\n"))

top_25 = top_500[:25]
top_25_set = set(top_25)

term_indices = {term: idx for idx, term in enumerate(top_25)}

log_likes = np.zeros((25, 25))

with open("./data/term_co-occ_log_likelihoods.csv", "r") as handle:
    for line in handle:
        line = line.strip("\n").split(",")
        if line[0] in top_25_set and line[1] in top_25_set:
            log_likes[term_indices[line[0]], term_indices[line[1]]] = float(line[2])
            log_likes[term_indices[line[1]], term_indices[line[0]]] = float(line[2])
            
plot = sns.heatmap(log_likes)
fig = plot.get_figure()
fig.savefig("/home/wkg/Desktop/test.png")
