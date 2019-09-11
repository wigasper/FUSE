#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 14:27:24 2019

@author: wkg
"""

import seaborn as sns
import numpy as np

sns.set(font="helvetica")
# stuff that both need:

term_names = {}
with open("./data/mesh_data.tab", "r") as handle:
    for line in handle:
        line = line.strip("\n").split("\t")
        term_names[line[0]] = line[1]
        
top_500 = []
with open("./data/top_500_terms", "r") as handle:
    for line in handle:
        top_500.append(line.strip("\n"))

##############################################
# this is more like a correlation matrix

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

mask = np.zeros_like(log_likes)
mask[np.triu_indices_from(mask)] = True

labels = [term_names[uid] for uid in top_25]

with sns.axes_style("white"):
    plot = sns.heatmap(log_likes, mask=mask, square=True, xticklabels=labels, 
                     yticklabels=labels, 
                     cbar_kws={"label": "Co-occurrence Log-likelihood Ratio"})

plot.set_xticklabels(plot.get_xticklabels(), fontsize=7)
plot.set_yticklabels(plot.get_yticklabels(), fontsize=7)

fig = plot.get_figure()
fig.savefig("/home/wkg/Desktop/test.nr.png", bbox_inches="tight", dpi=600)

##################################

# This creates a non-redundant heatmap

top_25_evens = [uid for idx, uid in enumerate(top_500) if idx % 2 == 0][:25]
top_25_odds = [uid for idx, uid in enumerate(top_500) if idx % 2 == 1][:25]

top_25_evens_set = set(top_25_evens)
top_25_odds_set = set(top_25_odds)

evens_indices = {term: idx for idx, term in enumerate(top_25_evens)}
odds_indices = {term: idx for idx, term in enumerate(top_25_odds)}

log_likes = np.zeros((25, 25))

with open("./data/term_co-occ_log_likelihoods.csv", "r") as handle:
    for line in handle:
        line = line.strip("\n").split(",")
        if line[0] in top_25_evens_set and line[1] in top_25_odds_set:
            log_likes[evens_indices[line[0]], odds_indices[line[1]]] = float(line[2])
        if line[0] in top_25_odds_set and line[1] in top_25_evens_set:
            log_likes[evens_indices[line[1]], odds_indices[line[0]]] = float(line[2])

y_labels = [term_names[uid] for uid in top_25_evens]
x_labels = [term_names[uid] for uid in top_25_odds]

plot = sns.heatmap(log_likes, xticklabels=x_labels, yticklabels=y_labels,
                   cbar_kws={"label": "Co-occurrence Log-likelihood Ratio"})
                   #cmap="Greens")
#plot.set_xticklabels(plot.get_xticklabels(), rotation=85)
plot.set_xticklabels(plot.get_xticklabels(), fontsize=7)
plot.set_yticklabels(plot.get_yticklabels(), fontsize=7)

fig = plot.get_figure()
fig.savefig("/home/wkg/Desktop/test3_labs.1.png", bbox_inches="tight", dpi=600)


#############################################
#
#log_likelihood_ratios = []
#with open("./data/term_co-occ_log_likelihoods.csv", "r") as handle:
#    for line in handle:
#        line = line.strip("\n").split(",")
#        line[2] = float(line[2])
#        log_likelihood_ratios.append(line)
#        
#sorted_ratios = sorted(log_likelihood_ratios, key=lambda l:l[2], reverse=True)
#
#rows = [i[0] for i in sorted_ratios[:30]]
#cols = [i[1] for i in sorted_ratios[:30]]
#
#rows = list(dict.fromkeys(rows))[:25]
#cols = list(dict.fromkeys(cols))[:25]
#
#row_indices = {term: idx for idx, term in enumerate(rows)}
#col_indices = {term: idx for idx, term in enumerate(cols)}
#
#log_likes = np.zeros((25, 25))
#
#with open("./data/term_co-occ_log_likelihoods.csv", "r") as handle:
#    for line in handle:
#        line = line.strip("\n").split(",")
#        if line[0] in rows and line[1] in cols:
#            log_likes[row_indices[line[0]], col_indices[line[1]]] = float(line[2])
#        if line[1] in rows and line[0] in cols:
#            log_likes[row_indices[line[1]], col_indices[line[0]]] = float(line[2])
#            
#            
#for item in sorted_ratios:
#    if item[0] == 'D002525' and item[1] == 'D049711':
#        print(f"yes, {item[2]}")
#    
#    if item[0] == 'D049711' and item[1] == 'D002525':
#        print(f"yes2, {item[2]}")
#        
#for item in sorted_ratios:
#    if item[2] == 0.0:
#        print(f"{item[0]}, {item[1]}")
#        break