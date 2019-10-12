#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 13:35:41 2019

@author: wkg
"""

from nxviz.plots import CircosPlot
import networkx as nx
import matplotlib.pyplot as plt

term_names = {}
with open("data/mesh_data.tab", "r") as handle:
    for line in handle:
        line = line.strip("\n").split("\t")
        term_names[line[0]] = line[1]

nodes = ["D013527", "D005791", "D013514", "D013812", "D006296", "D003631",
         "D059035", "D026421", "D000529", "D011300", "D000161", "D011182",
         "D011300"]

nodes_idxs = {node: idx for idx, node in enumerate(nodes)}

nodes_idxs_rev = {idx: node for idx, node in enumerate(nodes)}

from itertools import combinations
import numpy as np

adj_matrix = np.zeros((len(nodes), len(nodes)))



#combs = combinations(nodes, 2)
#
#combs = list(combs)
#
#edge_list = [(comb[0], comb[1]) for comb in combs]
#
#G = nx.from_edgelist(edge_list)
#
#d = {}
#for comb in combs:
#    d[comb[0]] = {comb[1]: 0}
    
with open("data/semantic_similarities_rev1.csv", "r") as handle:
    for line in handle:
        line = line.strip("\n").split(",")
        if line[0] in nodes and line[1] in nodes:
            adj_matrix[nodes_idxs[line[0]], nodes_idxs[line[1]]] = float(line[2])
            adj_matrix[nodes_idxs[line[1]], nodes_idxs[line[0]]] = float(line[2])



weights = []
min_col = 1
for row in range(adj_matrix.shape[0]):
    for col in range(min_col, adj_matrix.shape[0]):
        weight = adj_matrix[row, col]
        if weight > 0:
            weights.append(weight * 20)
    min_col += 1
    
g = nx.Graph(adj_matrix)
g = nx.relabel_nodes(g, nodes_idxs_rev)
g = nx.relabel_nodes(g, term_names)

c = CircosPlot(g, dpi=800, fig_size=(20,20), edge_width=weights, node_labels=True)
c.draw()
#plt.show()
plt.savefig("/home/wkg/Desktop/circos_test.png", bbox_inches="tight", dpi=600)
