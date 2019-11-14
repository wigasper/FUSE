#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 13:35:41 2019

@author: wkg
"""

from nxviz.plots import CircosPlot
import networkx as nx
import matplotlib.pyplot as plt
from random import choice

term_names = {}
uids = []
with open("data/mesh_data.tab", "r") as handle:
    for line in handle:
        line = line.strip("\n").split("\t")
        term_names[line[0]] = line[1]
        uids.append(line[0])
        
nodes_g1 = ["D013527", "D005791", "D013514", "D013812", "D006296", "D003631",
         "D059035", "D026421", "D000529", "D011300", "D000161", "D011182",
         "D011300"]

nodes_g2 = ["D008660", "D000222", "D000917", "D055633", "D010599", "D055614",
            "D005075", "D015321", "D007109", "D008213", "D000220", "D000898",
            "D010587"]

nodes_g3 = [choice(uids) for _ in range(30)]
nodes_g3 = [node for node in nodes_g3 if node not in nodes_g1 and node not in nodes_g2][0:8]

nodes = [node for node in nodes_g1]
nodes.extend([node for node in nodes_g2])
nodes.extend([node for node in nodes_g3])

nodes_idxs = {node: idx for idx, node in enumerate(nodes)}

nodes_idxs_rev = {idx: node for idx, node in enumerate(nodes)}

#from itertools import combinations
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
            weights.append(weight * 10)
    min_col += 1
    
g = nx.Graph(adj_matrix)
g = nx.relabel_nodes(g, nodes_idxs_rev)

for node in g.nodes():
    if node in nodes_g1:
        g.node[node]["class"] = "g1"
    if node in nodes_g2:
        g.node[node]["class"] = "g2"
    if node in nodes_g3:
        g.node[node]["class"] = "g3"

g = nx.relabel_nodes(g, term_names)

#c = CircosPlot(g, node_label_layout="numbers", dpi=800, fontsize=10, 
#               node_color="class", node_grouping="class",
#               fig_size=(20,20), edge_width=weights, node_labels=True)
c = CircosPlot(g, dpi=800, fontsize=8, node_label_layout="numbers", 
               node_color="class", node_grouping="class",
               fig_size=(20,20), edge_width=weights, node_labels=True)
c.draw()
#plt.show()
plt.savefig("/home/wkg/Desktop/circos_test5.png", bbox_inches="tight", dpi=600)
