#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 15:14:24 2019

@author: wkg
"""
parents = []
children = []

edges = []

with open("./data/edge_list_build_aug7.csv", "r") as handle:
    for line in handle:
        line = line.strip("\n").split(",")
        parents.append(line[0])
        children.append(line[1])
        edges.append([line[0], line[1]])
        
from collections import Counter

counts = Counter(children)

child_counts = [[key, val] for key, val in counts.items()]

child_counts = sorted(child_counts, key=lambda child_counts: child_counts[1],
                      reverse=True)

parents = set(parents)

edges_filt = [edge for edge in edges if edge[1] in parents]
