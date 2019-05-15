#!/usr/bin/env python3

import os
from math import log
import logging

import numpy as np
from bs4 import BeautifulSoup
from tqdm import tqdm

# Set up logging
logging.basicConfig(filename="errors.log", level=logging.INFO,
                    format="AIC Compute: %(levelname)s - %(message)s")
logger = logging.getLogger()

uids = []
names = []
trees = []

with open("./data/mesh_data.tab", "r") as handle:
    for line in handle:
        line = line.strip("\n").split("\t")
        uids.append(line[0])
        names.append(line[1])
        trees.append(line[4].split(","))

docs = os.listdir("./mesh_xmls")

# Create term_trees dict for quick and easy lookup later
term_trees = {uids[idx]:trees[idx] for idx in range(len(uids))}

#########################################
# commented out for run
#########################################
term_counts = {uid:0 for uid in uids}

# Count MeSH terms
for doc in tqdm(docs):
    with open("./mesh_xmls/{}".format(doc), "r") as handle:
        soup = BeautifulSoup(handle.read())
        
        mesh_terms = []
                        
        for mesh_heading in soup.find_all("meshheading"):
            if mesh_heading.descriptorname is not None:
                term_id = mesh_heading.descriptorname['ui']
                term_counts[term_id] += 1
##########################################
##########################################

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# Delete eventually
#with open ("./data/mesh_term_doc_counts.csv", "w") as out:
#    for term in term_counts.items():
#        out.write("".join([term[0], ",", str(term[1]), "\n"]))
#
term_counts = {}
with open("./data/mesh_term_doc_counts.csv", "r") as handle:
    for line in handle:
        line = line.strip("\n").split(",")
        term_counts[line[0]] = int(line[1])
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        
# what is the deepest tree?
#longest = 0
#longest_doc = ""
#for doc in doc_tree.keys():
#    for tree in doc_tree[doc]:
#        if len(tree.split(".")) > longest:
#            longest = len(tree.split("."))
#            longest_doc = doc
            
term_freqs = {uid:-1 for uid in uids}

# First let's get the obvious leaf nodes out of the way, don't need to do this
# recursively
#for doc in term_freqs.keys():
#    for tree in doc_tree[doc]:
#        if len(tree.split(".")) == 13:
#            term_freqs[doc] = term_counts[doc]
# memoize to avoid recomputation?

def get_children(uid):
    children = []
    #tree = 
    for tree in term_trees[uid]:
        parent_depth = len(tree.split("."))
        for key, vals in term_trees.items():
            for val in vals:
                child_depth = len(val.split("."))
                if tree in val and uid != key and child_depth == parent_depth + 1:
                    children.append(key)
    
    return list(dict.fromkeys(children))

def freq(uid):
    #total=0
    total = term_counts[uid]
    if term_freqs[uid] != -1:
        return term_freqs[uid]
    if len(get_children(uid)) == 0:
        #return term_counts[uid]
        return total
    else:
        for child in get_children(uid):
            total += freq(child)
        return total
        #return total

for doc in term_freqs.keys():
    term_freqs[doc] = freq(doc)

###################################################
with open ("./data/mesh_term_freq_vals.csv", "w") as out:
    for term in term_freqs.items():
        out.write("".join([term[0], ",", str(term[1]), "\n"]))

term_freqs = {}
with open("./data/mesh_term_freq_vals.csv", "r") as handle:
    for line in handle:
        line = line.strip("\n").split(",")
        term_freqs[line[0]] = int(line[1])
#################################################

# Get all root UIDs
roots = {}
for term in term_trees:
    for tree in term_trees[term]:
        if len(tree.split(".")) == 1:
            roots[tree] = term

# Get term probs
term_probs = {uid:-1 for uid in uids}
for term in term_probs.keys():
    term_roots = [tree.split(".")[0] for tree in term_trees[term]]
    
    probs = []
    for root in term_roots:
        if term_freqs[roots[root]] != 0:
            probs.append(term_freqs[term] / term_freqs[roots[root]])
        elif term_freqs[term] == 0 and term_freqs[roots[root]] == 0:
            probs.append(0)
        elif term_freqs[term] != 0 and term_freqs[roots[root]] == 0:
            logger.error("Bad div by 0: {}".format(term))
    term_probs[term] = sum(probs) / len(probs)
    
ics = {uid:-1 for uid in uids}
for term in ics.keys():
    if term_probs[term] != 0:
        ics[term] = -1 * log(term_probs[term])
    else:
        ics[term] = np.NaN
    