#!/usr/bin/env python3

import os
import math
import logging

import numpy as np
from bs4 import BeautifulSoup
from tqdm import tqdm

# Set up logging
#logging.basicConfig(filename="errors.log", level=logging.ERROR,
#                    format="%(asctime)s - %(levelname)s - %(modules)s: %(funcName)s - %(message)s",
#                    datefmt="%d %b, %H%M")
logging.basicConfig(filename="errors.log", level=logging.INFO,
                    format="Compute term AIC: %(levelname)s - %(message)s")
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

# Create term_trees dict and reverse for quick and easy lookup later
term_trees = {uids[idx]:trees[idx] for idx in range(len(uids))}
term_trees_rev = {tree:uids[idx] for idx in range(len(uids)) for tree in trees[idx]}

#########################################
# commented out for run
#########################################
#term_counts = {uid:0 for uid in uids}
#
## Count MeSH terms
#for doc in tqdm(docs):
#    with open("./mesh_xmls/{}".format(doc), "r") as handle:
#        soup = BeautifulSoup(handle.read())
#        
#        mesh_terms = []
#                        
#        for mesh_heading in soup.find_all("meshheading"):
#            if mesh_heading.descriptorname is not None:
#                term_id = mesh_heading.descriptorname['ui']
#                term_counts[term_id] += 1
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

def get_children(uid):
    # Return empty list for terms (like 'D005260' - 'Female') that aren't
    # actually part of any trees
    if len(term_trees[uid][0]) == 0:
        return []
    
    children = []

    for tree in term_trees[uid]:
        parent_depth = len(tree.split("."))
        for key, vals in term_trees.items():
            for val in vals:
                child_depth = len(val.split("."))
                if tree in val and uid != key and child_depth == parent_depth + 1:
                    children.append(key)
    
    return list(dict.fromkeys(children))

def freq(uid):
    total = term_counts[uid]
    if term_freqs[uid] != -1:
        return term_freqs[uid]
    if len(get_children(uid)) == 0:
        return total
    else:
        for child in get_children(uid):
            total += freq(child)
        return total

term_freqs = {uid:-1 for uid in uids}
for term in term_freqs.keys():
    term_freqs[term] = freq(term)

###################################################
#with open ("./data/mesh_term_freq_vals.csv", "w") as out:
#    for term in term_freqs.items():
#        out.write("".join([term[0], ",", str(term[1]), "\n"]))
#
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

# Computing aggregate information content is done in a step-by-step
# process here to make it easy to follow along. I used Song, Li, Srimani,
# Yu, and Wang's paper, "Measure the Semantic Similarity of GO Terms Using
# Aggregate Information Content" as a guide
            
# Get term probs
term_probs = {uid:-1 for uid in uids}
for term in term_probs:
    term_roots = [tree.split(".")[0] for tree in term_trees[term]]
    
    probs = []
    for root in term_roots:
        try:
            if term_freqs[roots[root]] != 0:
                probs.append(term_freqs[term] / term_freqs[roots[root]])
            elif term_freqs[term] == 0 and term_freqs[roots[root]] == 0:
                probs.append(0)
        except ZeroDivisionError:
            logger.error("Bad div by 0 at term_probs compute step: {}".format(term))
    term_probs[term] = sum(probs) / len(probs)

# Compute IC values
ics = {uid:np.NaN for uid in uids}
for term in ics:
    if term_probs[term] != 0:
        ics[term] = -1 * math.log(term_probs[term])

# Compute knowledge for each term
knowledge = {uid:np.NaN for uid in uids}
for term in knowledge:
    try:
        if ics[term] is not np.NaN:
            knowledge[term] = 1 / ics[term]
    except ZeroDivisionError:
        logger.error("Bad div by 0 at knowledge compute step: {}".format(term))
        
# Compute semantic weight for each term
sws = {uid:np.NaN for uid in uids}
for term in sws:
    if knowledge[term] is not np.NaN:
        sws[term] = 1 / (1 + math.exp(-1 * knowledge[term]))
    
# Compute semantic value for each term by adding the semantic weights
# of all its ancestors
svs = {uid:np.NaN for uid in uids}
for term in svs:
    sv = 0
    ancestors = [".".join(tree.split(".")[:-1]) for tree in term_trees[term]]
    ancestors = [ancestor for ancestor in ancestors if ancestor]
    while ancestors:
        sv += sws[term_trees_rev[ancestors[0]]]
        ancestors.extend([".".join(tree.split(".")[:-1]) for tree in term_trees[term_trees_rev[ancestors[0]]]])
        ancestors = [ancestor for ancestor in ancestors if ancestor]
        ancestors = ancestors[1:]
    svs[term] = sv