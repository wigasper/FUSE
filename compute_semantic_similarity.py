#!/usr/bin/env python3

import os
import math
import logging
import traceback
from itertools import combinations

import numpy as np
from bs4 import BeautifulSoup
from tqdm import tqdm

# Gets a list of children for a term. Because we we don't actually have a graph
# to traverse, it is done by searching according to position on the graph
def get_children(uid, term_trees):
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

# Recursively computes the frequency according to Song et al by adding
# the term's count to sum of the frequencies of all its children
def freq(uid, term_counts, term_freqs, term_trees):
    total = term_counts[uid]
    if term_freqs[uid] != -1:
        return term_freqs[uid]
    if len(get_children(uid, term_trees)) == 0:
        return total
    else:
        for child in get_children(uid):
            total += freq(child)
        return total

# Get all ancestors of a term
def get_ancestors(uid, term_trees, term_trees_rev):
    ancestors = [tree for tree in term_trees[uid]]
    # Remove empty strings if they exist
    ancestors = [ancestor for ancestor in ancestors if ancestor]
    idx = 0
    while idx < len(ancestors):
        ancestors.extend([".".join(tree.split(".")[:-1]) for tree in term_trees[term_trees_rev[ancestors[idx]]]])
        ancestors = [ancestor for ancestor in ancestors if ancestor]
        ancestors = list(dict.fromkeys(ancestors))
        idx += 1
    ancestors = [term_trees_rev[ancestor] for ancestor in ancestors]
    ancestors = list(dict.fromkeys(ancestors))
    return ancestors

# Compute semantic similarity for 2 terms
def semantic_similarity(uid1, uid2, sws, svs):
    uid1_ancs = get_ancestors(uid1, term_trees, term_trees_rev)
    uid2_ancs = get_ancestors(uid2, term_trees, term_trees_rev)
    intersection = [anc for anc in uid1_ancs if anc in uid2_ancs]
    num = sum([(2 * sws[term]) for term in intersection])
    denom = svs[uid1] + svs[uid2]
    
    return 0 if num is np.NaN or denom is 0 else num / denom

# Set up logging
# As the implementation stands, logging is not terribly important, but it is 
# very helpful during development. I keep this around though in case it is 
# needed in the future.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("compute_semantic_similarity.log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

uids = []
names = []
trees = []

with open("./data/mesh_data.tab", "r") as handle:
    for line in handle:
        line = line.strip("\n").split("\t")
        uids.append(line[0])
        names.append(line[1])
        trees.append(line[4].split(","))

docs = os.listdir("./pubmed_bulk")

# Create term_trees dict and reverse for quick and easy lookup later
term_trees = {uids[idx]:trees[idx] for idx in range(len(uids))}
term_trees_rev = {tree:uids[idx] for idx in range(len(uids)) for tree in trees[idx]}

term_counts = {uid:0 for uid in uids}

# Count MeSH terms
for doc in tqdm(docs):
    with open("./pubmed_bulk/{}".format(doc), "r") as handle:
        soup = BeautifulSoup(handle.read())
        
        mesh_terms = []
                        
        for mesh_heading in soup.find_all("meshheading"):
            if mesh_heading.descriptorname is not None:
                term_id = mesh_heading.descriptorname['ui']
                term_counts[term_id] += 1
    
term_freqs = {uid:-1 for uid in uids}
for term in term_freqs.keys():
    term_freqs[term] = freq(term, term_counts, term_freqs, term_trees)

root_freq = sum(term_freqs.values())

# Computing aggregate information content is done in a step-by-step
# process here to make it easy to follow along. I used Song, Li, Srimani,
# Yu, and Wang's paper, "Measure the Semantic Similarity of GO Terms Using
# Aggregate Information Content" as a guide
            
# Get term probs
term_probs = {uid:-1 for uid in uids}
for term in term_probs:
    term_probs[term] = term_freqs[term] / root_freq

# Compute IC values
ics = {uid:np.NaN for uid in uids}
for term in ics:
    if term_probs[term] != 0:
        ics[term] = -1 * math.log(term_probs[term])

# Compute knowledge for each term
knowledge = {uid:np.NaN for uid in uids}
for term in knowledge:
    knowledge[term] = 1 / ics[term]
        
# Compute semantic weight for each term
sws = {uid:np.NaN for uid in uids}
for term in sws:
    sws[term] = 1 / (1 + math.exp(-1 * knowledge[term]))
    
# Compute semantic value for each term by adding the semantic weights
# of all its ancestors
svs = {uid:np.NaN for uid in uids}
for term in svs:
    sv = 0
    ancestors = get_ancestors(term, term_trees, term_trees_rev)
    for ancestor in ancestors:
        sv += sws[ancestor]
    svs[term] = sv

# Compute semantic similarity for each pair
pairs = {}
logger.info("Semantic similarity compute start")
for pair in combinations(uids, 2):
    try:
        with open("./data/semantic_similarities_rev0.csv", "a") as out:
            out.write("".join([pair[0], ",", pair[1], ",", str(semantic_similarity(pair[0], pair[1], sws, svs)), "\n"]))
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(repr(e))
        logger.critical(trace)
logger.info("Semantic similarity compute end")
