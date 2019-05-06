#!/usr/bin/env python3

import logging
import json

from bs4 import BeautifulSoup
from tqdm import tqdm

# Set up logging
logging.basicConfig(filename="errors.log", level=logging.INFO,
                    format="MeSH Extract: %(levelname)s - %(message)s")
logger = logging.getLogger()

doc_refs_dict = {}

ids_to_get = []

with open("./data/edge_list.csv", "r") as handle:
    for line in handle:
        line = line.strip("\n").split(",")
        
        if line[0] not in doc_refs_dict.keys():
            doc_refs_dict[line[0]] = []
        
        doc_refs_dict[line[0]].append(line[1])
        
        # Term extraction is the time bottleneck, with a little extra work
        # they can be put into their own list, delete duplicates, create a dict
        # to save some minutes
        ids_to_get.append(line[1])

# Drop duplicates
ids_to_get = list(dict.fromkeys(ids_to_get))

# Create a dict to store the MeSH terms for each PMID
doc_term_dict = {}

# Extract MeSH terms for each PMID
for pmid in tqdm(ids_to_get):
    try:
        with open("./MeSH XMLs/{}.xml".format(pmid), "r") as handle:
            soup = BeautifulSoup(handle.read())
            
            mesh_terms = []
                            
            for mesh_heading in soup.find_all("meshheading"):
                if mesh_heading.descriptorname is not None:
                    term_id = mesh_heading.descriptorname['ui']
                    mesh_terms.append(term_id)

            doc_term_dict[pmid] = mesh_terms
            
    except FileNotFoundError:
        logger.error("FNFE: {}".format(str(pmid)))

# Get term counts for references of each parent node
term_counts = []

for doc in doc_refs_dict.keys():
    try:
        doc_counts = {}
        for ref in doc_refs_dict[doc]:
            for term in doc_term_dict[ref]:
                if term not in doc_counts.keys():
                    doc_counts[term] = 1
                else:
                    doc_counts[term] += 1
                    
        term_counts.append([doc, doc_counts])
    except KeyError:
        logger.error("KeyError at counts - PMID: {}".format(str(ref)))

# Change counts to relative frequency
for doc in term_counts:
    total_count = 0
    for term_id in doc[1].keys():
        total_count += doc[1][term_id]
    for term_id in doc[1].keys():
        doc[1][term_id] = doc[1][term_id] / total_count

with open("./data/train_term_counts.json", "w") as out:
    json.dump(term_counts, out)
        
