#!/usr/bin/env python3

import os

from bs4 import BeautifulSoup
from tqdm import tqdm

uids = []
with open("./data/mesh_data.tab", "r") as handle:
    for line in handle:
        line = line.split("\t")
        uids.append(line[0])

docs = os.listdir("./mesh_xmls")

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

with open ("./data/mesh_term_doc_counts.csv", "w") as out:
    for term in term_counts.items():
        out.write("".join([term[0], ",", str(term[1]), "\n"]))