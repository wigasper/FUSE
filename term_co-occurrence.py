#!/usr/bin/env python3
import os
import re
import time
import json
import logging
import traceback

import numpy as np
from tqdm import tqdm

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("./logs/term_co-occurrence.log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Compile regexes
pm_article_start = re.compile(r"\s*<PubmedArticle>")
pm_article_stop = re.compile(r"\s*</PubmedArticle>")
pmid = re.compile(r"\s*<PMID.*>(\d*)</PMID>")
mesh_list_start = re.compile(r"\s*<MeshHeadingList>")
mesh_list_stop = re.compile(r"\s*</MeshHeadingList>")
mesh_term_id = re.compile(r'\s*<DescriptorName UI="(D\d+)".*>')

# Get docs list, initialize variables
docs = os.listdir("./pubmed_bulk")
doc_terms = {}
doc_pmid = ""
term_ids = []

logger.info("Starting doc/term counting")
for doc in tqdm(docs):
    try:
        with open(f"./pubmed_bulk/{doc}", "r") as handle:
            start_doc_count = len(doc_terms.keys())
            start_time = time.perf_counter()

            line = handle.readline()
            while line:
                if pm_article_start.search(line):
                    if doc_pmid:
                        doc_terms[doc_pmid] = term_ids
                        doc_pmid = ""
                        term_ids = []
                    while not pm_article_stop.search(line):
                        if not doc_pmid and pmid.search(line):
                            doc_pmid = pmid.search(line).group(1)
                        if mesh_list_start.search(line):
                            while not mesh_list_stop.search(line):
                                if mesh_term_id.search(line):
                                    term_ids.append(mesh_term_id.search(line).group(1))
                                line = handle.readline()
                        line = handle.readline()
                line = handle.readline()
            doc_terms[doc_pmid] = term_ids

            # Get count for log
            docs_counted = len(doc_terms.keys()) - start_doc_count
            # Get elapsed time and truncate for log
            elapsed_time = int((time.perf_counter() - start_time) * 10) / 10.0
            logger.info(f"{doc} parsing completed - terms extracted for {docs_counted} documents in {elapsed_time} seconds")
            
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(repr(e))
        logger.critical(trace)

logger.info("Stopping doc/term counting")

###########
with open("./data/pm_bulk_doc_term_counts.json", "w") as out:
    json.dump(doc_terms, out)


########### test area
########### need to convert this to csv and read to lists
with open("./data/pm_bulk_doc_term_counts.json", "r") as handle:
    doc_terms = json.load(handle)
##########
with open("./data/pm_bulk_doc_term_counts.csv", "w") as out:
    for doc in doc_terms:
        out.write("".join([doc, ","]))
        out.write(",".join(doc_terms[doc]))
        out.write("\n")

# build matrix
uids = []

with open("./data/mesh_data.tab", "r") as handle:
    for line in handle:
        line = line.strip("\n").split("\t")
        uids.append(line[0])

td_matrix = []

with open("./data/pm_bulk_doc_term_counts.csv", "r") as handle:
    for line in handle:
        line = line.strip("\n").split(",")
        terms = line[1:]
        row = []
        for uid in uids:
            if uid in terms:
                row.append(1)
            else:
                row.append(0)
        td_matrix.append(row)

###############
# build matrix
uids = []

with open("./data/mesh_data.tab", "r") as handle:
    for line in handle:
        line = line.strip("\n").split("\t")
        uids.append(line[0])

td_matrix = []

for doc in doc_terms:
    row = []
    for uid in uids:
        if uid in doc_terms[doc]:
            row.append(1)
        else:
            row.append(0)
    td_matrix.append(row)

td_matrix = np.array(td_matrix)
co_matrix = np.dot(td_matrix.transpose(), td_matrix)

