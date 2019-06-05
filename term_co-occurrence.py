#!/usr/bin/env python3
import os
import re
import json
import logging
import traceback

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
            logger.info(f"{doc} parsing started")
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
            logger.info(f"{doc} parsing completed - terms extracted for {len(doc_terms.keys()) - start_doc_count} documents")
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(repr(e))
        logger.critical(trace)
        
with open("./data/pm_bulk_doc_term_counts.json", "w") as out:
    json.dump(doc_terms, out)