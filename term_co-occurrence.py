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

# Compile regexes:
pm_article_start = re.compile("\w*<PubmedArticle>")
pm_article_stop = re.compile("\w*</PubmedArticle>")
pmid = re.compile("\w*<PMID.*>(\d*)</PMID>")
mesh_list_start = re.compile("\w*<MeshHeadingList>")

docs = os.listdir("./pubmed_bulk")

doc_terms = {}

pmid = ""
term_ids = []

logger.info("Starting doc/term counting")
for doc in tqdm(docs):
    try:
        with open(f"./pubmed_bulk/{doc}", "r") as handle:
            for line in handle:
                if pm_article_start.search(line):
                    if pmid:
                        doc_terms[pmid] = term_ids
                        pmid = ""
                        term_ids = []
                while not pm_article_stop.search(line):
                    if pmid.search(line):
                        pmid = pmid.search(line).group(1)

    except Exception as e:
        trace = traceback.format_exc()
        logger.error(repr(e))
        logger.critical(trace)
        
with open("./data/pm_bulk_doc_term_counts.json", "w") as out:
    json.dump(doc_terms, out)