#!/usr/bin/env python3
import os
import json
import logging
import traceback

from tqdm import tqdm
from bs4 import BeautifulSoup

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("term_co-occurrence.log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

uids = []

with open("./data/mesh_data.tab", "r") as handle:
    for line in handle:
        line = line.strip("\n").split("\t")
        uids.append(line[0])

docs = os.listdir("./pubmed_bulk")

doc_terms = {}

for doc in tqdm(docs):
    try:
        with open("./pubmed_bulk/{}".format(doc), "r") as handle:
            soup = BeautifulSoup(handle.read())
            for article in soup.find_all("pubmedarticle"):
                term_ids = []
                for mesh_heading in article.find_all("meshheading"):
                    if mesh_heading.descriptorname is not None:
                        term_ids.append(mesh_heading.descriptorname['ui'])
                doc_terms[article.pubmeddata.articleidlist.articleid.string] = term_ids
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(repr(e))
        logger.critical(trace)
        
with open("./data/pm_bulk_doc_term_counts.json", "w") as out:
    json.dump(doc_terms, out)