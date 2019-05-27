#!/usr/bin/env python3

import time
import logging
from pathlib import Path

import xmltodict
from Bio import Entrez
from tqdm import tqdm

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("pubmed_api_pull.log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger = logging.getLogger()

with open("../ncbi.key") as handle:
    api_key = handle.read()

ids_to_get = []

with open("../data/edge_list.csv", "r") as handle:
    for line in handle:
        line = line.strip("\n").split(",")
        ids_to_get.append(line[0])
        ids_to_get.append(line[1])

# Drop duplicates:
ids_to_get = list(dict.fromkeys(ids_to_get))

for pmid in ids_to_get:
    start_time = time.perf_counter()
    file = Path(f"../mesh_xmls/{pmid}.xml")

    if not file.exists():
        Entrez.email = "kgasper@unomaha.edu"
        Entrez.api_key = api_key
        handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml")
        xmlString = handle.read()
        element = xmltodict.parse(xmlString)
    
        pm_error = False
    
        # Check for an error on PubMed's side and record it
        if isinstance(element['PubmedArticleSet'], dict):
            for key in element['PubmedArticleSet'].keys():
                if key == 'error':
                    logger.error(f"PubMed API - ID: {pmid}")
                    pm_error = True
            if not pm_error:
                with open(f"../mesh_xmls/{pmid}.xml", "w") as file_out:
                    file_out.write(xmlString)
        if not isinstance(element['PubmedArticleSet'], dict):
            logger.error(f"Not dict - ID: {pmid}")
            
        # This is a delay in accordance with PubMed API usage guidelines
        if time.perf_counter() - start_time < .1:
            time.sleep(.1 - (time.perf_counter() - start_time))
