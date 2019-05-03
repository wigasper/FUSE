import time
import os
from pathlib import Path

import xmltodict
from Bio import Entrez
import pandas as pd
from tqdm import tqdm

edge_list = pd.read_csv("./data/edge_list.csv", index_col=None)

ids_to_get = edge_list['1'].tolist()

# Drop duplicates:
ids_to_get = list(dict.fromkeys(ids_to_get))

# pm_errors stores any errors from PubMed's side
pm_errors = []
fnfe= []

for ID in tqdm(ids_to_get):
    try:
        start_time = time.perf_counter()
        file = Path("./MeSH XMLs/{}.xml".format(ID))
    
        if not file.exists():
            Entrez.email = "kgasper@unomaha.edu"
            handle = Entrez.efetch(db="pubmed", id=ID, retmode="xml")
            xmlString = handle.read()
            element = xmltodict.parse(xmlString)
        
            pm_error = False
        
            # Check for an error on PMC's side and record it
            if isinstance(element['PubmedArticleSet'], dict):
                for key in element['PubmedArticleSet'].keys():
                    if key == 'error':
                        pm_errors.append(ID)
                        pm_error = True
                if not pm_error:
                    with open("./MeSH XMLs/{}.xml".format(ID), "w") as file_out:
                        file_out.write(xmlString)
            if not isinstance(element['PubmedArticleSet'], dict):
                pm_errors.append(ID)
            # This is a delay in accordance with PubMed API usage guidelines.
            if time.perf_counter() - start_time < .4:
                time.sleep(.4 - (time.perf_counter() - start_time))
    except FileNotFoundError:
        fnfe.append(ID)