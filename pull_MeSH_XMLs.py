import xmltodict
from Bio import Entrez
import pandas as pd
import time
import os
from tqdm import tqdm
from pathlib import P

#os.chdir('/Users/wigasper/Documents/Research Project')
os.chdir('/home/wkg/Documents/Research Project')

edge_list = pd.read_csv("edge_list.csv", index_col=None)

ids_to_get = edge_list['1'].tolist()

# Drop duplicates:
ids_to_get = list(dict.fromkeys(ids_to_get))

# pm_errors stores any errors from PubMed's side
pm_errors = []

for ID in tqdm(ids_to_get):
    Entrez.email = "kgasper@unomaha.edu"
    handle = Entrez.efetch(db="pubmed", id=ID, retmode="xml")
    xmlString = handle.read()
    element = xmltodict.parse(xmlString)

    pm_error = False

    # Check for an error on PMC's side and record it
    for key in element['PubmedArticleSet'].keys():
        if key == 'error':
            pm_errors.append(ID)
    if not pm_error:
        file_out = open("./MeSH XMLs/{}.xml".format(ID), "x")
        file_out.write(xmlString)

    # This is a delay in accordance with PubMed API usage guidelines.
    time.sleep(.3)
