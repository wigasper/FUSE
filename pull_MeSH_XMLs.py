import xmltodict
from Bio import Entrez
import pandas as pd
import time
import os
from tqdm import tqdm

#os.chdir('/Users/wigasper/Documents/Research Project')
os.chdir('/home/wkg/Documents/Research Project')

mti_train = pd.read_csv("./FUSE/2013_MTI_in_OA_train.csv", index_col=None)
mti_train.PMID = mti_train.PMID.astype(int).astype(str)

ids_to_get = mti_train.PMID.tolist()

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
        file_out = open("./MeSH XMLs/{}.xml".format(ID), "w")
        file_out.write(xmlString)

    # This is a delay in accordance with PubMed API usage guidelines.
    # It should not be set lower than .34.
    time.sleep(.4)